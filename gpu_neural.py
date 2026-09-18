"""Optional transactional CUDA epoch backend; CPU state remains authoritative.

Sparse float64 atomic accumulation is guarded by an exact-sum bit bound.
No reduced precision, fast math, or GPU RNG are used.
"""
from __future__ import annotations
import copy
import math
import time
import numpy as np


def cuda_kernels():
    from numba import cuda, float32, float64
    from numba.cuda import libdevice

    @cuda.jit
    def weights(edge_ids, base, mod_base, scale, fast, slow):
        e=cuda.grid(1)
        if e<edge_ids.size:
            s=scale[edge_ids[e]]
            fast[e]=libdevice.fmul_rn(base[e],s)
            slow[e]=libdevice.fmul_rn(mod_base[e],s)

    @cuda.jit
    def deliver(ptr, targets, fast, slow, mod_row, active, active_counts, initial, step, acc, touched):
        tid=cuda.grid(1);lane=tid%32;warp=tid//32;warps=cuda.gridsize(1)//32
        slot=(initial[0]+step)%active_counts.size
        for j in range(warp,active_counts[slot],warps):
            pre=active[slot,j];row=mod_row[pre]
            for e in range(ptr[pre]+lane,ptr[pre+1],32):
                target=targets[e];val=fast[e]
                if val!=0.:
                    cuda.atomic.add(acc,(0,target),float64(val))
                    cuda.atomic.or_(touched,target,1)
                strength=slow[e]
                if row>=0 and strength!=0.:
                    cuda.atomic.add(acc,(row+1,target),float64(strength))

    @cuda.jit
    def clear_count(active_counts,initial,step,delay):
        if cuda.grid(1)==0:
            active_counts[(initial[0]+step+delay)%active_counts.size]=0

    @cuda.jit
    def tick(accum,touched,v,g,mod,refr,ring,active,active_counts,initial,delay,
             external,step,eligible,impulses,dec,rest,em,couple,eg,threshold,reset,forced,ref_steps,counts):
        i=cuda.grid(1)
        if i>=v.size:return
        acc=accum[0,i];m0=accum[1,i];m1=accum[2,i];m2=accum[3,i]
        seen=touched[i]!=0;s0=m0!=0.;s1=m1!=0.;s2=m2!=0.
        for j in range(4):accum[j,i]=float64(0.)
        touched[i]=0
        gi=g[i]
        if seen:gi=libdevice.fadd_rn(gi,float32(acc))
        a=mod[0,i];b=mod[1,i];c=mod[2,i]
        if s0:a=libdevice.fadd_rn(a,float32(m0))
        if s1:b=libdevice.fadd_rn(b,float32(m1))
        if s2:c=libdevice.fadd_rn(c,float32(m2))
        mod[0,i]=libdevice.fmul_rn(a,dec[0])
        mod[1,i]=libdevice.fmul_rn(b,dec[1])
        mod[2,i]=libdevice.fmul_rn(c,dec[2])
        vi=libdevice.fadd_rn(libdevice.fadd_rn(rest,libdevice.fmul_rn(libdevice.fsub_rn(v[i],rest),em)),libdevice.fmul_rn(gi,couple))
        gi=libdevice.fmul_rn(gi,eg)
        if eligible[i] and impulses[step]!=0.:
            vi=libdevice.fadd_rn(vi,impulses[step])
        if external[step,i]:vi=forced
        fired=vi>=threshold and refr[i]<=0
        dest=(initial[0]+step+delay)%ring.shape[0]
        ring[dest,i]=fired
        ri=refr[i]
        if fired:
            index=cuda.atomic.add(active_counts,dest,1)
            active[dest,index]=i
            counts[i]+=1;vi=reset;gi=float32(0.);ri=ref_steps
        if ri>0:
            ri-=1
            if ri>0:vi=reset
        v[i]=vi;g[i]=gi;refr[i]=ri
    return weights,deliver,clear_count,tick


class CudaEpoch:
    def __init__(self,core):
        from numba import cuda
        if not cuda.is_available():raise RuntimeError('CUDA unavailable')
        self.cuda=cuda;self.n=core.n;self.edges=core.edge_count
        self.ptr=cuda.to_device(core.indptr)
        self.src=cuda.to_device(core.indices)
        self.edge_ids=cuda.to_device(np.arange(core.edge_count,dtype=np.int32))
        self.base=cuda.to_device(core.w0)
        mod_base=core._delivery_log_syn_lut[core.synapse_count]*np.float32(core.config.modulator_edge_scale)
        self.mod_base=cuda.to_device(mod_base.astype(np.float32))
        self.mod_scale=core.config.modulator_edge_scale
        rows=np.full(core.n,-1,np.int8)
        for j,code in enumerate((core.NT_DA,core.NT_OA,core.NT_5HT)):rows[core.nt_code==code]=j
        self.rows=cuda.to_device(rows)
        self.fast=cuda.device_array(core.edge_count,np.float32);self.slow=cuda.device_array(core.edge_count,np.float32)
        self.weights,self.deliver,self.clear_count,self.tick=cuda_kernels()
        self.max_indegree=int(np.bincount(core.indices,minlength=core.n).max())
        self.ranges=[]
        for values in (core.w0,mod_base):
            positive=np.abs(values[values!=0.])
            self.ranges.append((float(positive.min()),float(positive.max())) if len(positive) else (0.,0.))
        self.last_wall_s=None
        self.buffers=None;self.graph=None;self.graph_disabled=False
        self._configuration_cache={}
        self.stream=cuda.stream()

    def run(self,core,ext_idx,ext_p,steps):
        """Return uncommitted state. Failures never mutate core or its RNG."""
        if core.n!=self.n or core.edge_count!=self.edges:raise ValueError('GPU graph changed')
        started=time.perf_counter();cuda=self.cuda;c=core.config;n=core.n
        if c.modulator_edge_scale!=self.mod_scale:raise ValueError('GPU modulation configuration changed')
        scale_min=float(core._delivery_scale.min());scale_max=float(core._delivery_scale.max())
        if not (scale_min>0 and math.isfinite(scale_max)):raise ValueError('Scale range requires CPU accumulation')
        # Every float32 contribution is an integer multiple of this smallest ULP.
        # If the absolute sum occupies <=53 bits, float64 addition is exact in ANY
        # order, including concurrent atomics. The bound covers all active subsets
        # and cancellation, and is checked on each epoch's actual scale range.
        for low,high in self.ranges:
            if high==0:continue
            lower=low*scale_min*(1.-2.**-23);upper=high*scale_max*(1.+2.**-23)
            quantum_exp=math.frexp(lower)[1]-24
            if math.log2(upper*max(1,self.max_indegree))-quantum_exp>52:
                raise ValueError('Exact GPU sum bound exceeded; use ordered CPU path')
        rng=copy.deepcopy(core.rng)
        external=np.zeros((steps,n),dtype=np.bool_)
        for step in range(steps):
            if len(ext_idx):external[step,ext_idx[rng.random(len(ext_idx))<ext_p]]=True
        # Preserve the exact CPU scalar clock and phase math.
        wave=copy.copy(core);wave._keepalive_bias=core._keepalive_bias.copy()
        impulses=np.zeros(steps,np.float32)
        for step in range(steps):
            if wave._keepalive_ticks_until_update<=0:
                wave._refresh_keepalive_bias()
                ids=np.flatnonzero(wave._keepalive_bias!=0.)
                if len(ids):impulses[step]=wave._keepalive_bias[ids[0]]*np.float32(wave._keepalive_pulse_fraction)
                wave._keepalive_ticks_until_update=wave._keepalive_update_ticks
            wave._keepalive_time_ms+=float(c.dt_ms);wave._keepalive_ticks_until_update-=1
        ring=np.zeros((len(core.delay_ring),n),dtype=np.bool_)
        for j,ids in enumerate(core.delay_ring):
            if len(ids)>1 and np.any(ids[1:]<=ids[:-1]):raise ValueError('Noncanonical delay order requires CPU')
            ring[j,ids]=True
        active=np.zeros((len(ring),n),np.int32)
        active_counts=np.array([len(a) for a in core.delay_ring],np.int32)
        for j,ids in enumerate(core.delay_ring):active[j,:len(ids)]=ids
        host=dict(active=active,active_counts=active_counts,v=core.v,g=core.g,mod=core.mod_state,refr=core.refr,ring=ring,external=external,
                  impulses=impulses,eligible=core._keepalive_eligible,dec=core.mod_decay,
                  counts=np.zeros(n,np.int32),scale=core._delivery_scale,
                  accum=np.zeros((4,n),np.float64),touched=np.zeros(n,np.int32),
                  initial=np.array([core.ring_pos],np.int32))
        config=(steps,len(ring),c.v_rest,float(core.em),float(core.couple),float(core.eg),c.v_thresh,c.v_reset,core.ref_steps,core.delay_steps)
        if self.buffers is None or self.buffer_config!=config:
            # Uneven integer-tick slices (83/84 ticks for 100 ms / 12) alternate
            # two shapes. Retain both allocations/graphs instead of recapturing
            # thousands of CUDA launches at every window boundary.
            if self.buffers is not None:
                self._configuration_cache[self.buffer_config]=(self.buffers,self.graph)
            cached=self._configuration_cache.pop(config,None)
            if cached is None:
                self.buffers={k:cuda.device_array(a.shape,a.dtype,stream=self.stream) for k,a in host.items()}
                self.graph=None
            else:
                self.buffers,self.graph=cached
            self.buffer_config=config
            while len(self._configuration_cache)>1:
                key=next(iter(self._configuration_cache))
                _,old_graph=self._configuration_cache.pop(key)
                if old_graph is not None:old_graph.close()
        for k,a in host.items():self.buffers[k].copy_to_device(a,stream=self.stream)
        v=self.buffers['v'];g=self.buffers['g'];mod=self.buffers['mod'];refr=self.buffers['refr']
        dring=self.buffers['ring'];dext=self.buffers['external'];imp=self.buffers['impulses']
        eligible=self.buffers['eligible'];dec=self.buffers['dec'];counts=self.buffers['counts'];scale=self.buffers['scale']
        active=self.buffers['active'];active_counts=self.buffers['active_counts']
        accum=self.buffers['accum'];touched=self.buffers['touched'];initial=self.buffers['initial']
        self.weights[(self.edges+255)//256,256,self.stream](self.edge_ids,self.base,self.mod_base,scale,self.fast,self.slow)
        def args(step):
            return (accum,touched,v,g,mod,refr,dring,active,active_counts,initial,core.delay_steps,
                dext,step,eligible,imp,dec,np.float32(c.v_rest),core.em,core.couple,core.eg,
                np.float32(c.v_thresh),np.float32(c.v_reset),np.float32(c.v_thresh+.5),core.ref_steps,counts)
        if self.graph is None and not self.graph_disabled:
            # Compile before capture. No live state is executed by specialize().
            self.deliver.specialize(self.ptr,self.src,self.fast,self.slow,self.rows,active,active_counts,initial,0,accum,touched)
            self.clear_count.specialize(active_counts,initial,0,core.delay_steps)
            self.tick.specialize(*args(0))
            self.stream.synchronize()
            try:
                from cuda_graph import Graph
                def record():
                    for step in range(steps):
                        self.deliver[64,128,self.stream](self.ptr,self.src,self.fast,self.slow,self.rows,active,active_counts,initial,step,accum,touched)
                        self.clear_count[1,1,self.stream](active_counts,initial,step,core.delay_steps)
                        self.tick[(n+127)//128,128,self.stream](*args(step))
                self.graph=Graph(self.stream,record)
            except Exception:
                self.graph_disabled=True
        if self.graph is not None:
            self.graph.launch(self.stream)
        else:
            for step in range(steps):
                self.deliver[64,128,self.stream](self.ptr,self.src,self.fast,self.slow,self.rows,active,active_counts,initial,step,accum,touched)
                self.clear_count[1,1,self.stream](active_counts,initial,step,core.delay_steps)
                self.tick[(n+127)//128,128,self.stream](*args(step))
        self.stream.synchronize()
        slot=(core.ring_pos+steps)%len(ring)
        result=dict(v=v.copy_to_host(),g=g.copy_to_host(),mod_state=mod.copy_to_host(),refr=refr.copy_to_host(),
                    spike_counts=counts.copy_to_host(),ring=dring.copy_to_host(),ring_pos=slot,
                    rng_state=copy.deepcopy(rng.bit_generator.state))
        result['ring'][(slot-1)%len(ring)]=False
        result['delay_ring']=[np.flatnonzero(a).astype(np.int32) for a in result.pop('ring')]
        for attr in ('_keepalive_bias','_keepalive_time_ms','_keepalive_ticks_until_update','_keepalive_phase','_keepalive_active_neurons'):
            result[attr]=getattr(wave,attr)
        cuda.synchronize();self.last_wall_s=time.perf_counter()-started
        return result

    @staticmethod
    def commit(core,result):
        for attr in ('v','g','mod_state','refr','delay_ring','ring_pos','_keepalive_bias',
                     '_keepalive_time_ms','_keepalive_ticks_until_update','_keepalive_phase','_keepalive_active_neurons'):
            setattr(core,attr,result[attr])
        core.rng.bit_generator.state=result['rng_state']

