#!/usr/bin/env python3
"""Specimen-003 full MaleCNS dynamical core.

Compared with the Specimen-002 core this loader understands the v2 full graph:

* every retained released neuron->neuron edge remains in ``indptr/indices``;
* ``synapse_count`` stores the measured aggregate connection strength;
* classical fast transmitters contribute signed conductance;
* dopamine, octopamine and serotonin edges remain structural and generate slow
  transmitter-specific postsynaptic fields instead of being deleted;
* plasticity can scale both fast and slow connections without changing their
  transmitter identity;
* an optional bounded homeostatic valence can consolidate activity that reduced
  internal error.  This is an engineered learning rule, not a claim of exact fly
  reinforcement physiology.

The class can also load the older 165,122-neuron graph for compatibility tests.
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from neural_stability import repair_compounded_gains, fast_state_sanity

try:
    from numba import njit, prange, set_num_threads as _numba_set_threads
    from numba import get_num_threads as _numba_get_threads
    from numba import config as _numba_config
    _NUMBA_AVAILABLE = True

    @njit(parallel=True, fastmath=False, cache=True, nogil=True)
    def _parallel_integrate(v, g, mod_state, mod_decay, v_rest, em, couple, eg):
        # Every iteration owns one neuron.  No reductions/scatters: changing the
        # worker count cannot change arithmetic order for a neuron.
        for i in prange(v.shape[0]):
            mod_state[0, i] = mod_state[0, i] * mod_decay[0]
            mod_state[1, i] = mod_state[1, i] * mod_decay[1]
            mod_state[2, i] = mod_state[2, i] * mod_decay[2]
            vr = v[i] - v_rest
            v[i] = v_rest + vr * em + g[i] * couple
            g[i] = g[i] * eg

    @njit(fastmath=False, cache=True, nogil=True)
    def _deliver_due_kernel(due, indptr, indices, w0, effective_scale, nt_code, synapse_count,
                            log_syn_lut, modulator_edge_scale, g, mod_state,
                            fast_acc, fast_mark, fast_touched, mod_acc, mod_mark, mod_touched,
                            nt_da, nt_oa, nt_5ht):
        # Exact-order sparse scatter.  The previous NumPy path rebuilt repeat/arange
        # expansions and dense bincount outputs on every 0.1-ms tick.  Here we walk
        # the same due-source order and each CSR edge in the same order, accumulate
        # in float64 like numpy.bincount, cast once to float32, then apply.
        nf = 0
        n0 = 0; n1 = 0; n2 = 0
        ms = np.float32(modulator_edge_scale)
        for ii in range(due.size):
            src = int(due[ii]); code = int(nt_code[src]); row = -1
            if code == nt_da: row = 0
            elif code == nt_oa: row = 1
            elif code == nt_5ht: row = 2
            for ee in range(int(indptr[src]), int(indptr[src + 1])):
                tgt = int(indices[ee]); scale = np.float32(effective_scale[ee])
                val = np.float32(w0[ee]) * scale
                if val != np.float32(0.0):
                    if not fast_mark[tgt]:
                        fast_mark[tgt] = True; fast_touched[nf] = tgt; nf += 1
                    fast_acc[tgt] += float(val)
                if row >= 0:
                    strength = np.float32(log_syn_lut[int(synapse_count[ee])]) * ms * scale
                    if strength != np.float32(0.0):
                        if not mod_mark[row, tgt]:
                            mod_mark[row, tgt] = True
                            if row == 0: mod_touched[row, n0] = tgt; n0 += 1
                            elif row == 1: mod_touched[row, n1] = tgt; n1 += 1
                            else: mod_touched[row, n2] = tgt; n2 += 1
                        mod_acc[row, tgt] += float(strength)
        for j in range(nf):
            tgt = int(fast_touched[j])
            g[tgt] = np.float32(g[tgt] + np.float32(fast_acc[tgt]))
            fast_acc[tgt] = 0.0; fast_mark[tgt] = False
        for row in range(3):
            nm = n0 if row == 0 else (n1 if row == 1 else n2)
            for j in range(nm):
                tgt = int(mod_touched[row, j])
                mod_state[row, tgt] = np.float32(mod_state[row, tgt] + np.float32(mod_acc[row, tgt]))
                mod_acc[row, tgt] = 0.0; mod_mark[row, tgt] = False

    @njit(fastmath=False, cache=True, nogil=True)
    def _fire_reset_kernel(v, g, refr, v_thresh, v_reset, ref_steps,
                           spike_counts, labels, read_counts, fired_out):
        # Preserve np.flatnonzero ascending neuron order while avoiding three full
        # temporary boolean masks per 0.1-ms tick.
        nf = 0
        for i in range(v.shape[0]):
            if v[i] >= v_thresh and refr[i] <= 0:
                fired_out[nf] = i; nf += 1
                spike_counts[i] += 1
                lab = int(labels[i])
                if lab >= 0: read_counts[lab] += 1
                v[i] = v_reset; g[i] = np.float32(0.0); refr[i] = ref_steps
        for i in range(v.shape[0]):
            if refr[i] > 0:
                refr[i] -= 1
                if refr[i] > 0: v[i] = v_reset
        return nf
except Exception:
    _NUMBA_AVAILABLE = False
    _numba_set_threads = None
    _numba_get_threads = None
    _numba_config = None
    _parallel_integrate = None
    _deliver_due_kernel = None
    _fire_reset_kernel = None

# CUDA is deliberately optional.  The first GPU hybrid path only accelerates an
# elementwise plastic-memory forgetting pass whose result can be checked bit-for-bit
# against the CPU implementation before it is ever enabled for a specimen.  The
# chronology-sensitive recurrent delivery loop remains on the CPU.
try:
    if _NUMBA_AVAILABLE:
        from numba import cuda as _numba_cuda
        _CUDA_IMPORT_AVAILABLE = True

        @_numba_cuda.jit
        def _cuda_forget_half(a, factor):
            i = _numba_cuda.grid(1)
            if i < a.size:
                # Match the CPU rule conceptually: float16 -> float32 multiply -> float16.
                a[i] = a[i] * factor
    else:
        _numba_cuda = None
        _CUDA_IMPORT_AVAILABLE = False
        _cuda_forget_half = None
except Exception:
    _numba_cuda = None
    _CUDA_IMPORT_AVAILABLE = False
    _cuda_forget_half = None


def _bounded_thread_search(permitted, measure, scores):
    """Find the fastest measured integer thread count without CPU-topology guesses.

    ``measure(n)`` must return a lower-is-better score and populate ``scores[n]``.
    The search makes +10 coarse jumps, then bisects/exhausts a small bounded
    region around the rollover.  It never assumes physical-core count, SMT ratio,
    NUMA shape, or that the runtime cap itself is optimal.
    """
    permitted=max(1,int(permitted))
    measure(1)
    coarse=[1]
    if permitted >= 2:
        measure(2); coarse.append(2)

    first_regression=None
    while coarse[-1] < permitted and first_regression is None:
        prev=coarse[-1]
        nxt=min(permitted,prev+10)
        if nxt == prev:
            break
        prev_score=float(scores[prev])
        nxt_score=float(measure(nxt)); coarse.append(nxt)
        if nxt_score > prev_score:
            first_regression=nxt
            break

    if first_regression is not None:
        hi=int(first_regression)
        i=coarse.index(hi)
        lo=int(coarse[max(0,i-2)])
    else:
        hi=int(coarse[-1])
        lo=int(coarse[max(0,len(coarse)-3)]) if len(coarse) >= 3 else 1
    if lo > hi: lo,hi=hi,lo
    anchor=hi

    def measured_in_bracket():
        return sorted(t for t in scores if lo <= t <= hi)

    def best_in_bracket():
        pts=measured_in_bracket()
        return min(pts,key=lambda t:float(scores[t]))

    def midpoint(a,b):
        return int((int(a)+int(b))//2)

    measure(lo); measure(hi)
    if hi-lo > 1:
        m=midpoint(lo,hi)
        old_best=best_in_bracket(); old_score=float(scores[old_best])
        measure(m)
        new_best=best_in_bracket()
        if float(scores[new_best]) < old_score and new_best == m and m-1 >= lo:
            measure(m-1)

    while True:
        pts=measured_in_bracket()
        if len(pts) >= (hi-lo+1):
            break
        best=best_in_bracket(); pos=pts.index(best)
        gaps=[]
        if pos > 0 and best-pts[pos-1] > 1:
            gaps.append((best-pts[pos-1], 'low', pts[pos-1], best))
        if pos+1 < len(pts) and pts[pos+1]-best > 1:
            gaps.append((pts[pos+1]-best, 'high', best, pts[pos+1]))
        if not gaps:
            for a,b in zip(pts,pts[1:]):
                if b-a > 1:
                    gaps.append((b-a,'other',a,b))
        if not gaps:
            break
        gaps.sort(key=lambda x:(x[0], x[1]=='high'), reverse=True)
        _,side,a,b=gaps[0]
        m=midpoint(a,b)
        if m <= a: m=a+1
        if m >= b: m=b-1
        before_best=best_in_bracket(); before_score=float(scores[before_best])
        measure(m)
        after_best=best_in_bracket()
        if float(scores[after_best]) < before_score and after_best == m and m-1 >= lo and (m-1) not in scores:
            measure(m-1)

    # Boundary guard: if the best result sits on the artificial bracket edge,
    # widen one +10 block and verify it. Repeat only while the winner remains on
    # that edge. This keeps the bounded search from becoming a hidden CPU guess.
    while True:
        best=best_in_bracket()
        if best == lo and lo > 1:
            old_lo=lo; lo=max(1,lo-10)
            for t in range(lo,old_lo): measure(t)
            continue
        if best == hi and hi < permitted:
            old_hi=hi; hi=min(permitted,hi+10)
            for t in range(old_hi+1,hi+1): measure(t)
            continue
        break

    best=min(scores,key=lambda th:float(scores[th]))
    return {
        'best':int(best),
        'coarse_probes':[int(x) for x in coarse],
        'first_regression':None if first_regression is None else int(first_regression),
        'refinement_anchor':int(anchor),
        'search_bracket':(int(lo),int(hi)),
    }


class AdaptiveThreadController:
    """Runtime-only thread governor; thread count never enters specimen state.

    Auto mode is selected once from a deterministic host microbenchmark.  We do
    not change Numba's worker count while a specimen is running; some runtimes
    dislike repeatedly resizing native worker pools.
    """
    def __init__(self):
        try:
            affinity = len(os.sched_getaffinity(0))
        except Exception:
            affinity = int(os.cpu_count() or 1)
        numba_cap = int(getattr(_numba_config, 'NUMBA_NUM_THREADS', affinity)) if _NUMBA_AVAILABLE else 1
        self.permitted = max(1, min(int(affinity), int(numba_cap)))
        self.backend = 'numba-parallel-exact' if _NUMBA_AVAILABLE else 'numpy-single-thread'
        self.mode = 'auto' if _NUMBA_AVAILABLE and self.permitted > 1 else 'manual'
        # `candidates` becomes the actual ordered probe trail from the last
        # Auto tune, rather than a theoretical powers-of-two list.
        self.candidates = []
        self.current = 1; self.best = 1; self.best_wall = float('inf')
        self.tuning_complete = self.mode != 'auto'
        self.last_epoch_wall = None
        self.benchmark_scores = {}
        self.first_regression_threads = None
        self.refinement_anchor_threads = None
        self.coarse_probes = []
        self.search_bracket = None
        self.tuning_strategy = ('1T baseline; 2T start; +10 coarse jumps until first slowdown/cap; '
                               'then bounded midpoint jumps inside the surrounding coarse bracket; '
                               'verify every integer left in that bracket using warmed median-of-five scores; choose fastest measured count')
        self._apply(1)

    def _apply(self, n):
        n=max(1,min(int(n),self.permitted)); self.current=n
        if _NUMBA_AVAILABLE: _numba_set_threads(n)

    def set_auto_result(self, best, scores, probe_order=None, first_regression=None, refinement_anchor=None, coarse_probes=None, search_bracket=None):
        self.mode='auto'; self.best=int(best); self.best_wall=float(scores.get(best,float('inf')))
        self.benchmark_scores={int(k):float(v) for k,v in scores.items()}
        self.candidates=[int(x) for x in (probe_order or scores.keys())]
        self.first_regression_threads=None if first_regression is None else int(first_regression)
        self.refinement_anchor_threads=None if refinement_anchor is None else int(refinement_anchor)
        self.coarse_probes=[int(x) for x in (coarse_probes or [])]
        self.search_bracket=None if search_bracket is None else [int(search_bracket[0]),int(search_bracket[1])]
        self.tuning_complete=True; self._apply(best)

    def configure_manual(self, n):
        self.mode='manual'; self.tuning_complete=True; self._apply(int(n)); self.best=self.current

    def observe(self, wall_s):
        self.last_epoch_wall=float(wall_s)

    def info(self):
        return {
            'backend':self.backend,'mode':self.mode,'threads':int(self.current),
            'permitted':int(self.permitted),'best_threads':int(self.best),
            'tuning_complete':bool(self.tuning_complete),
            'last_epoch_wall_s':None if self.last_epoch_wall is None else float(self.last_epoch_wall),
            'candidates':list(self.candidates),
            'benchmark_scores':dict(self.benchmark_scores),
            'first_regression_threads':self.first_regression_threads,
            'refinement_anchor_threads':self.refinement_anchor_threads,
            'coarse_probes':list(self.coarse_probes),
            'search_bracket':None if self.search_bracket is None else list(self.search_bracket),
            'tuning_strategy':self.tuning_strategy,
        }


@dataclass
class FullPlasticityConfig:
    # Fast neural dynamics
    v_rest: float = -52.0
    v_thresh: float = -45.0
    v_reset: float = -52.0
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    dt_ms: float = 0.1

    # Existing-edge plasticity
    learning_rate: float = 0.014
    reinforcement_rate: float = 0.010
    forgetting_per_epoch: float = 0.0008
    max_fractional_change: float = 0.35
    min_pre_spikes: int = 2
    min_post_spikes: int = 1
    target_post_activity: float = 0.20
    update_clip_per_epoch: float = 0.010
    familiarity_alpha: float = 0.06
    min_surprise_hz: float = 0.5
    max_plastic_presynaptic_neurons: int = 14000

    # Slow monoaminergic fields.  Time constants and coupling are deliberately
    # explicit engineering parameters, not asserted Drosophila receptor kinetics.
    dopamine_tau_ms: float = 1200.0
    octopamine_tau_ms: float = 700.0
    serotonin_tau_ms: float = 1800.0
    modulator_edge_scale: float = 0.018
    local_modulation_plasticity_gain: float = 0.25

    # Endogenous sleep/state-wave scaffold.  Legacy keepalive_* names are retained
    # solely for checkpoint/UI compatibility.  Patch20 targets the identified ER5
    # (R5 sleep-need) and ExR1 (helicon) networks instead of washing across the CNS.
    # A 1-Hz subthreshold resting-potential oscillation lies inside the measured
    # 0.5-1.5 Hz R5/helicon SWA band.  It never directly drives sensory/motor cells.
    keepalive_wave_enabled: bool = True
    keepalive_wave_fraction: float = 0.03
    keepalive_wave_period_ms: float = 1000.0
    keepalive_wave_half_width: float = 0.035  # legacy/unused after Patch20
    keepalive_wave_update_ms: float = 5.0


@dataclass
class FullEpochReport:
    neural_ms: float
    network_hz: float
    readout_hz: dict[str, float]
    active_neurons: int
    plastic_edges_touched: int
    novelty_hz: float
    familiar_neurons: int
    mean_abs_plastic_delta: float
    max_abs_plastic_delta: float
    homeostatic_valence: float
    dopamine_mean: float
    octopamine_mean: float
    serotonin_mean: float
    keepalive_wave_phase: float
    keepalive_wave_active_neurons: int
    keepalive_wave_peak_mv: float

    def to_dict(self):
        return asdict(self)


class FullMaleCNSCore:
    """Persistent connectome core retaining fast and slow transmitter wiring."""

    NT_UNKNOWN = 0
    NT_ACH = 1
    NT_GABA = 2
    NT_GLUT = 3
    NT_HIST = 4
    NT_DA = 5
    NT_OA = 6
    NT_5HT = 7
    NT_UNCLEAR = 8

    def __init__(self, graph_path: str | Path, *, seed: int = 7, config: FullPlasticityConfig | None = None, auto_tune: bool = True):
        self.graph_path = Path(graph_path)
        self.config = config or FullPlasticityConfig()
        z = np.load(self.graph_path, allow_pickle=True)
        storage = str(z["storage"].item()) if "storage" in z else "legacy-csr-post-rows"

        if storage == "csc-pre-columns-v2":
            self.indptr = np.asarray(z["indptr"], dtype=np.int64)
            self.indices = np.asarray(z["indices"], dtype=np.int32)
            self.w0 = np.asarray(z["data"], dtype=np.float32)
            self.edge_gain = np.asarray(z["edge_gain"], dtype=np.float32) if "edge_gain" in z else np.ones(len(self.w0), dtype=np.float32)
            self.n = int(np.asarray(z["shape"])[0])
            if "synapse_count" in z:
                self.synapse_count = np.asarray(z["synapse_count"], dtype=np.uint32)
            else:
                mv = float(z["mv_per_synapse"]) if "mv_per_synapse" in z else 0.275
                self.synapse_count = np.rint(np.abs(self.w0) / max(mv, 1e-9)).astype(np.uint32)
        else:
            # Specimen-002 compatibility: old graph stored postsynaptic-row CSR.
            W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])).tocsc()
            self.indptr = W.indptr.astype(np.int64, copy=False)
            self.indices = W.indices.astype(np.int32, copy=False)
            self.w0 = W.data.astype(np.float32, copy=False)
            self.edge_gain = np.ones(len(self.w0), dtype=np.float32)
            self.n = int(W.shape[0])
            self.synapse_count = np.maximum(1, np.rint(np.abs(self.w0) / 0.275)).astype(np.uint32)

        self.edge_count = int(len(self.indices))
        if len(self.w0) != self.edge_count or len(self.edge_gain) != self.edge_count or int(self.indptr[-1]) != self.edge_count:
            raise ValueError("graph sparse arrays are inconsistent")

        def arr(name, dtype=str, default=None):
            if name in z:
                return np.asarray(z[name], dtype=dtype)
            if default is None:
                if dtype is str:
                    return np.full(self.n, "", dtype="<U1")
                return np.zeros(self.n, dtype=dtype)
            return np.asarray(default, dtype=dtype)

        self.bodies = arr("bodies", np.int64, np.arange(self.n, dtype=np.int64))
        # Patch23A: historical Specimen-003 growth rebases repeatedly multiplied
        # live plastic memory into frozen edge_gain.  Restore the demonstrated
        # structural gain invariant in memory only; raw graph bytes, topology,
        # synapse counts, and the separate plastic-memory overlay stay untouched.
        self.edge_gain, self.gain_repair_report = repair_compounded_gains(
            self.edge_gain, self.indptr, self.indices, self.bodies, self.graph_path)
        self.types = arr("types", str)
        self.instance = arr("instance", str)
        self.side = arr("side", str)
        self.superclass = arr("superclass", str)
        self.subclass = arr("subclass", str)
        self.clazz = arr("clazz", str)
        self.receptor = arr("receptor", str)
        self.entry_nerve = arr("entry_nerve", str)
        self.exit_nerve = arr("exit_nerve", str)
        self.soma_neuromere = arr("soma_neuromere", str)
        self.nt = np.char.lower(arr("nt", str).astype(str))
        self.sign = arr("sign", np.float32)
        # A coarse posterior->anterior coordinate for the internal keep-alive wave.
        # MaleCNS graph v2 preserves somaNeuromere but not per-neuron XYZ.  We use
        # the real neuromere label whenever available and only spread otherwise-
        # unlocalized neurons *within their known anatomical region* to avoid one
        # enormous synchronous slab.  This tie-break is explicitly not anatomy.
        self._keepalive_axis, self._keepalive_eligible = self._build_keepalive_axis()
        if "nt_code" in z:
            self.nt_code = np.asarray(z["nt_code"], dtype=np.uint8)
        else:
            code = {
                "acetylcholine": self.NT_ACH, "gaba": self.NT_GABA,
                "glutamate": self.NT_GLUT, "histamine": self.NT_HIST,
                "dopamine": self.NT_DA, "octopamine": self.NT_OA,
                "serotonin": self.NT_5HT, "unclear": self.NT_UNCLEAR,
            }
            self.nt_code = np.asarray([code.get(str(x), self.NT_UNKNOWN) for x in self.nt], dtype=np.uint8)

        # Learning overlay covers every structural edge, including slow edges.
        self.plastic_delta = np.zeros(self.edge_count, dtype=np.float16)
        self.activity_ema_hz = np.zeros(self.n, dtype=np.float16)

        # Runtime-only acceleration buffers.  Persisted learning remains float16 and
        # authoritative; effective_scale is rebuilt once per learning window, or
        # each standalone epoch. No acceleration cache is persisted.
        self._delivery_scale = np.empty(self.edge_count, dtype=np.float32)
        _max_syn = int(np.max(self.synapse_count)) if self.edge_count else 0
        self._delivery_log_syn_lut = np.log1p(np.arange(_max_syn + 1, dtype=np.float32)).astype(np.float32, copy=False)
        self._delivery_fast_acc = np.zeros(self.n, dtype=np.float64)
        self._delivery_fast_mark = np.zeros(self.n, dtype=np.bool_)
        self._delivery_fast_touched = np.empty(self.n, dtype=np.int32)
        self._delivery_mod_acc = np.zeros((3, self.n), dtype=np.float64)
        self._delivery_mod_mark = np.zeros((3, self.n), dtype=np.bool_)
        self._delivery_mod_touched = np.empty((3, self.n), dtype=np.int32)
        self._fired_scratch = np.empty(self.n, dtype=np.int32)

        c = self.config
        self.ref_steps = int(math.ceil(c.refractory_ms / c.dt_ms))
        self.delay_steps = max(1, int(round(c.delay_ms / c.dt_ms)))
        self.em = np.float32(math.exp(-c.dt_ms / c.tau_m_ms))
        self.eg = np.float32(math.exp(-c.dt_ms / c.tau_syn_ms))
        self.couple = np.float32(c.tau_syn_ms / (c.tau_syn_ms - c.tau_m_ms) * (self.eg - self.em))
        self.mod_decay = np.asarray([
            math.exp(-c.dt_ms / c.dopamine_tau_ms),
            math.exp(-c.dt_ms / c.octopamine_tau_ms),
            math.exp(-c.dt_ms / c.serotonin_tau_ms),
        ], dtype=np.float32)
        self._keepalive_peak_mv = float(max(0.0, c.keepalive_wave_fraction) * max(0.0, c.v_thresh - c.v_rest))
        self._keepalive_update_ticks = max(1, int(round(max(c.dt_ms, c.keepalive_wave_update_ms) / c.dt_ms)))
        # Pulse size is the exact discrete-time equivalent of temporarily shifting
        # the resting equilibrium by keepalive_peak_mv over one update interval.
        # Repeated exposure therefore asymptotes near +peak_mv instead of adding it
        # on every 0.1 ms neural tick.
        update_ms=float(self._keepalive_update_ticks)*float(c.dt_ms)
        self._keepalive_pulse_fraction=float(1.0-math.exp(-update_ms/max(float(c.tau_m_ms),1e-9)))

        self.rng = np.random.default_rng(seed)
        self._initial_seed = int(seed)
        self.thread_controller = AdaptiveThreadController()
        self._static_nonunity_gains = int(np.count_nonzero(np.abs(self.edge_gain-1.0)>1e-6))
        self._memory_cache = None
        self._epoch_gpu_requested = False
        self._epoch_gpu_backend = None
        self._epoch_gpu_reason = 'not requested'
        self._epoch_gpu_last_s = None
        self._gpu_enabled = False
        self._gpu_available = False
        self._gpu_requested = False
        self._gpu_device_id = None
        self._gpu_device_name = None
        self._gpu_reason = 'not tested'
        self._gpu_cpu_probe_s = None
        self._gpu_probe_s = None
        self._gpu_estimated_full_s = None
        self._gpu_exact = None
        self._gpu_buffer = None
        self._gpu_last_forget_s = None
        self.reset_fast_state()
        self._warm_parallel_backend()
        if auto_tune:
            self._autotune_threads()
        else:
            self.thread_controller.configure_manual(1)
            self.thread_controller.mode='deferred'
            self.thread_controller.tuning_complete=False
        self._refresh_memory_cache()

    # ---------- endogenous slow/state wave ----------
    def _build_keepalive_axis(self):
        """Return compatibility arrays for the R5/helicon slow-wave substrate.

        MaleCNS names the sleep-need R5 ring neurons ER5 and the locomotion/visual
        helicon cells ExR1.  We deliberately do not stimulate EPG, dFB proxies,
        sensory, motor, endocrine, or arbitrary central neurons here.
        """
        typ=np.asarray(self.types,dtype=str)
        r5=(typ=="ER5")
        helicon=(typ=="ExR1")
        eligible=r5|helicon
        # Compatibility axis: 0=ER5, 1=ExR1; no anatomical sweep is implied.
        axis=np.full(int(self.n),np.nan,dtype=np.float32)
        axis[r5]=0.0; axis[helicon]=1.0
        self._statewave_r5_ids=np.flatnonzero(r5).astype(np.int32)
        self._statewave_helicon_ids=np.flatnonzero(helicon).astype(np.int32)
        return axis,eligible

    def _refresh_keepalive_bias(self):
        """Refresh a tiny coherent ER5/ExR1 slow-wave equilibrium shift.

        The oscillator is an internal membrane-state scaffold, not an external spike
        drive.  A signed sinusoid avoids the old one-way depolarizing traveling wash.
        Network connectivity is left to produce downstream filtering.
        """
        c=self.config
        self._keepalive_bias.fill(0.0)
        if (not bool(c.keepalive_wave_enabled)) or self._keepalive_peak_mv<=0:
            self._keepalive_phase=0.0; self._keepalive_active_neurons=0
            return
        ids=np.flatnonzero(self._keepalive_eligible)
        if not len(ids):
            self._keepalive_phase=0.0; self._keepalive_active_neurons=0
            return
        period=max(float(c.keepalive_wave_period_ms),float(c.dt_ms))
        phase=float((self._keepalive_time_ms % period)/period)
        wave=math.sin(2.0*math.pi*phase)
        # Same coherent carrier for ER5 and ExR1.  We do not invent a fixed phase lag;
        # any network lag must arise from the retained connectome/dynamics.
        self._keepalive_bias[ids]=np.float32(self._keepalive_peak_mv*wave)
        self._keepalive_phase=phase
        self._keepalive_active_neurons=int(len(ids) if abs(wave)>1e-7 else 0)

    def keepalive_info(self) -> dict[str,Any]:
        """Compatibility telemetry; semantically this is Patch20 state-wave activity."""
        c=self.config
        period=max(float(c.keepalive_wave_period_ms),float(c.dt_ms))
        return {
            'enabled':bool(c.keepalive_wave_enabled),
            'mode':'ER5(R5)+ExR1(helicon) coherent slow-wave scaffold',
            'fraction':float(c.keepalive_wave_fraction),
            'peak_mv':float(self._keepalive_peak_mv),
            'period_ms':period,
            'frequency_hz':1000.0/period,
            'phase':float((getattr(self,'_keepalive_time_ms',0.0)%period)/period) if c.keepalive_wave_enabled else 0.0,
            'active_neurons':int(getattr(self,'_keepalive_active_neurons',0)),
            'eligible_neurons':int(np.count_nonzero(self._keepalive_eligible)),
            'r5_er5_neurons':int(len(getattr(self,'_statewave_r5_ids',()))),
            'helicon_exr1_neurons':int(len(getattr(self,'_statewave_helicon_ids',()))),
            'clock':'independent neural state-wave phase; never reset by sensory/world events',
        }

    # ---------- runtime performance (observer/host only) ----------
    def _warm_parallel_backend(self):
        if not _NUMBA_AVAILABLE:
            return
        # Compile outside measured epochs on tiny scratch arrays.
        v=np.zeros(8,dtype=np.float32); g=np.zeros(8,dtype=np.float32); m=np.zeros((3,8),dtype=np.float32)
        _parallel_integrate(v,g,m,self.mod_decay,np.float32(self.config.v_rest),self.em,self.couple,self.eg)

    def _autotune_threads(self, progress=None):
        tc=self.thread_controller
        def report(msg, **payload):
            if progress is not None:
                try: progress(str(msg), **payload)
                except TypeError: progress(str(msg))
                except Exception: pass
        report(f'CPU runtime permits {tc.permitted} worker threads', phase='cpu')
        if not _NUMBA_AVAILABLE or tc.permitted <= 1:
            tc.configure_manual(1); report('Parallel CPU backend unavailable; using one worker', phase='cpu'); return tc.info()

        # Benchmark scratch data only: tuning must not advance or mutate the
        # specimen, RNG, plastic memory, body/world state, or growth pressure.
        # Thread count is the ONLY search variable. No assumptions are made
        # about physical cores, SMT topology, NUMA layout, or where an optimum
        # "ought" to be.
        n=max(8192,min(int(self.n),166703))
        edge_sample=max(250000,min(int(self.edge_count),4_000_000))
        scores={}; probe_order=[]
        vr=np.float32(self.config.v_rest); em=self.em; cp=self.couple; eg=self.eg
        dec=self.mod_decay.copy(); factor=np.float32(1.0-self.config.forgetting_per_epoch)
        base=np.ones(edge_sample,dtype=np.float16)

        def measure(th):
            th=max(1,min(int(th),tc.permitted))
            if th in scores:
                return scores[th]
            report(f'Testing CPU worker count {th}/{tc.permitted} ...', phase='cpu', workers=int(th))
            tc._apply(th)
            integrate_samples=[]; forget_samples=[]
            cuts=np.linspace(0,edge_sample,th+1,dtype=np.int64)
            def forget_once(a):
                def work(k):
                    lo=int(cuts[k]); hi=int(cuts[k+1]); a[lo:hi]=(a[lo:hi].astype(np.float32)*factor).astype(np.float16)
                if th <= 1: work(0)
                else:
                    with ThreadPoolExecutor(max_workers=th,thread_name_prefix='NMF-tune') as ex: list(ex.map(work,range(th)))

            # Warm this exact worker count before timing it.  Thread-pool startup,
            # Numba scheduling and cache/frequency transients otherwise make close
            # neighboring counts look randomly better or worse on repeated launches.
            v=np.full(n,vr,dtype=np.float32); g=np.full(n,np.float32(.01),dtype=np.float32); m=np.full((3,n),np.float32(.01),dtype=np.float32)
            for __ in range(8): _parallel_integrate(v,g,m,dec,vr,em,cp,eg)
            forget_once(base.copy())

            # Median-of-five keeps the bounded integer search from choosing a
            # thread count because of a single scheduler hiccup.
            for _ in range(5):
                v=np.full(n,vr,dtype=np.float32); g=np.full(n,np.float32(.01),dtype=np.float32); m=np.full((3,n),np.float32(.01),dtype=np.float32)
                t=time.perf_counter()
                for __ in range(40): _parallel_integrate(v,g,m,dec,vr,em,cp,eg)
                integrate_samples.append(time.perf_counter()-t)
                a=base.copy()
                t=time.perf_counter(); forget_once(a)
                forget_samples.append(time.perf_counter()-t)
            # Integrator executes 1000x/100 ms epoch; forgetting scans all edges once.
            tint=float(np.median(integrate_samples))*25.0
            cpu_forget=float(np.median(forget_samples))*(self.edge_count/edge_sample)
            tforget=(float(self._gpu_estimated_full_s) if self._gpu_enabled and self._gpu_estimated_full_s is not None else cpu_forget)
            score=tint+tforget
            scores[int(th)]=float(score); probe_order.append(int(th))
            report(f'CPU {th}: score {score:.4f}s (integrator {tint:.4f}s + forgetting {tforget:.4f}s)', phase='cpu', workers=int(th), score=float(score))
            return float(score)

        search=_bounded_thread_search(tc.permitted,measure,scores)
        best=int(search['best'])
        tc.set_auto_result(best,scores,probe_order=probe_order,
                           first_regression=search['first_regression'],
                           refinement_anchor=search['refinement_anchor'],
                           coarse_probes=search['coarse_probes'],
                           search_bracket=search['search_bracket'])
        report(f'CPU tuning complete: selected {best}/{tc.permitted} workers', phase='cpu', selected=int(best))
        return tc.info()

    def configure_threads(self, value='auto', *, force=False, progress=None):
        text=str(value).strip().lower() or 'auto'
        if text in ('auto','adaptive'):
            if (not force and self.thread_controller.mode=='auto' and self.thread_controller.tuning_complete):
                self.thread_controller._apply(self.thread_controller.best)
                return self.thread_controller.info()
            return self._autotune_threads(progress=progress)
        self.thread_controller.configure_manual(int(text)); return self.thread_controller.info()

    def thread_info(self):
        return self.thread_controller.info()

    # ---------- optional exact GPU hybrid ----------
    def gpu_info(self):
        return {
            'requested':bool(self._gpu_requested), 'available':bool(self._gpu_available),
            'enabled':bool(self._gpu_enabled or getattr(self,'_epoch_gpu_backend',None) is not None),
            'forgetting_enabled':bool(self._gpu_enabled),
            'neural_enabled':getattr(self,'_epoch_gpu_backend',None) is not None,
            'neural_reason':getattr(self,'_epoch_gpu_reason','not requested'),
            'neural_epoch_s':getattr(self,'_epoch_gpu_last_s',None), 'device_id':self._gpu_device_id,
            'device_name':self._gpu_device_name, 'reason':str(self._gpu_reason),
            'exact_probe':self._gpu_exact, 'cpu_probe_s':self._gpu_cpu_probe_s,
            'gpu_probe_s':self._gpu_probe_s, 'estimated_full_forget_s':self._gpu_estimated_full_s,
            'last_forget_s':self._gpu_last_forget_s,
            'scope':'optional exact sparse neural epochs and plastic forgetting; CPU fallback preserves state',
        }

    def _gpu_report(self, progress, msg, **payload):
        if progress is not None:
            try: progress(str(msg), **payload)
            except TypeError: progress(str(msg))
            except Exception: pass

    def _detect_cuda_device(self):
        if not _CUDA_IMPORT_AVAILABLE or _numba_cuda is None:
            return None, 'Numba CUDA module unavailable'
        try:
            if not _numba_cuda.is_available():
                return None, 'CUDA driver/device unavailable'
            dev=_numba_cuda.get_current_device()
            name=dev.name.decode(errors='replace') if isinstance(dev.name,(bytes,bytearray)) else str(dev.name)
            return (int(getattr(dev,'id',0)),name), None
        except Exception as e:
            return None, f'CUDA detection failed: {type(e).__name__}: {e}'

    @staticmethod
    def _cpu_forget_scratch(src, factor, workers):
        a=np.array(src,copy=True)
        workers=max(1,int(workers))
        cuts=np.linspace(0,len(a),workers+1,dtype=np.int64)
        def work(k):
            lo=int(cuts[k]); hi=int(cuts[k+1]); a[lo:hi]=(a[lo:hi].astype(np.float32)*factor).astype(np.float16)
        if workers <= 1: work(0)
        else:
            with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='NMF-gpu-probe-cpu') as ex:list(ex.map(work,range(workers)))
        return a

    def probe_gpu_hybrid(self, *, requested=True, progress=None):
        self._epoch_gpu_requested=bool(requested)
        self._epoch_gpu_backend=None
        self._epoch_gpu_reason='disabled by user' if not requested else 'awaiting activation'
        self._probe_gpu_forgetting(requested=requested,progress=progress)
        if requested and self._gpu_available:
            self._activate_gpu_epoch(progress)
        elif requested:
            self._epoch_gpu_requested=False
            self._epoch_gpu_reason='CUDA unavailable; using CPU'
        return self.gpu_info()

    def _activate_gpu_epoch(self,progress=None):
        try:
            from gpu_neural import CudaEpoch
            self._epoch_gpu_backend=CudaEpoch(self)
            self._epoch_gpu_reason='exact sparse CUDA neural epochs'
            self._gpu_report(progress,'GPU neural backend ready; CPU fallback retained',phase='gpu')
        except Exception as exc:
            self._epoch_gpu_backend=None;self._epoch_gpu_requested=False
            self._epoch_gpu_reason=f'CPU fallback: {type(exc).__name__}: {exc}'
            self._gpu_report(progress,self._epoch_gpu_reason,phase='gpu')

    def _probe_gpu_forgetting(self, *, requested=True, progress=None):
        self._gpu_requested=bool(requested); self._gpu_enabled=False; self._gpu_buffer=None
        if not requested:
            self._gpu_reason='disabled by user'; self._gpu_exact=None
            self._gpu_report(progress,'GPU hybrid disabled by user',phase='gpu'); return self.gpu_info()
        self._gpu_report(progress,'Detecting CUDA GPU ...',phase='gpu')
        found,err=self._detect_cuda_device()
        self._gpu_available=found is not None
        if found is None:
            self._gpu_reason=err or 'no CUDA GPU'; self._gpu_exact=None
            self._gpu_report(progress,self._gpu_reason,phase='gpu'); return self.gpu_info()
        self._gpu_device_id,self._gpu_device_name=found
        self._gpu_report(progress,f'CUDA device: {self._gpu_device_name}',phase='gpu')
        # Use deterministic scratch memory only.  The specimen is not touched.
        n=max(262144,min(int(self.edge_count),4_000_000))
        idx=np.arange(n,dtype=np.uint32)
        src=((idx % 4093).astype(np.float32)/4093.0*np.float32(.4)-np.float32(.2)).astype(np.float16)
        factor=np.float32(1.0-self.config.forgetting_per_epoch)
        workers=max(1,int(self.thread_controller.current))
        try:
            # CPU reference and timing, using the same disjoint-slice rule as live mode.
            cpu_times=[]; ref=None
            for _ in range(3):
                t=time.perf_counter(); out=self._cpu_forget_scratch(src,factor,workers); cpu_times.append(time.perf_counter()-t); ref=out
            cpu_med=float(np.median(cpu_times))

            threads=256; blocks=(n+threads-1)//threads
            d=_numba_cuda.to_device(src)
            _cuda_forget_half[blocks,threads](d,factor); _numba_cuda.synchronize()
            gpu_times=[]; gout=None
            for _ in range(5):
                t=time.perf_counter(); d.copy_to_device(src); _cuda_forget_half[blocks,threads](d,factor); gout=d.copy_to_host(); _numba_cuda.synchronize(); gpu_times.append(time.perf_counter()-t)
            gpu_med=float(np.median(gpu_times))
            exact=bool(np.array_equal(ref,gout))
            scale=float(self.edge_count)/float(n)
            self._gpu_cpu_probe_s=cpu_med*scale; self._gpu_probe_s=gpu_med*scale; self._gpu_estimated_full_s=gpu_med*scale; self._gpu_exact=exact
            if not exact:
                self._gpu_reason='rejected: GPU scratch result was not bit-identical to CPU'; self._gpu_enabled=False
            elif gpu_med >= cpu_med*0.97:
                self._gpu_reason=f'rejected: round-trip GPU pass not faster ({gpu_med:.4f}s vs CPU {cpu_med:.4f}s sample)'; self._gpu_enabled=False
            else:
                self._gpu_reason=f'accepted: exact scratch pass and {cpu_med/gpu_med:.2f}x sample speedup'; self._gpu_enabled=True
            self._gpu_report(progress,f'GPU hybrid {self._gpu_reason}',phase='gpu',enabled=bool(self._gpu_enabled))
        except Exception as e:
            self._gpu_enabled=False; self._gpu_exact=False; self._gpu_reason=f'rejected: {type(e).__name__}: {e}'
            self._gpu_report(progress,f'GPU hybrid {self._gpu_reason}',phase='gpu',enabled=False)
        finally:
            # Probe allocations belong to the bootstrap thread/context.  Live mode
            # lazily allocates its own buffer in the SimulationWorker thread.
            self._gpu_buffer=None
        return self.gpu_info()

    def apply_runtime_acceleration(self):
        # Numba's thread mask and CUDA contexts are runtime-thread concerns. Apply
        # the selected policy again inside the actual simulation worker.
        self.thread_controller._apply(self.thread_controller.best if self.thread_controller.mode=='auto' else self.thread_controller.current)
        self._gpu_buffer=None
        return {'threads':self.thread_info(),'gpu':self.gpu_info()}

    def inherit_acceleration_from(self, other):
        oi=other.thread_info(); scores=oi.get('benchmark_scores') or {}
        if oi.get('mode')=='auto' and oi.get('tuning_complete') and scores:
            self.thread_controller.set_auto_result(oi.get('best_threads',oi.get('threads',1)),scores,probe_order=oi.get('candidates'),
                first_regression=oi.get('first_regression_threads'),refinement_anchor=oi.get('refinement_anchor_threads'),
                coarse_probes=oi.get('coarse_probes'),search_bracket=oi.get('search_bracket'))
        else:
            self.thread_controller.configure_manual(oi.get('threads',1))
        self._gpu_requested=bool(getattr(other,'_gpu_requested',False)); self._gpu_available=bool(getattr(other,'_gpu_available',False)); self._gpu_enabled=bool(getattr(other,'_gpu_enabled',False))
        self._gpu_device_id=getattr(other,'_gpu_device_id',None); self._gpu_device_name=getattr(other,'_gpu_device_name',None)
        self._gpu_reason=str(getattr(other,'_gpu_reason','inherited')); self._gpu_cpu_probe_s=getattr(other,'_gpu_cpu_probe_s',None)
        self._gpu_probe_s=getattr(other,'_gpu_probe_s',None); self._gpu_estimated_full_s=getattr(other,'_gpu_estimated_full_s',None)
        self._gpu_exact=getattr(other,'_gpu_exact',None); self._gpu_buffer=None
        self._epoch_gpu_requested=bool(getattr(other,'_epoch_gpu_requested',False))
        self._epoch_gpu_backend=None
        self._epoch_gpu_reason='new graph; CUDA activation pending' if self._epoch_gpu_requested else 'CPU'
        return {'threads':self.thread_info(),'gpu':self.gpu_info()}

    def _integrate_tick(self):
        c=self.config
        if _NUMBA_AVAILABLE:
            _parallel_integrate(self.v,self.g,self.mod_state,self.mod_decay,np.float32(c.v_rest),self.em,self.couple,self.eg)
        else:
            self.mod_state *= self.mod_decay[:,None]
            vrel=self.v-c.v_rest
            self.v = c.v_rest + vrel*self.em + self.g*self.couple
            self.g *= self.eg

        # Every few neural milliseconds update the endogenous ER5/ExR1 slow wave.
        # The signed pulse is the discrete equivalent of a tiny oscillating resting
        # equilibrium (3% of rest->threshold at peak), so it remains subthreshold.
        if self._keepalive_ticks_until_update <= 0:
            self._refresh_keepalive_bias()
            ids=np.flatnonzero(self._keepalive_bias!=0.0)
            if len(ids):
                self.v[ids] += self._keepalive_bias[ids]*np.float32(self._keepalive_pulse_fraction)
            self._keepalive_ticks_until_update=self._keepalive_update_ticks
        # Private neural clock: no sensory/world event reads, resets, or synchronizes it.
        self._keepalive_time_ms += float(c.dt_ms)
        self._keepalive_ticks_until_update -= 1

    def _forget_plasticity(self):
        f=float(self.config.forgetting_per_epoch)
        if f <= 0:
            return
        factor=np.float32(1.0-f); n=int(self.thread_controller.current)
        if self._gpu_enabled and _CUDA_IMPORT_AVAILABLE and _cuda_forget_half is not None:
            try:
                st=time.perf_counter(); threads=256; blocks=(self.edge_count+threads-1)//threads
                if self._gpu_buffer is None or int(getattr(self._gpu_buffer,'size',-1)) != int(self.edge_count):
                    self._gpu_buffer=_numba_cuda.device_array(self.edge_count,dtype=np.float16)
                self._gpu_buffer.copy_to_device(self.plastic_delta)
                _cuda_forget_half[blocks,threads](self._gpu_buffer,factor)
                self._gpu_buffer.copy_to_host(self.plastic_delta); _numba_cuda.synchronize()
                self._gpu_last_forget_s=time.perf_counter()-st
                return
            except Exception as e:
                # Never risk a run because an optional accelerator disappeared.
                self._gpu_enabled=False; self._gpu_buffer=None; self._gpu_reason=f'live fallback to CPU after GPU error: {type(e).__name__}: {e}'
        self._gpu_last_forget_s=None
        if n <= 1 or self.edge_count < 1_000_000:
            self.plastic_delta = (self.plastic_delta.astype(np.float32)*factor).astype(np.float16)
            return
        # Disjoint slices: exact same elementwise float32->float16 operation, just
        # issued concurrently.  No shared reduction and no neural arithmetic order change.
        cuts=np.linspace(0,self.edge_count,n+1,dtype=np.int64)
        a=self.plastic_delta
        def work(k):
            lo=int(cuts[k]); hi=int(cuts[k+1])
            a[lo:hi]=(a[lo:hi].astype(np.float32)*factor).astype(np.float16)
        with ThreadPoolExecutor(max_workers=n,thread_name_prefix='NMF-plastic') as ex:
            list(ex.map(work,range(n)))

    def _refresh_memory_cache(self):
        d=self.plastic_delta.astype(np.float32); ad=np.abs(d); nz=np.flatnonzero(ad>1e-6)
        ema=self.activity_ema_hz.astype(np.float32)
        self._memory_cache={
            'neurons':self.n,'edges':self.edge_count,'modified_edges':int(len(nz)),
            'nonunity_frozen_edge_gains':self._static_nonunity_gains,
            'modified_fraction':float(len(nz)/max(1,self.edge_count)),
            'mean_abs_delta_all_edges':float(np.mean(ad)) if len(ad) else 0.0,
            'mean_abs_delta_modified':float(np.mean(ad[nz])) if len(nz) else 0.0,
            'max_abs_delta':float(np.max(ad)) if len(ad) else 0.0,
            'familiar_neurons':int(np.count_nonzero(ema>0)),
        }
        return self._memory_cache

    # ---------- identity ----------
    def ids(self, *, neuron_type: str | None = None, side: str | None = None,
            superclass: str | None = None, subclass: str | None = None) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        if neuron_type is not None:
            mask &= self.types == neuron_type
        if superclass is not None:
            mask &= self.superclass == superclass
        if subclass is not None:
            mask &= self.subclass == subclass
        if side is not None:
            s = np.char.lower(self.side.astype(str)); wanted = side.lower()
            if wanted in ("l", "left"):
                mask &= (s == "l") | (s == "left") | np.char.endswith(np.char.lower(self.instance.astype(str)), "_l")
            elif wanted in ("r", "right"):
                mask &= (s == "r") | (s == "right") | np.char.endswith(np.char.lower(self.instance.astype(str)), "_r")
            else:
                raise ValueError("side must be L/left or R/right")
        return np.flatnonzero(mask).astype(np.int32)

    # ---------- lifecycle ----------
    def reset_fast_state(self):
        c = self.config
        self.v = np.full(self.n, c.v_rest, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refr = np.zeros(self.n, dtype=np.int16)
        self.mod_state = np.zeros((3, self.n), dtype=np.float32)  # DA, OA, 5HT
        self.delay_ring = [np.empty(0, dtype=np.int32) for _ in range(self.delay_steps + 1)]
        self.ring_pos = 0
        self.last_rates_hz = np.zeros(self.n, dtype=np.float32)
        self._keepalive_bias = np.zeros(self.n,dtype=np.float32)
        self._keepalive_time_ms = 0.0
        self._keepalive_ticks_until_update = 0
        self._keepalive_phase = 0.0
        self._keepalive_active_neurons = 0
        self._refresh_keepalive_bias()

    def reset_learning(self):
        window=getattr(self,'_learning_window',None)
        if window is not None:window.pop('delivery_scale_ready',None)
        self.plastic_delta.fill(0)
        self.activity_ema_hz.fill(0)
        self._refresh_memory_cache()

    def reset_all(self, seed: int | None = None):
        self.reset_learning()
        self.rng = np.random.default_rng(self._initial_seed if seed is None else int(seed))
        self.reset_fast_state()

    # ---------- dynamics ----------
    def _deliver_due(self, due: np.ndarray):
        if not len(due):
            return
        if _NUMBA_AVAILABLE and _deliver_due_kernel is not None:
            _deliver_due_kernel(
                due, self.indptr, self.indices, self.w0, self._delivery_scale,
                self.nt_code, self.synapse_count, self._delivery_log_syn_lut,
                self.config.modulator_edge_scale, self.g, self.mod_state,
                self._delivery_fast_acc, self._delivery_fast_mark, self._delivery_fast_touched,
                self._delivery_mod_acc, self._delivery_mod_mark, self._delivery_mod_touched,
                self.NT_DA, self.NT_OA, self.NT_5HT)
            return
        starts = self.indptr[due]
        cnt = self.indptr[due + 1] - starts
        tot = int(cnt.sum())
        if tot <= 0:
            return
        base = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        edge_idx = np.repeat(starts - base, cnt) + np.arange(tot)
        src = np.repeat(due, cnt)
        tgt = self.indices[edge_idx]
        scale = self.edge_gain[edge_idx] * (1.0 + self.plastic_delta[edge_idx].astype(np.float32))
        val = self.w0[edge_idx] * scale
        nz = val != 0.0
        if np.any(nz):
            self.g += np.bincount(tgt[nz], weights=val[nz], minlength=self.n).astype(np.float32)
        src_code = self.nt_code[src]
        strength = np.log1p(self.synapse_count[edge_idx].astype(np.float32)) * np.float32(self.config.modulator_edge_scale) * scale
        for row, code in enumerate((self.NT_DA, self.NT_OA, self.NT_5HT)):
            m = src_code == code
            if np.any(m):
                self.mod_state[row] += np.bincount(tgt[m], weights=strength[m], minlength=self.n).astype(np.float32)

    def begin_learning_window(self, learn=True):
        if getattr(self, '_learning_window', None) is not None:
            raise RuntimeError('Learning window already active')
        self._learning_window = dict(counts=np.zeros(self.n, np.int32), seconds=0.,
                                     learn=bool(learn), valence=0.)

    def finish_learning_window(self):
        window = getattr(self, '_learning_window', None)
        if window is None:
            return None
        self._learning_window = None
        if window['seconds'] <= 0.:
            return None
        counts = window['counts']
        rates = counts.astype(np.float32)/max(window['seconds'], 1e-9)
        expected = self.activity_ema_hz.astype(np.float32)
        surprise = np.maximum(0., rates-expected)
        self.last_rates_hz = rates.copy()
        touched = 0
        if window['learn']:
            touched = self._consolidate_epoch(counts, rates, surprise, window['valence'])
            a = float(self.config.familiarity_alpha)
            self.activity_ema_hz = ((1-a)*expected+a*rates).astype(np.float16)
        mem = self._refresh_memory_cache() if window['learn'] or self._memory_cache is None else self._memory_cache
        return dict(plastic_edges_touched=int(touched), familiar_neurons=int(mem['familiar_neurons']),
                    mean_abs_plastic_delta=float(mem['mean_abs_delta_all_edges']),
                    max_abs_plastic_delta=float(mem['max_abs_delta']))

    def run_epoch(self, drives: list[tuple[np.ndarray, float]], readouts: dict[str, np.ndarray] | None = None,
                  *, neural_ms: float = 100.0, learn: bool = True,
                  homeostatic_valence: float = 0.0) -> FullEpochReport:
        c = self.config
        epoch_started=time.perf_counter()
        readouts = readouts or {}
        valence = float(np.clip(homeostatic_valence, -1.0, 1.0))

        active_drives = [(np.asarray(ids, dtype=np.int32), float(hz)) for ids, hz in drives if len(ids) and hz > 0]
        if active_drives:
            ext_idx = np.concatenate([x[0] for x in active_drives])
            ext_p = np.concatenate([np.full(len(ids), min(1.0, hz*c.dt_ms/1000.0), dtype=np.float32) for ids, hz in active_drives])
        else:
            ext_idx = np.empty(0, dtype=np.int32); ext_p = np.empty(0, dtype=np.float32)

        names = list(readouts)
        labels = np.full(self.n, -1, dtype=np.int16)
        sizes = np.empty(len(names), dtype=np.int32)
        for j, name in enumerate(names):
            ids = np.asarray(readouts[name], dtype=np.int32)
            sizes[j] = len(ids); labels[ids] = j
        read_counts = np.zeros(len(names), dtype=np.int64)
        spike_counts = np.zeros(self.n, dtype=np.int32)

        steps = max(1, int(round(neural_ms / c.dt_ms)))
        window=getattr(self,'_learning_window',None)
        if (_NUMBA_AVAILABLE and _deliver_due_kernel is not None
                and (window is None or not window.get('delivery_scale_ready',False))):
            # Same float16 -> float32 plastic conversion and multiply as the legacy
            # per-delivery expression, computed once because plasticity is constant
            # throughout an epoch and changes only during consolidation afterward.
            self._delivery_scale[:] = self.plastic_delta
            self._delivery_scale += np.float32(1.0)
            self._delivery_scale *= self.edge_gain
            self._delivery_scale_token=object()
            if window is not None:window['delivery_scale_ready']=True
        gpu_result=None
        if self._epoch_gpu_requested and self._epoch_gpu_backend is None:
            self._activate_gpu_epoch()
        if self._epoch_gpu_backend is not None:
            try:
                gpu_result=self._epoch_gpu_backend.run(self,ext_idx,ext_p,steps)
                self._epoch_gpu_last_s=self._epoch_gpu_backend.last_wall_s
            except Exception as exc:
                # No CPU state (including RNG/delays/clock) has been committed yet.
                # Execute this very same epoch once on the original CPU state.
                gpu_result=None
                self._epoch_gpu_reason=f'CPU fallback: {type(exc).__name__}: {exc}'
                self._epoch_gpu_backend=None;self._epoch_gpu_requested=False
                self._epoch_gpu_last_s=None
        if gpu_result is not None:
            self._epoch_gpu_backend.commit(self,gpu_result)
            spike_counts=gpu_result['spike_counts']
            total_spikes=int(np.sum(spike_counts,dtype=np.int64))
            valid=labels>=0
            np.add.at(read_counts,labels[valid],spike_counts[valid].astype(np.int64))
        else:
            total_spikes = 0
            for _ in range(steps):
                due = self.delay_ring[self.ring_pos]
                self._deliver_due(due)
                self.delay_ring[self.ring_pos] = np.empty(0, dtype=np.int32)

                self._integrate_tick()

                if len(ext_idx):
                    hit = ext_idx[self.rng.random(len(ext_idx)) < ext_p]
                    if len(hit):
                        self.v[hit] = c.v_thresh + 0.5

                if _NUMBA_AVAILABLE and _fire_reset_kernel is not None:
                    nf = int(_fire_reset_kernel(self.v, self.g, self.refr, np.float32(c.v_thresh), np.float32(c.v_reset),
                                                int(self.ref_steps), spike_counts, labels, read_counts, self._fired_scratch))
                    if nf:
                        total_spikes += nf
                        # Delay ring owns this spike list beyond the reusable scratch call.
                        self.delay_ring[(self.ring_pos + self.delay_steps) % len(self.delay_ring)] = self._fired_scratch[:nf].copy()
                else:
                    fired = np.flatnonzero((self.v >= c.v_thresh) & (self.refr <= 0)).astype(np.int32)
                    if len(fired):
                        total_spikes += len(fired); spike_counts[fired] += 1
                        self.v[fired] = c.v_reset; self.g[fired] = 0.0; self.refr[fired] = self.ref_steps
                        self.delay_ring[(self.ring_pos + self.delay_steps) % len(self.delay_ring)] = fired
                        if names:
                            lab = labels[fired]; lab = lab[lab >= 0]
                            if len(lab): read_counts += np.bincount(lab, minlength=len(names))
                    self.refr[self.refr > 0] -= 1
                    self.v[self.refr > 0] = c.v_reset
                self.ring_pos = (self.ring_pos + 1) % len(self.delay_ring)


        seconds = steps*c.dt_ms/1000.0
        window = getattr(self, '_learning_window', None)
        if window is not None:
            window['counts'] += spike_counts
            window['seconds'] += seconds
            window['valence'] = valence
            learn = False  # Consolidate once over the whole accumulation window.
        read_hz = {name: read_counts[j]/max(1,int(sizes[j]))/seconds for j,name in enumerate(names)}
        network_hz = total_spikes/self.n/seconds
        rates_hz = spike_counts.astype(np.float32)/max(seconds,1e-9)
        self.last_rates_hz = rates_hz.copy()
        expected = self.activity_ema_hz.astype(np.float32)
        surprise = np.maximum(0.0, rates_hz-expected)
        active_surprise = surprise[rates_hz > 0]
        novelty_hz = float(np.mean(active_surprise)) if len(active_surprise) else 0.0

        touched = 0
        if learn:
            touched = self._consolidate_epoch(spike_counts, rates_hz, surprise, valence)
            a = float(c.familiarity_alpha)
            self.activity_ema_hz = ((1-a)*expected + a*rates_hz).astype(np.float16)

        if learn or self._memory_cache is None:
            mem=self._refresh_memory_cache()
        else:
            mem=self._memory_cache
        report=FullEpochReport(
            neural_ms=float(neural_ms), network_hz=float(network_hz),
            readout_hz={k:float(v) for k,v in read_hz.items()},
            active_neurons=int(np.count_nonzero(spike_counts)), plastic_edges_touched=int(touched),
            novelty_hz=novelty_hz, familiar_neurons=int(mem['familiar_neurons']),
            mean_abs_plastic_delta=float(mem['mean_abs_delta_all_edges']),
            max_abs_plastic_delta=float(mem['max_abs_delta']),
            homeostatic_valence=valence,
            dopamine_mean=float(np.mean(self.mod_state[0])),
            octopamine_mean=float(np.mean(self.mod_state[1])),
            serotonin_mean=float(np.mean(self.mod_state[2])),
            keepalive_wave_phase=float((self._keepalive_time_ms % max(float(c.keepalive_wave_period_ms),float(c.dt_ms))) / max(float(c.keepalive_wave_period_ms),float(c.dt_ms))) if c.keepalive_wave_enabled else 0.0,
            keepalive_wave_active_neurons=int(self._keepalive_active_neurons),
            keepalive_wave_peak_mv=float(self._keepalive_peak_mv),
        )
        self.thread_controller.observe(time.perf_counter()-epoch_started)
        return report

    def _consolidate_epoch(self, spike_counts: np.ndarray, rates_hz: np.ndarray,
                           surprise: np.ndarray, valence: float) -> int:
        c = self.config
        if c.forgetting_per_epoch > 0:
            self._forget_plasticity()

        pre_ids = np.flatnonzero((spike_counts >= c.min_pre_spikes) & (surprise >= c.min_surprise_hz)).astype(np.int32)
        if len(pre_ids) > c.max_plastic_presynaptic_neurons:
            order = np.argsort(spike_counts[pre_ids])[-c.max_plastic_presynaptic_neurons:]
            pre_ids = pre_ids[order]
        touched = 0
        local_mod = np.tanh(np.sum(self.mod_state, axis=0)).astype(np.float32)

        for pre in pre_ids:
            start,end = int(self.indptr[pre]),int(self.indptr[pre+1])
            if end <= start: continue
            edge_idx = np.arange(start,end,dtype=np.int64)
            posts = self.indices[start:end]
            post_sp = spike_counts[posts].astype(np.float32)
            post_surprise = surprise[posts].astype(np.float32)
            eligible = (post_sp >= c.min_post_spikes) & (post_surprise >= c.min_surprise_hz)
            if not np.any(eligible): continue

            pre_strength = min(1.0, math.log1p(float(spike_counts[pre]))/math.log(8.0))
            active_post = post_surprise[eligible]
            scale = max(1.0,float(np.mean(active_post)))
            post_norm = active_post/scale
            centered = post_norm-float(np.mean(post_norm))
            if len(active_post)==1: centered[:] = c.target_post_activity

            e = edge_idx[eligible]
            p = posts[eligible]
            mod_gain = 1.0 + c.local_modulation_plasticity_gain*np.abs(local_mod[p])
            unsup = c.learning_rate*pre_strength*centered
            # Homeostatic reinforcement: improvement (>0) strengthens coactive
            # branches; worsening (<0) weakens them. No pain/death signal exists.
            reinf = c.reinforcement_rate*float(valence)*pre_strength*np.clip(post_norm,0.0,2.0)
            upd = np.clip((unsup+reinf)*mod_gain, -c.update_clip_per_epoch, c.update_clip_per_epoch)
            old = self.plastic_delta[e].astype(np.float32)
            self.plastic_delta[e] = np.clip(old+upd,-c.max_fractional_change,c.max_fractional_change).astype(np.float16)
            touched += len(e)
        return int(touched)

    # ---------- persistence ----------
    def memory_summary(self) -> dict[str, Any]:
        if self._memory_cache is None:
            self._refresh_memory_cache()
        out=dict(self._memory_cache)
        out['modulator_mean']={
            'dopamine':float(np.mean(self.mod_state[0])),
            'octopamine':float(np.mean(self.mod_state[1])),
            'serotonin':float(np.mean(self.mod_state[2])),
        }
        out['threads']=self.thread_info()
        return out

    def save_memory(self,path:str|Path):
        path=Path(path); d=self.plastic_delta.astype(np.float32); nz=np.flatnonzero(np.abs(d)>1e-6).astype(np.int32)
        ema=self.activity_ema_hz.astype(np.float32); fam=np.flatnonzero(ema>1e-4).astype(np.int32)
        meta={"format":"Island-of-Dr-hell-no-full-plastic-memory-v2","graph_name":self.graph_path.name,
              "edge_count":self.edge_count,"neuron_count":self.n,"config":asdict(self.config),"summary":self.memory_summary()}
        np.savez_compressed(path,edge_index=nz,delta=d[nz].astype(np.float16),familiar_neuron_index=fam,
                            expected_hz=ema[fam].astype(np.float16),metadata=np.array(json.dumps(meta)))
        return path

    def load_memory(self,path:str|Path):
        window=getattr(self,'_learning_window',None)
        if window is not None:window.pop('delivery_scale_ready',None)
        mem=np.load(path,allow_pickle=False); meta=json.loads(str(mem["metadata"]))
        if int(meta["edge_count"])!=self.edge_count or int(meta["neuron_count"])!=self.n: raise ValueError("memory does not match graph")
        self.plastic_delta.fill(0); self.activity_ema_hz.fill(0)
        self.plastic_delta[mem["edge_index"].astype(np.int64)]=mem["delta"].astype(np.float16)
        if "familiar_neuron_index" in mem:
            self.activity_ema_hz[mem["familiar_neuron_index"].astype(np.int64)]=mem["expected_hz"].astype(np.float16)
        self._refresh_memory_cache()
        return meta

    def save_fast_state(self,path:str|Path):
        meta={"format":"Island-of-Dr-hell-no-full-fast-neural-state-v2","neuron_count":self.n,
              "delay_steps":self.delay_steps,"ring_pos":int(self.ring_pos),"rng_state":self.rng.bit_generator.state,
              "keepalive_time_ms":float(self._keepalive_time_ms),
              "keepalive_ticks_until_update":int(self._keepalive_ticks_until_update),
              "state_wave_mode":"ER5+ExR1-SWA-v1",
              "state_wave_period_ms":float(self.config.keepalive_wave_period_ms)}
        payload={"v":self.v,"g":self.g,"refr":self.refr,"last_rates_hz":self.last_rates_hz,
                 "mod_state":self.mod_state.astype(np.float32),"metadata":np.array(json.dumps(meta))}
        for i,a in enumerate(self.delay_ring): payload[f"delay_{i:03d}"]=np.asarray(a,dtype=np.int32)
        np.savez_compressed(path,**payload); return Path(path)

    def _restore_state_wave_from_meta(self, meta: dict[str, Any]):
        # Patch20 migration: old "keepalive" checkpoints used a 4730-ms broad CNS
        # traveling wash. Preserve fractional phase while moving to ER5/ExR1 SWA.
        old_time=float(meta.get("keepalive_time_ms",0.0))
        new_period=max(float(self.config.keepalive_wave_period_ms),float(self.config.dt_ms))
        if meta.get("state_wave_mode") == "ER5+ExR1-SWA-v1":
            old_period=max(float(meta.get("state_wave_period_ms",new_period)),float(self.config.dt_ms))
            phase=(old_time % old_period)/old_period
            cycles=math.floor(old_time/old_period)
            self._keepalive_time_ms=(cycles+phase)*new_period
        elif "keepalive_time_ms" in meta:
            legacy_period=4730.0
            phase=(old_time % legacy_period)/legacy_period
            cycles=math.floor(old_time/legacy_period)
            self._keepalive_time_ms=(cycles+phase)*new_period
        else:
            self._keepalive_time_ms=0.0
        self._keepalive_ticks_until_update=int(meta.get("keepalive_ticks_until_update",0))
        self._refresh_keepalive_bias()

    def transient_state_sanity(self) -> dict[str, Any]:
        return fast_state_sanity(self.v,self.g,self.mod_state)

    def load_fast_state(self,path:str|Path):
        st=np.load(path,allow_pickle=False); meta=json.loads(str(st["metadata"]))
        if int(meta["neuron_count"])!=self.n or int(meta["delay_steps"])!=self.delay_steps: raise ValueError("fast state does not match graph")

        sv=st["v"].astype(np.float32); sg=st["g"].astype(np.float32)
        sm=st["mod_state"].astype(np.float32) if "mod_state" in st else np.zeros((3,self.n),np.float32)
        sanity=fast_state_sanity(sv,sg,sm)
        # Always preserve deterministic chronology metadata even when the transient
        # electrical state itself is numerically impossible.
        self.rng.bit_generator.state=meta["rng_state"]
        if sanity["sane"]:
            self.v[:]=sv; self.g[:]=sg; self.refr[:]=st["refr"].astype(np.int16)
            self.last_rates_hz[:]=st["last_rates_hz"].astype(np.float32) if "last_rates_hz" in st else 0
            self.mod_state[:]=sm
            self.delay_ring=[st[f"delay_{i:03d}"].astype(np.int32) for i in range(self.delay_steps+1)]
            self.ring_pos=int(meta["ring_pos"])%len(self.delay_ring)
            repaired=False
        else:
            # One-time legacy migration: clear only momentary electrical state.
            # Long-term plastic_delta/activity_ema are stored in the memory file and
            # are deliberately not touched here.  This is equivalent to letting an
            # invalid transient attractor decay to rest, not erasing learned memory.
            self.v.fill(self.config.v_rest); self.g.fill(0); self.refr.fill(0)
            self.last_rates_hz.fill(0); self.mod_state.fill(0)
            self.delay_ring=[np.empty(0,dtype=np.int32) for _ in range(self.delay_steps+1)]
            self.ring_pos=0
            repaired=True
        self._restore_state_wave_from_meta(meta)
        self.fast_state_repair_report={**sanity,"repaired":bool(repaired),"source":Path(path).name}
        meta=dict(meta); meta["fast_state_repair"]=dict(self.fast_state_repair_report)
        return meta
