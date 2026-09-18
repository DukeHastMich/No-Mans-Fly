#!/usr/bin/env python3
"""Structural growth/freezing for the Specimen-003 full-connectome graph.

The v2 graph keeps raw synapse counts separate from learned/frozen gain.  That matters:
we never rewrite an EM synapse count to pretend learning changed anatomy.  A frozen
working generation preserves structural gain while ``synapse_count`` remains
provenance-preserving structural data.  Newly grown provisional edges are explicitly
synthetic and receive synthetic synapse counts plus provenance in the generation
manifest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from connectome_evolution import GrowthPressureTracker, GrowthPolicyConfig
from neural_stability import repair_compounded_gains

MV_PER_SYNAPSE_DEFAULT = 0.275
META_KEYS = (
    "bodies","sign","nt_code","nt","types","instance","side","superclass","clazz",
    "subclass","receptor","status","status_label","soma_neuromere","entry_nerve",
    "exit_nerve","hemilineage","supertype","birthtime",
)


def sha256_file(path: str | Path, block: int = 8*1024*1024) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b=f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()


def _savez_deflate_fast(path: str | Path, arrays: dict[str, Any], compresslevel: int = 0):
    """Atomically write a NumPy-compatible generation NPZ.

    Live structural growth is latency-sensitive.  The active generation therefore
    uses ZIP_STORED (compresslevel <= 0): no redundant CPU compression pass, faster
    reload, and ordinary ``np.load`` compatibility.  Once that generation becomes
    historical, the controller recompresses it asynchronously in the archive.
    """
    path=Path(path);tmp=path.with_name(path.name+'.writing.tmp')
    compression=zipfile.ZIP_STORED if int(compresslevel)<=0 else zipfile.ZIP_DEFLATED
    kwargs={} if compression==zipfile.ZIP_STORED else {'compresslevel':int(compresslevel)}
    try:
        with zipfile.ZipFile(tmp,'w',compression=compression,allowZip64=True,**kwargs) as zf:
            for key,value in arrays.items():
                a=np.asanyarray(value)
                with zf.open(str(key)+'.npy','w',force_zip64=True) as f:
                    np.lib.format.write_array(f,a,allow_pickle=True)
        # Verify the central directory before publishing the new canonical graph.
        with zipfile.ZipFile(tmp,'r') as zf:
            if zf.testzip() is not None: raise IOError('generation NPZ failed ZIP CRC verification')
        os.replace(tmp,path)
    finally:
        try:
            if tmp.exists():tmp.unlink()
        except Exception:pass


def _json_scalar(x) -> str:
    if isinstance(x, np.ndarray): return str(x.item())
    return str(x)


@dataclass
class FullGrowthEdge:
    source_ref: int
    target_ref: int
    synapse_count: int
    birth_gain: float = 1.0


@dataclass
class FullNewNeuron:
    synthetic_body_id: int
    parent_body_id: int
    parent_index: int
    generation: int
    serial: int
    reason: str
    metadata: dict[str, Any]


@dataclass
class FullGrowthOverlay:
    format: str = "no-mans-fly-full-growth-overlay-v2"
    parent_graph_sha256: str = ""
    generation: int = 1
    new_neurons: list[FullNewNeuron] = field(default_factory=list)
    new_edges: list[FullGrowthEdge] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    gain_transfers: list[dict[str,Any]] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        p=Path(path)
        p.write_text(json.dumps({
            "format":self.format,"parent_graph_sha256":self.parent_graph_sha256,
            "generation":self.generation,
            "new_neurons":[asdict(x) for x in self.new_neurons],
            "new_edges":[asdict(x) for x in self.new_edges],"notes":self.notes,
            "gain_transfers":self.gain_transfers,
        },indent=2),encoding="utf-8")
        return p

    @classmethod
    def load(cls,path: str | Path) -> "FullGrowthOverlay":
        d=json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != "no-mans-fly-full-growth-overlay-v2":
            raise ValueError("not a full-growth-v2 overlay")
        return cls(format=d["format"],parent_graph_sha256=d.get("parent_graph_sha256",""),
                   generation=int(d.get("generation",1)),
                   new_neurons=[FullNewNeuron(**x) for x in d.get("new_neurons",[])],
                   new_edges=[FullGrowthEdge(**x) for x in d.get("new_edges",[])],
                   notes=list(d.get("notes",[])),gain_transfers=list(d.get("gain_transfers",[])))


def _synthetic_body_id(generation: int, serial: int) -> int:
    return -int((int(generation)<<32) | (int(serial)+1))


def _new_ref(serial: int) -> int:
    return -(int(serial)+1)


def _resolve_ref(ref: int, base_n: int) -> int:
    return int(ref) if ref >= 0 else int(base_n + (-ref-1))


def _load_v2(path: str | Path):
    z=np.load(path,allow_pickle=True)
    storage=str(z["storage"].item()) if "storage" in z else ""
    if storage != "csc-pre-columns-v2":
        raise ValueError("full structural growth requires male-cns full graph v2")
    n=int(z["shape"][0]); indptr=np.asarray(z["indptr"],dtype=np.int64)
    indices=np.asarray(z["indices"],dtype=np.int32)
    data=np.asarray(z["data"],dtype=np.float32)
    syn=np.asarray(z["synapse_count"],dtype=np.uint32)
    gain=np.asarray(z["edge_gain"],dtype=np.float32) if "edge_gain" in z else np.ones(len(data),np.float32)
    if not (len(indices)==len(data)==len(syn)==len(gain)==int(indptr[-1])):
        raise ValueError("v2 graph edge arrays disagree")
    meta={}
    for k in META_KEYS:
        if k in z: meta[k]=np.array(z[k],copy=True)
        else:
            if k=="bodies": meta[k]=np.arange(n,dtype=np.int64)
            elif k=="sign": meta[k]=np.zeros(n,dtype=np.float32)
            elif k=="nt_code": meta[k]=np.zeros(n,dtype=np.uint8)
            else: meta[k]=np.full(n,"",dtype=object)
    mv=float(z["mv_per_synapse"]) if "mv_per_synapse" in z else MV_PER_SYNAPSE_DEFAULT
    return n,indptr,indices,data,syn,gain,meta,mv



def load_full_growth_parent(graph_path: str | Path) -> dict[str,Any]:
    """Load and validate one parent graph once for seed+freeze growth work.

    Live growth used to decompress the same ~25M-edge NPZ once while selecting a
    daughter overlay and again immediately while freezing it.  This shared immutable
    parent bundle removes that duplicate I/O/decompression and computes the lineage
    hash / repaired structural gain once.
    """
    gp=Path(graph_path)
    n,indptr,indices,data,syn,gain,meta,mv=_load_v2(gp)
    repaired,repair_report=repair_compounded_gains(gain,indptr,indices,meta["bodies"],gp)
    return {
        "path":gp,"sha256":sha256_file(gp),"n":n,"indptr":indptr,"indices":indices,
        "data":data,"syn":syn,"gain":repaired,"meta":meta,"mv":mv,
        "gain_repair_report":repair_report,
    }


def _parent_arrays(graph_path: str | Path, loaded_parent: dict[str,Any] | None):
    gp=Path(graph_path)
    if loaded_parent is None:
        n,indptr,indices,data,syn,gain,meta,mv=_load_v2(gp)
        gain,rep=repair_compounded_gains(gain,indptr,indices,meta["bodies"],gp)
        return n,indptr,indices,data,syn,gain,meta,mv,sha256_file(gp),rep
    lp=loaded_parent
    if Path(lp["path"]).resolve()!=gp.resolve():
        raise ValueError("loaded growth parent belongs to another graph")
    # Metadata dict is shallow-copied because freeze appends child metadata arrays.
    return (int(lp["n"]),lp["indptr"],lp["indices"],lp["data"],lp["syn"],lp["gain"],
            dict(lp["meta"]),float(lp["mv"]),str(lp["sha256"]),dict(lp["gain_repair_report"]))

def _apply_memory(gain: np.ndarray, n: int, memory: str | Path | None):
    if not memory: return gain.copy(),None
    m=np.load(memory,allow_pickle=False); md=json.loads(_json_scalar(m["metadata"]))
    if int(md["edge_count"])!=len(gain) or int(md["neuron_count"])!=n:
        raise ValueError("plastic memory does not match full graph")
    out=gain.copy(); idx=m["edge_index"].astype(np.int64); delta=m["delta"].astype(np.float32)
    out[idx] *= (1.0+delta)
    return out,md

def _memory_metadata(edge_count: int, n: int, memory: str | Path | None):
    if not memory: return None
    m=np.load(memory,allow_pickle=False); md=json.loads(_json_scalar(m["metadata"]))
    if int(md["edge_count"])!=int(edge_count) or int(md["neuron_count"])!=int(n):
        raise ValueError("plastic memory does not match full graph")
    return md


def seed_duplicate_neuron_full(graph_path: str | Path, parent_index: int, *,
                               plastic_memory: str | Path | None=None,
                               generation: int=1, serial: int=0,
                               reason: str="sustained-capacity-pressure",
                               config: GrowthPolicyConfig | None=None,
                               _loaded_parent: dict[str,Any] | None=None) -> FullGrowthOverlay:
    cfg=config or GrowthPolicyConfig()
    if not 0.0<float(cfg.birth_weight_scale)<0.5:
        raise ValueError('Birth fraction must be positive and below 0.5')
    n,indptr,indices,data,syn,gain,meta,mv,parent_sha,_repair=_parent_arrays(graph_path,_loaded_parent)
    p=int(parent_index)
    if not (0<=p<n): raise IndexError("parent outside graph")
    eff_gain,_=_apply_memory(gain,n,plastic_memory)
    overlay=FullGrowthOverlay(parent_graph_sha256=parent_sha,generation=int(generation))
    md={k:(int(meta[k][p]) if k in ("bodies","nt_code") else float(meta[k][p]) if k=="sign" else str(meta[k][p])) for k in META_KEYS}
    body=_synthetic_body_id(generation,serial); md["bodies"]=body
    md["instance"]=(md.get("instance") or md.get("types") or "neuron")+f"__G{generation}N{serial:04d}"
    md["status"]="Synthetic grown"; md["status_label"]="Synthetic grown"
    child=FullNewNeuron(body,int(meta["bodies"][p]),p,int(generation),int(serial),str(reason),md)
    overlay.new_neurons.append(child); cref=_new_ref(serial)

    # Select edges by *structural* strength. Monoaminergic edges have zero fast data
    # but remain eligible because synapse_count is never discarded.
    s,e=int(indptr[p]),int(indptr[p+1])
    oe=np.arange(s,e,dtype=np.int64)
    if len(oe):
        score=syn[oe].astype(np.float32)*eff_gain[oe]
        keep=np.flatnonzero(score>=max(1.0,float(cfg.min_seed_weight_abs)/max(mv,1e-9)))
        if len(keep)>cfg.max_outgoing_seed_edges:
            keep=keep[np.argsort(score[keep])[-cfg.max_outgoing_seed_edges:]]
        for j in keep:
            count=max(1,int(round(float(syn[oe[j]])*float(cfg.birth_weight_scale))))
            fraction=float(cfg.birth_weight_scale);edge=int(oe[j])
            newborn_gain=float(syn[edge])*fraction*float(gain[edge])/count
            overlay.new_edges.append(FullGrowthEdge(cref,int(indices[edge]),count,newborn_gain))
            overlay.gain_transfers.append(dict(edge_index=edge,fraction=fraction,new_edge=len(overlay.new_edges)-1))

    ie=np.flatnonzero(indices==p).astype(np.int64)
    if len(ie):
        src=np.searchsorted(indptr,ie,side="right")-1
        score=syn[ie].astype(np.float32)*eff_gain[ie]
        keep=np.flatnonzero(score>=max(1.0,float(cfg.min_seed_weight_abs)/max(mv,1e-9)))
        if len(keep)>cfg.max_incoming_seed_edges:
            keep=keep[np.argsort(score[keep])[-cfg.max_incoming_seed_edges:]]
        for j in keep:
            count=max(1,int(round(float(syn[ie[j]])*float(cfg.birth_weight_scale))))
            fraction=float(cfg.birth_weight_scale);edge=int(ie[j])
            newborn_gain=float(syn[edge])*fraction*float(gain[edge])/count
            overlay.new_edges.append(FullGrowthEdge(int(src[j]),cref,count,newborn_gain))
            overlay.gain_transfers.append(dict(edge_index=edge,fraction=fraction,new_edge=len(overlay.new_edges)-1))

    outn=sum(1 for x in overlay.new_edges if x.source_ref==cref); inn=sum(1 for x in overlay.new_edges if x.target_ref==cref)
    overlay.notes.append(f"daughter of body {int(meta['bodies'][p])} / {meta['types'][p]}: {inn} incoming, {outn} outgoing provisional structural edges")
    return overlay



def _merge_growth_edges_incremental(indptr: np.ndarray, indices: np.ndarray, data: np.ndarray,
                                    syn: np.ndarray, gain: np.ndarray, new_n: int,
                                    ns: np.ndarray, nt: np.ndarray, nd: np.ndarray,
                                    nc: np.ndarray, ng: np.ndarray):
    """Append a sparse growth overlay without re-sorting the immutable base graph.

    The frozen graph is already ordered by (source,target).  Structural growth adds a
    tiny overlay (normally <=256 edges) to a ~25M-edge graph.  Reconstructing an
    explicit 25M-element source vector and globally lexsorting it is therefore both
    unnecessary and catastrophically expensive.

    We instead copy unchanged source ranges in large contiguous slices and locally
    stable-sort only source blocks that receive new edges.  Stable target ordering
    matches the old concatenate-then-lexsort semantics: pre-existing duplicate keys,
    if any, remain before newly appended duplicates.
    """
    indptr=np.asarray(indptr,dtype=np.int64)
    indices=np.asarray(indices,dtype=np.int32)
    data=np.asarray(data,dtype=np.float32)
    syn=np.asarray(syn,dtype=np.uint32)
    gain=np.asarray(gain,dtype=np.float32)
    ns=np.asarray(ns,dtype=np.int32); nt=np.asarray(nt,dtype=np.int32)
    nd=np.asarray(nd,dtype=np.float32); nc=np.asarray(nc,dtype=np.uint32); ng=np.asarray(ng,dtype=np.float32)
    base_n=len(indptr)-1; old_e=len(indices); add_e=len(ns)
    if not (len(nt)==len(nd)==len(nc)==len(ng)==add_e):
        raise ValueError('growth edge arrays disagree')
    if new_n < base_n:
        raise ValueError('new graph cannot have fewer neurons than parent')
    if add_e:
        if np.any(ns<0) or np.any(ns>=new_n) or np.any(nt<0) or np.any(nt>=new_n):
            raise ValueError('growth edge outside graph')
        # Preserve old global lexsort semantics for the tiny overlay itself.
        add_order=np.lexsort((np.arange(add_e,dtype=np.int64),nt,ns))
        ns,nt,nd,nc,ng=(x[add_order] for x in (ns,nt,nd,nc,ng))
        add_counts=np.bincount(ns,minlength=new_n).astype(np.int64)
    else:
        add_counts=np.zeros(new_n,dtype=np.int64)

    base_deg=np.zeros(new_n,dtype=np.int64)
    base_deg[:base_n]=np.diff(indptr)
    outdeg=base_deg+add_counts
    outptr=np.empty(new_n+1,dtype=np.int64);outptr[0]=0;np.cumsum(outdeg,out=outptr[1:])
    total_e=int(outptr[-1])
    out_i=np.empty(total_e,dtype=np.int32)
    out_d=np.empty(total_e,dtype=np.float32)
    out_s=np.empty(total_e,dtype=np.uint32)
    out_g=np.empty(total_e,dtype=np.float32)

    if add_e == 0:
        out_i[:old_e]=indices;out_d[:old_e]=data;out_s[:old_e]=syn;out_g[:old_e]=gain
        return outptr,out_i,out_d,out_s,out_g,{'changed_sources':0,'copied_edges':old_e,'locally_sorted_edges':0}

    changed=np.flatnonzero(add_counts).astype(np.int64)
    add_ptr=np.empty(new_n+1,dtype=np.int64);add_ptr[0]=0;np.cumsum(add_counts,out=add_ptr[1:])
    old_cursor=0;new_cursor=0;local_sorted=0

    def _copy_old(a:int,b:int):
        nonlocal new_cursor
        ln=b-a
        if ln<=0:return
        e=new_cursor+ln
        out_i[new_cursor:e]=indices[a:b];out_d[new_cursor:e]=data[a:b]
        out_s[new_cursor:e]=syn[a:b];out_g[new_cursor:e]=gain[a:b]
        new_cursor=e

    for src in changed:
        src=int(src)
        if src < base_n:
            os,oe=int(indptr[src]),int(indptr[src+1])
        else:
            os=oe=old_e
        _copy_old(old_cursor,os)
        a,b=int(add_ptr[src]),int(add_ptr[src+1])
        ai,ad,asyn,ag=nt[a:b],nd[a:b],nc[a:b],ng[a:b]
        if oe>os:
            # Existing source block is already target-sorted.  If the new targets are
            # all beyond it (the normal daughter-neuron case), append directly with no
            # local sort.  Otherwise sort only this one affected source block.
            oi=indices[os:oe]
            if len(ai)==0:
                _copy_old(os,oe)
            elif int(oi[-1]) <= int(ai[0]):
                _copy_old(os,oe)
                e=new_cursor+len(ai)
                out_i[new_cursor:e]=ai;out_d[new_cursor:e]=ad;out_s[new_cursor:e]=asyn;out_g[new_cursor:e]=ag
                new_cursor=e
            else:
                bi=np.concatenate((oi,ai));bd=np.concatenate((data[os:oe],ad))
                bs=np.concatenate((syn[os:oe],asyn));bg=np.concatenate((gain[os:oe],ag))
                order=np.argsort(bi,kind='stable')
                e=new_cursor+len(order)
                out_i[new_cursor:e]=bi[order];out_d[new_cursor:e]=bd[order]
                out_s[new_cursor:e]=bs[order];out_g[new_cursor:e]=bg[order]
                local_sorted += len(order);new_cursor=e
        else:
            e=new_cursor+len(ai)
            out_i[new_cursor:e]=ai;out_d[new_cursor:e]=ad;out_s[new_cursor:e]=asyn;out_g[new_cursor:e]=ag
            new_cursor=e
        old_cursor=oe
    _copy_old(old_cursor,old_e)
    if new_cursor != total_e:
        raise RuntimeError(f'incremental growth merge wrote {new_cursor} edges, expected {total_e}')
    return outptr,out_i,out_d,out_s,out_g,{
        'changed_sources':int(len(changed)),'copied_edges':int(old_e),
        'locally_sorted_edges':int(local_sorted),'added_edges':int(add_e),
    }

def freeze_full_baseline(graph_path: str | Path, output_path: str | Path, *,
                         plastic_memory: str | Path | None=None,
                         growth_overlay: FullGrowthOverlay | str | Path | None=None,
                         generation_name: str="W001", source_specimen: str="Specimen-003",
                         _loaded_parent: dict[str,Any] | None=None) -> dict[str,Any]:
    graph_path=Path(graph_path); output_path=Path(output_path)
    n,indptr,indices,data,syn,gain,meta,mv,parent_sha,legacy_gain_repair=_parent_arrays(graph_path,_loaded_parent)
    # Patch23: frozen edge_gain is structural, never a second learning store.
    # Historical Specimen-003 rebases mistakenly baked the live plastic overlay into
    # this field. Restore the 1.0 structural invariant once, then carry it forward;
    # the separate plastic-memory file is remapped by the lineage engine after growth.
    memory_meta=_memory_metadata(len(gain),n,plastic_memory)
    if growth_overlay is None: overlay=FullGrowthOverlay(parent_graph_sha256=parent_sha)
    elif isinstance(growth_overlay,FullGrowthOverlay): overlay=growth_overlay
    else: overlay=FullGrowthOverlay.load(growth_overlay)
    if overlay.parent_graph_sha256 and overlay.parent_graph_sha256!=parent_sha:
        raise ValueError("growth overlay belongs to another graph")

    k=len(overlay.new_neurons); new_n=n+k
    # A clone is a budget split, never additive gain. Validate the transfer
    # ledger before touching any output. Raw measured counts are not rewritten.
    fractions={};seen=set()
    for transfer in overlay.gain_transfers:
        edge=int(transfer['edge_index']);new=int(transfer['new_edge']);fraction=float(transfer['fraction'])
        if not (0<=edge<len(gain) and 0<=new<len(overlay.new_edges)) or new in seen or not (0.<fraction<1.):
            raise ValueError('Invalid structural gain transfer')
        seen.add(new);ed=overlay.new_edges[new]
        expected=float(syn[edge])*float(gain[edge])*fraction
        actual=float(ed.synapse_count)*float(ed.birth_gain)
        if not np.isfinite(actual) or not np.isclose(actual,expected,rtol=1e-6,atol=1e-9):
            raise ValueError('New edge exceeds donated structural strength')
        fractions[edge]=fractions.get(edge,0.)+fraction
    if overlay.new_edges and len(seen)!=len(overlay.new_edges):
        raise ValueError('Every new edge requires an explicit strength-budget donor')
    if any(value>=1. for value in fractions.values()):
        raise ValueError('Structural donor budget exhausted')
    gain=gain.copy()
    for edge,fraction in fractions.items():gain[edge]*=np.float32(1.-fraction)

    # Keep the immutable base graph in its native source-block representation.
    # Growth overlays are tiny; do not materialize/re-sort all parent edges.
    tgt=indices; raw=data; sc=syn; eg=gain

    # Append child metadata before computing new edge fast signs.
    if k:
        ordered=sorted(overlay.new_neurons,key=lambda x:x.serial)
        if [x.serial for x in ordered] != list(range(k)): raise ValueError("new neuron serials must be contiguous")
        for key in META_KEYS:
            vals=[]
            for child in ordered:
                vals.append(child.metadata.get(key,""))
            if key=="bodies": ex=np.asarray(vals,dtype=np.int64)
            elif key=="sign": ex=np.asarray(vals,dtype=np.float32)
            elif key=="nt_code": ex=np.asarray(vals,dtype=np.uint8)
            else: ex=np.asarray(vals,dtype=object)
            meta[key]=np.concatenate([meta[key],ex])

    ns=[]; nt=[]; nc=[]; ng=[]
    for ed in overlay.new_edges:
        a=_resolve_ref(int(ed.source_ref),n); b=_resolve_ref(int(ed.target_ref),n)
        if not (0<=a<new_n and 0<=b<new_n): raise ValueError("growth edge outside graph")
        ns.append(a); nt.append(b); nc.append(max(1,int(ed.synapse_count))); ng.append(float(ed.birth_gain))
    ns=np.asarray(ns,np.int32); nt=np.asarray(nt,np.int32); nc=np.asarray(nc,np.uint32); ng=np.asarray(ng,np.float32)
    nd=(nc.astype(np.float32)*np.float32(mv)*meta["sign"][ns].astype(np.float32)) if len(ns) else np.empty(0,np.float32)
    outptr,tgt,raw,sc,eg,merge_stats=_merge_growth_edges_incremental(
        indptr,indices,data,syn,gain,new_n,ns,nt,nd,nc,ng)

    output_path.parent.mkdir(parents=True,exist_ok=True)
    arrays={
        "format":np.array("male-cns-full-graph-v2"),"storage":np.array("csc-pre-columns-v2"),
        "structural_gain_policy":np.array("budgeted-growth-v1"),
        "data":raw.astype(np.float32),"indices":tgt.astype(np.int32),"indptr":outptr,"shape":np.array([new_n,new_n],np.int64),
        "synapse_count":sc.astype(np.uint32),"edge_gain":eg.astype(np.float32),"mv_per_synapse":np.array(mv,np.float32),
        **meta,
    }
    _savez_deflate_fast(output_path,arrays,compresslevel=0)
    manifest={
        "format":"no-mans-fly-full-frozen-baseline-v2","generation":generation_name,
        "source_specimen":source_specimen,"parent_graph":graph_path.name,"parent_graph_sha256":parent_sha,
        "parent_neurons":n,"parent_edges":int(len(indices)),"new_neurons":k,"new_edges":len(overlay.new_edges),
        "frozen_neurons":new_n,"frozen_edges":int(len(tgt)),"output_graph":output_path.name,
        "output_graph_sha256":sha256_file(output_path),"plastic_memory_summary":None if memory_meta is None else memory_meta.get("summary"),
        "legacy_gain_repair":legacy_gain_repair,
        "growth_rebuild":{"method":"incremental-source-block-merge-v1","active_npz_compression":"stored",**merge_stats},
        "new_neuron_provenance":[asdict(x) for x in overlay.new_neurons],"notes":overlay.notes,
        "edge_gain_semantics":"raw EM synapse_count preserved; stable repaired frozen baseline in edge_gain; bounded live plastic_delta is remapped across growth rather than re-multiplied",
    }
    mp=output_path.with_suffix(output_path.suffix+".manifest.json"); mp.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return manifest


def validate_full_graph(path: str | Path) -> dict[str,Any]:
    n,indptr,indices,data,syn,gain,meta,mv=_load_v2(path)
    return {"neurons":n,"edges":len(indices),"raw_synapses_sum":int(np.sum(syn,dtype=np.uint64)),
            "monoamine_neurons":{x:int(np.count_nonzero(meta['nt']==x)) for x in ('dopamine','octopamine','serotonin')},
            "nonunity_edge_gains":int(np.count_nonzero(np.abs(gain-1)>1e-6)),"sha256":sha256_file(path)}


__all__=["GrowthPressureTracker","GrowthPolicyConfig","FullGrowthOverlay","load_full_growth_parent","seed_duplicate_neuron_full","freeze_full_baseline","validate_full_graph"]
