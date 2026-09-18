#!/usr/bin/env python3
"""Repair legacy rebaked plastic gains; preserve versioned structural budgets.

Unmarked historical Specimen-003 generations used unity structural gains.
Graphs explicitly marked budgeted-growth-v1 intentionally contain fractional
structural gains. Preserve those gains and keep learned memory separate.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import numpy as np


def generation_from_graph(path: str | Path, bodies: np.ndarray) -> int:
    m=re.search(r"(?:^|[-_])W(\d{3,})(?:\D|$)",Path(path).name,re.I)
    from_name=int(m.group(1)) if m else 0
    b=np.asarray(bodies,dtype=np.int64); neg=b[b<0]; from_ids=0
    if neg.size: from_ids=int(np.max((-neg.astype(object))//(1<<32)))
    return max(from_name,from_ids)


def repair_compounded_gains(gain: np.ndarray, indptr: np.ndarray, indices: np.ndarray,
                             bodies: np.ndarray, graph_path: str | Path) -> tuple[np.ndarray,dict[str,Any]]:
    """Restore the demonstrated Specimen-003 frozen-gain invariant.

    The sparse structure arguments are retained in the signature so old callers and
    forensic reports remain compatible. The function changes no topology. It only
    normalizes frozen edge_gain for a generation-bearing Specimen-003 lineage.
    """
    g=np.asarray(gain,dtype=np.float32)
    # Explicitly versioned structural budgets are not the legacy rebaking bug.
    # Read only this small NPZ member, never infer permission from a filename.
    with np.load(graph_path,allow_pickle=False) as archive:
        policy=str(archive['structural_gain_policy']) if 'structural_gain_policy' in archive else ''
    if policy:
        if policy!='budgeted-growth-v1':raise ValueError('Unknown structural gain policy')
        if np.any(~np.isfinite(g)) or np.any(g<=0):raise ValueError('Invalid budgeted structural gains')
        return g.copy(),dict(applied=False,method='preserve-budgeted-growth-v1',structural_gain_policy=policy)
    current=generation_from_graph(graph_path,bodies)
    nonfinite=int(np.count_nonzero(~np.isfinite(g)))
    nonpositive=int(np.count_nonzero(g<=0))
    nonunity=int(np.count_nonzero((~np.isfinite(g)) | (np.abs(g-1.0)>1e-6)))
    report={
        "applied":False,"method":"none","graph":Path(graph_path).name,
        "current_generation":int(current),"before_edges":int(len(g)),
        "before_nonunity":nonunity,"before_nonfinite":nonfinite,
        "before_nonpositive":nonpositive,
        "before_min":float(np.nanmin(g)) if len(g) else 1.0,
        "before_max":float(np.nanmax(g)) if len(g) else 1.0,
        "structural_gain_invariant":1.0,
    }
    if nonunity==0:
        report.update(after_nonunity=0,after_min=1.0,after_max=1.0)
        return g.copy(),report
    # The invariant repair is forensic and lineage-specific.  Do not reinterpret
    # custom graphs or another specimen merely because its filename contains W###.
    if "specimen-003" not in Path(graph_path).name.casefold():
        report["method"]="refused-non-specimen003-lineage"
        report.update(after_nonunity=nonunity,after_min=report["before_min"],after_max=report["before_max"])
        return g.copy(),report
    if current<=0:
        # Do not silently reinterpret unrelated/custom graphs that lack lineage
        # provenance. Specimen-003 W-generations have explicit W###/synthetic IDs.
        report["method"]="refused-no-generation-provenance"
        report.update(after_nonunity=nonunity,after_min=report["before_min"],after_max=report["before_max"])
        return g.copy(),report
    out=np.ones_like(g,dtype=np.float32)
    report.update(
        applied=True,
        method="restore-frozen-structural-gain-invariant-v2",
        repaired_edges=nonunity,
        after_nonunity=0,after_min=1.0,after_max=1.0,
        preserved_topology=True,preserved_synapse_counts=True,preserved_plastic_memory_external=True,
    )
    return out,report


def fast_state_sanity(v: np.ndarray,g: np.ndarray,mod_state: np.ndarray) -> dict[str,Any]:
    """Detect a saved transient that cannot be trusted under the repaired graph.

    W120 contains astronomical membrane/conductance/modulator magnitudes generated
    while the frozen-gain bug was active. These are millisecond/second transients,
    not plastic memory. A failed check causes the loader to restart only transient
    electrical state while preserving RNG chronology and state-wave phase.
    """
    v=np.asarray(v); g=np.asarray(g); m=np.asarray(mod_state)
    vmax=float(np.nanmax(np.abs(v))) if v.size else 0.0
    gmax=float(np.nanmax(np.abs(g))) if g.size else 0.0
    mmax=float(np.nanmax(np.abs(m))) if m.size else 0.0
    finite=bool(np.all(np.isfinite(v)) and np.all(np.isfinite(g)) and np.all(np.isfinite(m)))
    # Bounds are deliberately generous relative to a -52..-45 mV LIF membrane.
    sane=finite and vmax<=1.0e4 and gmax<=1.0e6 and mmax<=1.0e6
    return {"sane":bool(sane),"finite":finite,"max_abs_v":vmax,"max_abs_g":gmax,"max_abs_mod":mmax,
            "limits":{"abs_v":1.0e4,"abs_g":1.0e6,"abs_mod":1.0e6}}
