#!/usr/bin/env python3
"""Build the Specimen-003 full-fidelity MaleCNS graph.

Design rules
============
* Keep the published neuronal census rather than ``status == Traced`` only.
* Keep every released neuron->neuron edge between retained neurons.
* Never throw monoaminergic edges away: fast conductance can be zero while the
  structural edge and raw synapse count remain present for slow modulation.
* Preserve enough source metadata to build anatomical sensor/effector bridges.

Input files are the official MaleCNS v1.0 flat-connectome Feather downloads.
The builder intentionally requires pyarrow/pandas; the runtime package does not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

EXPECTED_RELEASE_CENSUS = 166_700
PAPER_CENSUS_REFERENCE = 166_691
EXPECTED_RELEASE_DIRECTED_EDGES = 25_582_938
EXPECTED_RELEASE_SYNAPTIC_CONTACTS = 124_177_617
MV_PER_SYNAPSE = 0.275
SOURCE_HASHES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    "body-neurotransmitters-male-cns-v1.0.feather": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
}
SOURCE_BYTES = {
    "body-annotations-male-cns-v1.0-minconf-0.5.feather": 14_483_314,
    "body-neurotransmitters-male-cns-v1.0.feather": 43_282_834,
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather": 1_051_241_946,
}

# Fast-sign model. Monoamines are intentionally *not* assigned a fake fast sign.
FAST_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 0.0,
    "octopamine": 0.0,
    "serotonin": 0.0,
    "unclear": 0.0,
    "unknown": 0.0,
}
NT_CODE = {
    "unknown": 0,
    "acetylcholine": 1,
    "gaba": 2,
    "glutamate": 3,
    "histamine": 4,
    "dopamine": 5,
    "octopamine": 6,
    "serotonin": 7,
    "unclear": 8,
}


def _savez_deflate_fast(path: Path, arrays: dict, compresslevel: int = 1):
    """NumPy-compatible NPZ with low-level DEFLATE for multi-million-edge graphs."""
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=int(compresslevel),allowZip64=True) as zf:
        for key,value in arrays.items():
            a=np.asanyarray(value)
            with zf.open(str(key)+'.npy','w',force_zip64=True) as f:
                np.lib.format.write_array(f,a,allow_pickle=True)


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_sources(data_dir: Path, *, skip_hash: bool = False) -> dict:
    out = {}
    for name, expected_hash in SOURCE_HASHES.items():
        p = data_dir / name
        if not p.exists():
            raise FileNotFoundError(p)
        size = p.stat().st_size
        if size != SOURCE_BYTES[name]:
            raise ValueError(f"{name}: size {size:,} != expected {SOURCE_BYTES[name]:,}")
        digest = None if skip_hash else sha256_file(p)
        if digest is not None and digest != expected_hash:
            raise ValueError(f"{name}: SHA256 mismatch: {digest}")
        out[name] = {"bytes": size, "sha256": expected_hash if skip_hash else digest}
    return out


def _series(df, *names):
    import pandas as pd
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([""] * len(df), index=df.index)


def choose_neurons(ann, *, allow_census_mismatch: bool = False):
    """Choose the pinned MaleCNS v1.0 release neuron catalog.

    The released annotation table marks neurons with a nonblank ``superclass``.
    On the pinned, hash-verified v1.0 files this yields 166,700 unique neuronal
    bodies.  The paper reports 166,691; we preserve the *released source data*
    rather than deleting nine source records merely to force the paper-era count.

    We intentionally do *not* require ``status == Traced`` or a terminal ``type``
    annotation: both gates would discard valid released neurons.
    """
    import pandas as pd

    body_col = "bodyId" if "bodyId" in ann.columns else "bodyid"
    if body_col not in ann.columns:
        raise KeyError("annotation table lacks bodyId/bodyid")

    superclass = _series(ann, "superclass").fillna("").astype(str).str.strip()
    status_label = _series(ann, "statusLabel", "status_label").fillna("").astype(str).str.strip().str.lower()
    status = _series(ann, "status").fillna("").astype(str).str.strip()
    body_numeric = pd.to_numeric(ann[body_col], errors="coerce")
    explicit_glia = status_label.eq("glia") | superclass.str.lower().eq("glia")
    valid_body = body_numeric.notna()
    # Publication Methods: bodies in the dataset are defined as neurons iff they
    # have a superclass.  Blank-superclass rows are fragments/artifacts, not neurons.
    keep = valid_body & superclass.ne("") & ~explicit_glia

    rows = ann.loc[keep].copy()
    rows[body_col] = pd.to_numeric(rows[body_col], errors="raise").astype(np.int64)
    rows = rows.drop_duplicates(subset=[body_col]).sort_values(body_col).reset_index(drop=True)

    diagnostics = {
        "rows_total": int(len(ann)),
        "valid_body_rows": int(valid_body.sum()),
        "explicit_glia_rows": int(explicit_glia.sum()),
        "unique_superclass_defined_neurons": int(len(rows)),
        "superclass_nonblank_rows": int((superclass != "").sum()),
        "blank_superclass_rows": int((superclass == "").sum()),
        "status_traced_rows": int(status.eq("Traced").sum()),
        "status_values": {str(k): int(v) for k, v in status.value_counts(dropna=False).head(50).items()},
        "statusLabel_values": {str(k): int(v) for k, v in status_label.value_counts(dropna=False).head(50).items()},
    }
    print(json.dumps(diagnostics, indent=2))

    if len(rows) != EXPECTED_RELEASE_CENSUS and not allow_census_mismatch:
        raise RuntimeError(
            f"Superclass-defined release census selected {len(rows):,} neurons, expected "
            f"{EXPECTED_RELEASE_CENSUS:,} from the pinned MaleCNS v1.0 annotation file. "
            "Refusing to guess. Inspect diagnostics or use --allow-census-mismatch "
            "for forensic work only."
        )
    return body_col, rows, diagnostics


def _meta(rows, name: str, aliases=()):
    for n in (name, *aliases):
        if n in rows.columns:
            return rows[n].fillna("").astype(str).to_numpy()
    return np.full(len(rows), "", dtype="<U1")


def build(data_dir: Path, out_path: Path, *, allow_census_mismatch=False, skip_hash=False):
    try:
        import pandas as pd
        import pyarrow.feather as feather
    except Exception as e:
        raise SystemExit(
            "Builder requires pandas + pyarrow. Run: python -m pip install pandas pyarrow numpy scipy\n"
            f"Import failure: {e}"
        )

    provenance = verify_sources(data_dir, skip_hash=skip_hash)
    ann_path = data_dir / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    nt_path = data_dir / "body-neurotransmitters-male-cns-v1.0.feather"
    w_path = data_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

    print("Reading annotations...")
    ann = pd.read_feather(ann_path)
    body_col, rows, census_diag = choose_neurons(ann, allow_census_mismatch=allow_census_mismatch)
    bodies = rows[body_col].to_numpy(dtype=np.int64)
    n = len(bodies)
    body_index = pd.Index(bodies)

    print("Reading neurotransmitter predictions...")
    nt_df = pd.read_feather(nt_path)
    nt_body_col = "body" if "body" in nt_df.columns else ("bodyId" if "bodyId" in nt_df.columns else "bodyid")
    nt_col = "consensus_nt" if "consensus_nt" in nt_df.columns else "predicted_nt"
    nt_map = nt_df[[nt_body_col, nt_col]].dropna(subset=[nt_body_col]).drop_duplicates(subset=[nt_body_col]).set_index(nt_body_col)[nt_col]
    nt = nt_map.reindex(bodies).fillna("unknown").astype(str).str.lower().to_numpy()
    sign = np.asarray([FAST_SIGN.get(x, 0.0) for x in nt], dtype=np.float32)
    nt_code = np.asarray([NT_CODE.get(x, 0) for x in nt], dtype=np.uint8)

    print("Reading full released neuron-pair graph (no extra weight threshold)...")
    wt = feather.read_table(w_path, columns=["body_pre", "body_post", "weight"]).to_pandas()
    pre_body = wt["body_pre"].to_numpy(dtype=np.int64, copy=False)
    post_body = wt["body_post"].to_numpy(dtype=np.int64, copy=False)
    syn = wt["weight"].to_numpy(copy=False)
    del wt

    # Vectorized mapping through sorted retained body IDs. get_indexer returns -1 for raw
    # EM fragments and other non-neuronal segments; those are not simulation nodes.
    print("Mapping edge endpoints into the published neuronal census...")
    pre = body_index.get_indexer(pre_body).astype(np.int32, copy=False)
    post = body_index.get_indexer(post_body).astype(np.int32, copy=False)
    ok = (pre >= 0) & (post >= 0)
    pre, post, syn = pre[ok], post[ok], syn[ok]
    del pre_body, post_body, ok

    # Keep *every* released neuron-neuron connection, including weight 1/2 and autapses.
    syn = np.asarray(syn, dtype=np.uint32)
    print(f"Retained raw neuron-pair rows: {len(syn):,}")

    # Sort source-major, then target-major. If the release contains repeated rows for the
    # same neuron pair, consolidate them without losing synapse count.
    order = np.lexsort((post, pre))
    pre, post, syn = pre[order], post[order], syn[order]
    if len(syn):
        new = np.ones(len(syn), dtype=bool)
        new[1:] = (pre[1:] != pre[:-1]) | (post[1:] != post[:-1])
        if not np.all(new):
            starts = np.flatnonzero(new)
            syn = np.add.reduceat(syn.astype(np.uint64), starts).astype(np.uint32)
            pre, post = pre[starts], post[starts]
            print(f"Consolidated duplicate pair rows -> {len(syn):,} directed edges")

    # Strong source-integrity checks.  The paper-era neuron count differs by nine
    # from the pinned release catalog, so we validate against properties of the
    # released graph itself rather than deleting records to force a literature count.
    directed_edges = int(len(syn))
    synaptic_contacts = int(np.asarray(syn, dtype=np.uint64).sum(dtype=np.uint64))
    if not allow_census_mismatch:
        if directed_edges != EXPECTED_RELEASE_DIRECTED_EDGES:
            raise RuntimeError(
                f"Retained release graph has {directed_edges:,} directed edges; expected "
                f"{EXPECTED_RELEASE_DIRECTED_EDGES:,} for the pinned MaleCNS v1.0 files."
            )
        if synaptic_contacts != EXPECTED_RELEASE_SYNAPTIC_CONTACTS:
            raise RuntimeError(
                f"Retained release graph represents {synaptic_contacts:,} synaptic contacts; expected "
                f"{EXPECTED_RELEASE_SYNAPTIC_CONTACTS:,} for the pinned MaleCNS v1.0 files."
            )

    connected_mask = np.zeros(n, dtype=bool)
    if len(pre):
        connected_mask[pre] = True
        connected_mask[post] = True
    connected_neurons = int(np.count_nonzero(connected_mask))

    outdeg = np.bincount(pre, minlength=n).astype(np.int64)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(outdeg, out=indptr[1:])
    indices = post.astype(np.int32, copy=False)

    # Explicit zero fast weights are preserved for monoaminergic/unclear edges. Their
    # structural weight remains in synapse_count and the v2 runtime handles them as slow
    # channels rather than deleting them.
    fast_data = syn.astype(np.float32) * np.float32(MV_PER_SYNAPSE) * sign[pre]

    types = _meta(rows, "type", aliases=("flywireType", "instance"))
    instance = _meta(rows, "instance")
    side = _meta(rows, "somaSide", aliases=("soma_side", "side"))
    superclass = _meta(rows, "superclass")
    clazz = _meta(rows, "class")
    subclass = _meta(rows, "subclass")
    receptor = _meta(rows, "receptorType", aliases=("receptor",))
    status = _meta(rows, "status")
    status_label = _meta(rows, "statusLabel", aliases=("status_label",))
    soma_neuromere = _meta(rows, "somaNeuromere", aliases=("soma_neuromere",))
    entry_nerve = _meta(rows, "entryNerve", aliases=("entry_nerve",))
    exit_nerve = _meta(rows, "exitNerve", aliases=("exit_nerve",))
    hemilineage = _meta(rows, "hemilineage")
    supertype = _meta(rows, "supertype")
    birthtime = _meta(rows, "birthtime")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Writing", out_path)
    _savez_deflate_fast(out_path,{
        "format":np.array("male-cns-full-graph-v2"),"storage":np.array("csc-pre-columns-v2"),
        "data":fast_data.astype(np.float32),"indices":indices,"indptr":indptr,"shape":np.array([n,n],dtype=np.int64),
        "synapse_count":syn,"edge_gain":np.ones(len(syn),dtype=np.float32),"bodies":bodies,"sign":sign,
        "nt_code":nt_code,"nt":nt,"types":types,"instance":instance,"side":side,"superclass":superclass,
        "clazz":clazz,"subclass":subclass,"receptor":receptor,"status":status,"status_label":status_label,
        "soma_neuromere":soma_neuromere,"entry_nerve":entry_nerve,"exit_nerve":exit_nerve,
        "hemilineage":hemilineage,"supertype":supertype,"birthtime":birthtime,
        "mv_per_synapse":np.array(MV_PER_SYNAPSE,dtype=np.float32),
    },compresslevel=1)

    edge_src_nt = nt_code[pre]
    mod_counts = {name: int(np.count_nonzero(edge_src_nt == code)) for name, code in NT_CODE.items() if code >= 5 and code <= 7}
    manifest = {
        "format": "male-cns-full-graph-v2-manifest",
        "graph_file": out_path.name,
        "graph_sha256": sha256_file(out_path),
        "graph_bytes": out_path.stat().st_size,
        "neurons": int(n),
        "directed_edges": int(len(indices)),
        "release_annotation_census": EXPECTED_RELEASE_CENSUS,
        "paper_census_reference": PAPER_CENSUS_REFERENCE,
        "paper_vs_release_delta": int(EXPECTED_RELEASE_CENSUS-PAPER_CENSUS_REFERENCE),
        "expected_release_directed_edges": EXPECTED_RELEASE_DIRECTED_EDGES,
        "expected_release_synaptic_contacts": EXPECTED_RELEASE_SYNAPTIC_CONTACTS,
        "connected_neurons": connected_neurons,
        "isolated_neurons": int(n-connected_neurons),
        "synaptic_contacts": synaptic_contacts,
        "edge_policy": "all released edges between retained neuronal annotations; no additional threshold; autapses retained",
        "node_policy": "all unique body IDs with nonblank superclass in the pinned MaleCNS v1.0 release, excluding explicit glia; no status==Traced or terminal-type gate; source release count retained even where it differs from the paper-era census",
        "fast_transmitters": ["acetylcholine", "gaba", "glutamate", "histamine"],
        "slow_preserved_transmitters": ["dopamine", "octopamine", "serotonin"],
        "modulatory_edge_counts": mod_counts,
        "census_diagnostics": census_diag,
        "sources": provenance,
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-hash", action="store_true", help="skip expensive source SHA checks (sizes are still checked)")
    ap.add_argument("--allow-census-mismatch", action="store_true", help="for forensic diagnosis only; normal builds must match the pinned 166,700-node / 25,582,938-edge release")
    args = ap.parse_args()
    build(args.data_dir, args.out, allow_census_mismatch=args.allow_census_mismatch, skip_hash=args.skip_hash)


if __name__ == "__main__":
    main()
