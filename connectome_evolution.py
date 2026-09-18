#!/usr/bin/env python3
"""Structural growth and generational baseline compiler for Island of Dr hell no.

The adult MaleCNS headless graph remains immutable. During an experimental
lineage, structural changes are represented as a GrowthOverlay. When a change
has been validated, ``freeze_baseline`` compiles:

    baseline graph + synaptic plasticity + structural growth

back into the *same NPZ graph schema* used by the original headless MaleCNS
export.  A frozen generation therefore requires no evolution-aware loader.

Important boundary
------------------
The growth policy in this module is an engineering hypothesis, not a claim
about literal adult Drosophila neurogenesis.  It is deliberately conservative:
activity alone never causes a new neuron. Sustained positive saturation or
repeated activity in a disconnected source identifies a trial candidate.
The desktop proposes connections for evaluation rather than cloning candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

GRAPH_KEYS = (
    "data", "indices", "indptr", "shape", "bodies", "sign", "types",
    "instance", "side", "superclass", "subclass", "receptor", "nt",
)
NEURON_META_KEYS = (
    "bodies", "sign", "types", "instance", "side", "superclass",
    "subclass", "receptor", "nt",
)


def sha256_file(path: str | Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _json_scalar(x: np.ndarray | str) -> str:
    if isinstance(x, np.ndarray):
        return str(x.item())
    return str(x)


@dataclass
class GrowthPolicyConfig:
    """Conservative eligibility thresholds for structural growth."""

    recruitment_rate_hz: float = 5.0
    overload_rate_hz: float = 120.0
    saturation_fraction_of_limit: float = 0.80
    min_saturated_outgoing_edges: int = 8
    min_saturated_outgoing_fraction: float = 0.20
    sustain_epochs: int = 24
    cooldown_epochs: int = 100
    max_new_neurons_per_checkpoint: int = 4

    # A newborn starts as a weak copy of useful local structure rather than a
    # full-strength clone of its parent.
    birth_weight_scale: float = 0.12
    max_incoming_seed_edges: int = 128
    max_outgoing_seed_edges: int = 128
    min_seed_weight_abs: float = 0.25


@dataclass
class NewNeuron:
    synthetic_body_id: int
    parent_body_id: int
    parent_index: int
    generation: int
    serial: int
    reason: str
    type: str
    instance: str
    side: str
    superclass: str
    subclass: str
    receptor: str
    nt: str
    sign: float


@dataclass
class GrowthEdge:
    source_ref: int
    target_ref: int
    weight: float
    # source_ref / target_ref use base neuron indices for >=0 values and
    # -(serial+1) for new neurons. This makes overlays independent of the
    # final appended array index.


@dataclass
class GrowthOverlay:
    format: str = "Island-of-Dr-hell-no-growth-overlay-v1"
    parent_graph_sha256: str = ""
    generation: int = 1
    new_neurons: list[NewNeuron] = field(default_factory=list)
    new_edges: list[GrowthEdge] = field(default_factory=list)
    pruned_edges: list[tuple[int, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        doc = {
            "format": self.format,
            "parent_graph_sha256": self.parent_graph_sha256,
            "generation": self.generation,
            "new_neurons": [asdict(x) for x in self.new_neurons],
            "new_edges": [asdict(x) for x in self.new_edges],
            "pruned_edges": [[int(a), int(b)] for a, b in self.pruned_edges],
            "notes": list(self.notes),
        }
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GrowthOverlay":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != "Island-of-Dr-hell-no-growth-overlay-v1":
            raise ValueError("Unknown growth overlay format")
        return cls(
            format=d["format"],
            parent_graph_sha256=d["parent_graph_sha256"],
            generation=int(d["generation"]),
            new_neurons=[NewNeuron(**x) for x in d.get("new_neurons", [])],
            new_edges=[GrowthEdge(**x) for x in d.get("new_edges", [])],
            pruned_edges=[(int(x[0]), int(x[1])) for x in d.get("pruned_edges", [])],
            notes=list(d.get("notes", [])),
        )


class GrowthPressureTracker:
    """Track persistent local capacity pressure without growing on a spike alone.

    A neuron is eligible only when:
      * its current firing rate exceeds ``overload_rate_hz``; AND
      * enough of its outgoing learned synapses are already near the configured
        plasticity ceiling; AND
      * the condition persists for ``sustain_epochs``.

    The tracker does not alter anatomy. It produces ranked candidate indices so
    growth can occur at an explicit checkpoint and remain auditable.
    """

    FORMAT = "Island-of-Dr-hell-no-growth-pressure-v1"
    PRESSURE_SEMANTICS_VERSION = 3

    def __init__(self, neuron_count: int, config: GrowthPolicyConfig | None = None):
        self.config = config or GrowthPolicyConfig()
        self.pressure_epochs = np.zeros(int(neuron_count), dtype=np.uint16)
        self.cooldown = np.zeros(int(neuron_count), dtype=np.uint16)
        self.total_epochs = 0
        # A topology rewrite invalidates pre-growth pressure evidence.  Hold all
        # structural growth for one conservative global cooldown, then require a fresh
        # sustained-pressure streak on the new graph before another generation.
        self.global_cooldown_epochs = 0
        self.legacy_pressure_invalidated = False

    def observe(self, core: Any, rates_hz: np.ndarray | None = None) -> list[dict[str, Any]]:
        if rates_hz is None:
            rates_hz = getattr(core, "last_rates_hz", None)
        if rates_hz is None:
            raise ValueError("Core has no last_rates_hz; run an epoch or pass rates_hz")
        rates_hz = np.asarray(rates_hz, dtype=np.float32)
        if rates_hz.shape != (core.n,):
            raise ValueError("rates_hz shape does not match core")

        # Current core has one plastic vector for the immutable baseline CSC.
        # Candidate growth is therefore limited to baseline source neurons.
        base_n = len(core.indptr) - 1
        if len(self.pressure_epochs) != core.n:
            old = len(self.pressure_epochs)
            self.pressure_epochs = np.pad(self.pressure_epochs, (0, core.n-old))
            self.cooldown = np.pad(self.cooldown, (0, core.n-old))

        self.total_epochs += 1
        self.cooldown[self.cooldown > 0] -= 1
        if self.global_cooldown_epochs > 0:
            self.global_cooldown_epochs -= 1
            # Do not bank a queue of candidates while the new topology settles.
            self.pressure_epochs[:] = 0
            return []
        hot = np.flatnonzero((rates_hz[:base_n] >= self.config.overload_rate_hz) & (self.cooldown[:base_n] == 0))
        # Disconnected but persistently stimulated cells need connection trials,
        # not an impossible requirement for saturated outgoing edges.
        disconnected=np.diff(core.indptr[:base_n+1])==0
        recruits=disconnected & (rates_hz[:base_n]>=self.config.recruitment_rate_hz) & (self.cooldown[:base_n]==0)
        qualifies = recruits.copy()
        threshold = float(core.config.max_fractional_change) * self.config.saturation_fraction_of_limit

        for pre in hot:
            s, e = int(core.indptr[pre]), int(core.indptr[pre+1])
            if e <= s:
                continue
            d = core.plastic_delta[s:e].astype(np.float32)
            sat = int(np.count_nonzero(d >= threshold))
            frac = sat / max(1, len(d))
            if sat >= self.config.min_saturated_outgoing_edges and frac >= self.config.min_saturated_outgoing_fraction:
                qualifies[pre] = True

        # Strict persistence: a missed epoch resets the streak.
        self.pressure_epochs[:base_n][~qualifies] = 0
        q = np.flatnonzero(qualifies)
        if len(q):
            self.pressure_epochs[q] = np.minimum(65535, self.pressure_epochs[q].astype(np.uint32) + 1).astype(np.uint16)

        ready = np.flatnonzero(self.pressure_epochs[:base_n] >= self.config.sustain_epochs)
        ranked: list[dict[str, Any]] = []
        for pre in ready:
            s, e = int(core.indptr[pre]), int(core.indptr[pre+1])
            d = core.plastic_delta[s:e].astype(np.float32)
            sat = int(np.count_nonzero(d >= threshold))
            ranked.append({
                "kind": "connection_recruitment" if disconnected[pre] else "capacity_trial",
                "index": int(pre),
                "body_id": int(core.bodies[pre]),
                "type": str(core.types[pre]),
                "instance": str(core.instance[pre]),
                "rate_hz": float(rates_hz[pre]),
                "pressure_epochs": int(self.pressure_epochs[pre]),
                "outgoing_edges": int(e-s),
                "saturated_edges": sat,
                "saturated_fraction": float(sat / max(1, e-s)),
            })
        ranked.sort(key=lambda x: (x["pressure_epochs"], x["saturated_fraction"], x["rate_hz"]), reverse=True)
        return ranked[: self.config.max_new_neurons_per_checkpoint]

    def mark_grown(self, neuron_index: int):
        i = int(neuron_index)
        self.pressure_epochs[:] = 0
        self.pressure_epochs[i] = 0
        self.cooldown[i] = np.uint16(min(65535, self.config.cooldown_epochs))
        self.global_cooldown_epochs = max(int(self.global_cooldown_epochs), int(self.config.cooldown_epochs))

    def save(self, path: str | Path) -> Path:
        meta = {
            "format": self.FORMAT,
            "config": asdict(self.config),
            "total_epochs": int(self.total_epochs),
            "neuron_count": int(len(self.pressure_epochs)),
            "pressure_semantics_version": int(self.PRESSURE_SEMANTICS_VERSION),
            "global_cooldown_epochs": int(self.global_cooldown_epochs),
        }
        np.savez_compressed(
            path,
            pressure_epochs=self.pressure_epochs,
            cooldown=self.cooldown,
            metadata=np.array(json.dumps(meta)),
        )
        return Path(path)

    @classmethod
    def load(cls, path: str | Path) -> "GrowthPressureTracker":
        z = np.load(path, allow_pickle=False)
        m = json.loads(str(z["metadata"]))
        if m.get("format") != cls.FORMAT:
            raise ValueError("Unknown growth pressure format")
        obj = cls(int(m["neuron_count"]), GrowthPolicyConfig(**m["config"]))
        obj.pressure_epochs[:] = z["pressure_epochs"].astype(np.uint16)
        obj.cooldown[:] = z["cooldown"].astype(np.uint16)
        obj.total_epochs = int(m["total_epochs"])
        semantics = int(m.get("pressure_semantics_version", 1))
        if semantics < cls.PRESSURE_SEMANTICS_VERSION:
            # v1 allowed several already-mature candidates to survive a topology
            # rewrite.  That produced W333->W338-style one-generation-per-step
            # cascades.  Old pressure streaks are therefore not valid evidence under
            # the rewritten graph and must be reacquired.
            obj.pressure_epochs[:] = 0
            obj.global_cooldown_epochs = int(obj.config.cooldown_epochs)
            obj.legacy_pressure_invalidated = True
        else:
            obj.global_cooldown_epochs = max(0, int(m.get("global_cooldown_epochs", 0)))
        return obj


def load_effective_csc(graph_path: str | Path, plastic_memory: str | Path | None = None):
    """Return graph metadata and a CSC whose weights include learned deltas.

    Plastic memory edge indices are defined in PlasticMaleCNSCore's CSC edge
    order, so deltas must be applied after CSR -> CSC conversion.
    """
    z = np.load(graph_path, allow_pickle=True)
    missing = [k for k in GRAPH_KEYS if k not in z.files]
    if missing:
        raise ValueError(f"Graph missing required keys: {missing}")
    shape = tuple(int(x) for x in z["shape"])
    csr = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)
    W = csr.tocsc().astype(np.float32)
    memory_meta = None
    if plastic_memory:
        mem = np.load(plastic_memory, allow_pickle=False)
        memory_meta = json.loads(_json_scalar(mem["metadata"]))
        if int(memory_meta["edge_count"]) != W.nnz or int(memory_meta["neuron_count"]) != W.shape[0]:
            raise ValueError("Plastic memory does not match parent graph")
        idx = mem["edge_index"].astype(np.int64)
        delta = mem["delta"].astype(np.float32)
        W.data[idx] *= (1.0 + delta)
    metadata = {k: np.array(z[k], copy=True) for k in NEURON_META_KEYS}
    return W, metadata, memory_meta


def _synthetic_body_id(generation: int, serial: int) -> int:
    # FlyWire body IDs in the source graph are positive. Negative int64 values
    # reserve a collision-free namespace while preserving the bodies[] dtype.
    return -int((int(generation) << 32) | (int(serial) + 1))


def _new_ref(serial: int) -> int:
    return -(int(serial) + 1)


def _resolve_ref(ref: int, base_n: int) -> int:
    if ref >= 0:
        return int(ref)
    return int(base_n + (-ref - 1))


def seed_duplicate_neuron(
    graph_path: str | Path,
    parent_index: int,
    *,
    plastic_memory: str | Path | None = None,
    generation: int = 1,
    serial: int = 0,
    reason: str = "sustained-capacity-pressure",
    config: GrowthPolicyConfig | None = None,
) -> GrowthOverlay:
    """Create one weak provisional daughter neuron from a parent's local wiring."""
    cfg = config or GrowthPolicyConfig()
    W, meta, _ = load_effective_csc(graph_path, plastic_memory)
    n = W.shape[0]
    p = int(parent_index)
    if not (0 <= p < n):
        raise IndexError("parent_index outside graph")

    overlay = GrowthOverlay(
        parent_graph_sha256=sha256_file(graph_path),
        generation=int(generation),
    )
    body = _synthetic_body_id(generation, serial)
    inst = str(meta["instance"][p])
    child_inst = f"{inst}__G{generation}N{serial:04d}" if inst else f"G{generation}N{serial:04d}"
    child = NewNeuron(
        synthetic_body_id=body,
        parent_body_id=int(meta["bodies"][p]),
        parent_index=p,
        generation=int(generation),
        serial=int(serial),
        reason=str(reason),
        type=str(meta["types"][p]),
        instance=child_inst,
        side=str(meta["side"][p]),
        superclass=str(meta["superclass"][p]),
        subclass=str(meta["subclass"][p]),
        receptor=str(meta["receptor"][p]),
        nt=str(meta["nt"][p]),
        sign=float(meta["sign"][p]),
    )
    overlay.new_neurons.append(child)
    cref = _new_ref(serial)

    # Outgoing parent column: parent -> targets.
    s, e = W.indptr[p], W.indptr[p+1]
    targets = W.indices[s:e].astype(np.int64)
    weights = W.data[s:e].astype(np.float32)
    keep = np.flatnonzero(np.abs(weights) >= cfg.min_seed_weight_abs)
    if len(keep) > cfg.max_outgoing_seed_edges:
        keep = keep[np.argsort(np.abs(weights[keep]))[-cfg.max_outgoing_seed_edges:]]
    for j in keep:
        overlay.new_edges.append(GrowthEdge(cref, int(targets[j]), float(weights[j] * cfg.birth_weight_scale)))

    # Incoming row: sources -> parent. getrow on CSC is acceptable for rare growth checkpoints.
    row = W.getrow(p).tocoo()
    sources = row.col.astype(np.int64)
    inw = row.data.astype(np.float32)
    keep = np.flatnonzero(np.abs(inw) >= cfg.min_seed_weight_abs)
    if len(keep) > cfg.max_incoming_seed_edges:
        keep = keep[np.argsort(np.abs(inw[keep]))[-cfg.max_incoming_seed_edges:]]
    for j in keep:
        overlay.new_edges.append(GrowthEdge(int(sources[j]), cref, float(inw[j] * cfg.birth_weight_scale)))

    overlay.notes.append(
        f"Seeded daughter of body {int(meta['bodies'][p])} / {meta['types'][p]} with "
        f"{sum(1 for x in overlay.new_edges if x.source_ref == cref)} outgoing and "
        f"{sum(1 for x in overlay.new_edges if x.target_ref == cref)} incoming provisional edges."
    )
    return overlay


def freeze_baseline(
    graph_path: str | Path,
    output_path: str | Path,
    *,
    plastic_memory: str | Path | None = None,
    growth_overlay: GrowthOverlay | str | Path | None = None,
    generation_name: str = "G1",
    source_specimen: str | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compile plasticity + growth into a standalone headless graph NPZ."""
    graph_path = Path(graph_path)
    output_path = Path(output_path)
    W, meta, memory_meta = load_effective_csc(graph_path, plastic_memory)
    base_n = W.shape[0]
    base_edges = int(W.nnz)
    parent_sha = sha256_file(graph_path)

    if growth_overlay is None:
        overlay = GrowthOverlay(parent_graph_sha256=parent_sha)
    elif isinstance(growth_overlay, GrowthOverlay):
        overlay = growth_overlay
    else:
        overlay = GrowthOverlay.load(growth_overlay)
    if overlay.parent_graph_sha256 and overlay.parent_graph_sha256 != parent_sha:
        raise ValueError("Growth overlay belongs to a different parent graph")

    k = len(overlay.new_neurons)
    new_n = base_n + k
    if k:
        W.resize((new_n, new_n))

    # Apply explicit pruning by setting existing pairs to zero. This is deliberately
    # rare/offline; ordinary synaptic weakening remains in plasticity instead.
    if overlay.pruned_edges:
        W = W.tolil()
        for sref, tref in overlay.pruned_edges:
            src = _resolve_ref(int(sref), base_n)
            tgt = _resolve_ref(int(tref), base_n)
            W[tgt, src] = 0.0
        W = W.tocsc()

    if overlay.new_edges:
        rows, cols, vals = [], [], []
        for edge in overlay.new_edges:
            src = _resolve_ref(int(edge.source_ref), base_n)
            tgt = _resolve_ref(int(edge.target_ref), base_n)
            if not (0 <= src < new_n and 0 <= tgt < new_n):
                raise ValueError("Growth edge resolves outside frozen graph")
            rows.append(tgt)
            cols.append(src)
            vals.append(float(edge.weight))
        A = sp.coo_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(new_n, new_n)).tocsc()
        W = (W + A).tocsc()
    W.eliminate_zeros()

    # Append metadata for newborns in serial order. Require contiguous serials so
    # the negative-ref namespace resolves deterministically.
    if k:
        ordered = sorted(overlay.new_neurons, key=lambda x: x.serial)
        expected = list(range(k))
        if [x.serial for x in ordered] != expected:
            raise ValueError(f"New-neuron serials must be contiguous {expected}")
        append: dict[str, list[Any]] = {key: [] for key in NEURON_META_KEYS}
        for x in ordered:
            append["bodies"].append(int(x.synthetic_body_id))
            append["sign"].append(float(x.sign))
            append["types"].append(x.type)
            append["instance"].append(x.instance)
            append["side"].append(x.side)
            append["superclass"].append(x.superclass)
            append["subclass"].append(x.subclass)
            append["receptor"].append(x.receptor)
            append["nt"].append(x.nt)
        for key in NEURON_META_KEYS:
            if key == "bodies":
                extra = np.asarray(append[key], dtype=np.int64)
            elif key == "sign":
                extra = np.asarray(append[key], dtype=np.float32)
            else:
                # Original file uses object string arrays; preserve that shape/schema.
                extra = np.asarray(append[key], dtype=object)
            meta[key] = np.concatenate([meta[key], extra])

    csr = W.tocsr()
    if csr.shape != (new_n, new_n):
        raise AssertionError("Frozen matrix shape mismatch")
    for key in NEURON_META_KEYS:
        if len(meta[key]) != new_n:
            raise AssertionError(f"Frozen metadata length mismatch: {key}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        data=csr.data.astype(np.float32),
        indices=csr.indices.astype(np.int32),
        indptr=csr.indptr.astype(np.int32),
        shape=np.asarray(csr.shape, dtype=np.int64),
        bodies=meta["bodies"].astype(np.int64),
        sign=meta["sign"].astype(np.float32),
        types=meta["types"].astype(object),
        instance=meta["instance"].astype(object),
        side=meta["side"].astype(object),
        superclass=meta["superclass"].astype(object),
        subclass=meta["subclass"].astype(object),
        receptor=meta["receptor"].astype(object),
        nt=meta["nt"].astype(object),
    )
    out_sha = sha256_file(output_path)
    manifest = {
        "format": "Island-of-Dr-hell-no-frozen-baseline-manifest-v1",
        "generation": generation_name,
        "source_specimen": source_specimen,
        "parent_graph": graph_path.name,
        "parent_graph_sha256": parent_sha,
        "plastic_memory": None if plastic_memory is None else Path(plastic_memory).name,
        "plastic_memory_summary": None if memory_meta is None else memory_meta.get("summary"),
        "growth_overlay_generation": overlay.generation,
        "new_neurons": k,
        "new_edges_requested": len(overlay.new_edges),
        "pruned_edges_requested": len(overlay.pruned_edges),
        "parent_neurons": base_n,
        "parent_edges": base_edges,
        "frozen_neurons": int(csr.shape[0]),
        "frozen_edges": int(csr.nnz),
        "output_graph": output_path.name,
        "output_graph_sha256": out_sha,
        "new_neuron_provenance": [asdict(x) for x in overlay.new_neurons],
        "notes": list(overlay.notes),
    }
    if manifest_path is None:
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_same_schema(graph_path: str | Path) -> dict[str, Any]:
    z = np.load(graph_path, allow_pickle=True)
    missing = [k for k in GRAPH_KEYS if k not in z.files]
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    n = int(z["shape"][0])
    if tuple(z["shape"]) != (n, n):
        raise ValueError("Graph is not square")
    for key in NEURON_META_KEYS:
        if len(z[key]) != n:
            raise ValueError(f"Metadata {key} length != neuron count")
    csr = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
    return {
        "neurons": n,
        "edges": int(csr.nnz),
        "keys": list(z.files),
        "body_min": int(np.min(z["bodies"])),
        "body_max": int(np.max(z["bodies"])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("seed-duplicate")
    p.add_argument("graph")
    p.add_argument("parent_index", type=int)
    p.add_argument("--memory")
    p.add_argument("--generation", type=int, default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--reason", default="sustained-capacity-pressure")

    p = sub.add_parser("freeze")
    p.add_argument("graph")
    p.add_argument("output")
    p.add_argument("--memory")
    p.add_argument("--overlay")
    p.add_argument("--generation-name", default="G1")
    p.add_argument("--source-specimen")

    p = sub.add_parser("validate")
    p.add_argument("graph")

    a = ap.parse_args()
    if a.cmd == "seed-duplicate":
        ov = seed_duplicate_neuron(a.graph, a.parent_index, plastic_memory=a.memory,
                                   generation=a.generation, reason=a.reason)
        ov.save(a.output)
        print(json.dumps({"new_neurons": len(ov.new_neurons), "new_edges": len(ov.new_edges), "notes": ov.notes}, indent=2))
    elif a.cmd == "freeze":
        ov = None if not a.overlay else GrowthOverlay.load(a.overlay)
        print(json.dumps(freeze_baseline(a.graph, a.output, plastic_memory=a.memory,
                                         growth_overlay=ov, generation_name=a.generation_name,
                                         source_specimen=a.source_specimen), indent=2))
    elif a.cmd == "validate":
        print(json.dumps(validate_same_schema(a.graph), indent=2))


if __name__ == "__main__":
    main()
