#!/usr/bin/env python3
"""Island of Dr hell no — persistent plastic MaleCNS core.

This module turns the verified adult MaleCNS wiring diagram from a fixed graph
into a *learnable* dynamical system while keeping the published/anatomical graph
immutable underneath.

Design goals
------------
* No reward, punishment, pain, hunger, or survival objective is required.
* Neural state persists continuously across observation epochs.
* Repeated correlated activity can alter effective synaptic strengths.
* Plasticity is bounded, slowly forgetful, and sign-preserving.
* Long-term plastic memory can be checkpointed independently of momentary spikes.
* Reset always returns to the original adult-connectome baseline.

The first learning rule is intentionally conservative and explicitly experimental:
an epochal, activity-dependent Hebbian competition rule with homeostatic decay.
It is NOT a claim that these exact equations describe Drosophila learning.
It is a replaceable learning layer around the real structural graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp


@dataclass
class PlasticityConfig:
    # Neural dynamics (same family as the Shiu-style core used elsewhere here)
    v_rest: float = -52.0
    v_thresh: float = -45.0
    v_reset: float = -52.0
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    dt_ms: float = 0.1

    # Plasticity. delta modifies magnitude as w_eff = w0 * (1 + delta).
    learning_rate: float = 0.018
    forgetting_per_epoch: float = 0.0015
    max_fractional_change: float = 0.35
    min_pre_spikes: int = 2
    min_post_spikes: int = 1
    target_post_activity: float = 0.20
    update_clip_per_epoch: float = 0.012

    # Familiarity / novelty expectation. Repeated activity becomes less surprising.
    familiarity_alpha: float = 0.08
    min_surprise_hz: float = 0.5

    # To stop one epoch touching millions of weakly relevant edges.
    max_plastic_presynaptic_neurons: int = 12000


@dataclass
class EpochReport:
    neural_ms: float
    network_hz: float
    readout_hz: dict[str, float]
    active_neurons: int
    plastic_edges_touched: int
    novelty_hz: float
    familiar_neurons: int
    mean_abs_plastic_delta: float
    max_abs_plastic_delta: float

    def to_dict(self):
        return asdict(self)


class PlasticMaleCNSCore:
    """Persistent full-connectome simulator with bounded unsupervised plasticity."""

    def __init__(
        self,
        graph_path: str | Path,
        *,
        seed: int = 7,
        config: PlasticityConfig | None = None,
    ):
        self.graph_path = Path(graph_path)
        self.config = config or PlasticityConfig()
        z = np.load(self.graph_path, allow_pickle=True)
        W = sp.csr_matrix(
            (z["data"], z["indices"], z["indptr"]),
            shape=tuple(z["shape"]),
        ).tocsc()

        self.indptr = W.indptr.astype(np.int64, copy=False)
        self.indices = W.indices.astype(np.int32, copy=False)
        self.w0 = W.data.astype(np.float32, copy=False)
        self.n = int(W.shape[0])
        self.edge_count = int(len(self.w0))
        self.types = np.asarray(z["types"], dtype=str)
        self.instance = np.asarray(z["instance"], dtype=str)
        self.side = np.asarray(z["side"], dtype=str)
        self.bodies = np.asarray(z["bodies"], dtype=np.int64) if "bodies" in z else np.arange(self.n, dtype=np.int64)

        # Float16 is sufficient for a +/-0.35 fractional modifier and costs ~20 MB
        # for the 10.2M-edge graph instead of ~41 MB.
        self.plastic_delta = np.zeros(self.edge_count, dtype=np.float16)
        # Running per-neuron expected activity. This is a second, slower memory trace
        # used to distinguish familiar activity from surprising activity.
        self.activity_ema_hz = np.zeros(self.n, dtype=np.float16)

        c = self.config
        self.ref_steps = int(math.ceil(c.refractory_ms / c.dt_ms))
        self.delay_steps = max(1, int(round(c.delay_ms / c.dt_ms)))
        self.em = np.float32(math.exp(-c.dt_ms / c.tau_m_ms))
        self.eg = np.float32(math.exp(-c.dt_ms / c.tau_syn_ms))
        self.couple = np.float32(
            c.tau_syn_ms / (c.tau_syn_ms - c.tau_m_ms) * (self.eg - self.em)
        )

        self.rng = np.random.default_rng(seed)
        self._initial_seed = int(seed)
        self.reset_fast_state()

    # ---------- identity / populations ----------
    def ids(self, *, neuron_type: str | None = None, side: str | None = None) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        if neuron_type is not None:
            mask &= self.types == neuron_type
        if side is not None:
            s = np.char.lower(self.side.astype(str))
            wanted = side.lower()
            if wanted in ("l", "left"):
                mask &= (s == "l") | (s == "left") | (np.char.find(np.char.lower(self.instance.astype(str)), "_l") >= 0)
            elif wanted in ("r", "right"):
                mask &= (s == "r") | (s == "right") | (np.char.find(np.char.lower(self.instance.astype(str)), "_r") >= 0)
            else:
                raise ValueError("side must be left/L or right/R")
        return np.flatnonzero(mask).astype(np.int32)

    # ---------- state lifecycle ----------
    def reset_fast_state(self):
        """Clear moment-to-moment electrical state but keep learned weights."""
        c = self.config
        self.v = np.full(self.n, c.v_rest, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refr = np.zeros(self.n, dtype=np.int16)
        self.delay_ring = [np.empty(0, dtype=np.int32) for _ in range(self.delay_steps + 1)]
        self.ring_pos = 0
        self.last_rates_hz = np.zeros(self.n, dtype=np.float32)

    def reset_learning(self):
        """Return effective connectivity to the immutable adult connectome."""
        self.plastic_delta.fill(0)
        self.activity_ema_hz.fill(0)

    def reset_all(self, seed: int | None = None):
        self.reset_learning()
        if seed is None:
            seed = self._initial_seed
        self.rng = np.random.default_rng(int(seed))
        self.reset_fast_state()

    # ---------- simulation ----------
    def run_epoch(
        self,
        drives: list[tuple[np.ndarray, float]],
        readouts: dict[str, np.ndarray] | None = None,
        *,
        neural_ms: float = 100.0,
        learn: bool = True,
    ) -> EpochReport:
        """Advance the existing neural state; optionally consolidate plasticity."""
        c = self.config
        readouts = readouts or {}

        active_drives = [(np.asarray(ids, dtype=np.int32), float(hz)) for ids, hz in drives if len(ids) and hz > 0]
        if active_drives:
            ext_idx = np.concatenate([x[0] for x in active_drives])
            ext_p = np.concatenate([
                np.full(len(ids), min(1.0, hz * c.dt_ms / 1000.0), dtype=np.float32)
                for ids, hz in active_drives
            ])
        else:
            ext_idx = np.empty(0, dtype=np.int32)
            ext_p = np.empty(0, dtype=np.float32)

        names = list(readouts)
        labels = np.full(self.n, -1, dtype=np.int16)
        sizes = np.empty(len(names), dtype=np.int32)
        for j, name in enumerate(names):
            ids = np.asarray(readouts[name], dtype=np.int32)
            sizes[j] = len(ids)
            labels[ids] = j
        read_counts = np.zeros(len(names), dtype=np.int64)
        spike_counts = np.zeros(self.n, dtype=np.int32)

        steps = max(1, int(round(neural_ms / c.dt_ms)))
        total_spikes = 0

        for _ in range(steps):
            due = self.delay_ring[self.ring_pos]
            if len(due):
                starts = self.indptr[due]
                cnt = self.indptr[due + 1] - starts
                tot = int(cnt.sum())
                if tot:
                    base = np.concatenate(([0], np.cumsum(cnt)[:-1]))
                    edge_idx = np.repeat(starts - base, cnt) + np.arange(tot)
                    tgt = self.indices[edge_idx]
                    # Preserve baseline sign; learning only changes magnitude.
                    scale = 1.0 + self.plastic_delta[edge_idx].astype(np.float32)
                    val = self.w0[edge_idx] * scale
                    self.g += np.bincount(tgt, weights=val, minlength=self.n).astype(np.float32)
            self.delay_ring[self.ring_pos] = np.empty(0, dtype=np.int32)

            vrel = self.v - c.v_rest
            self.v = c.v_rest + vrel * self.em + self.g * self.couple
            self.g *= self.eg

            if len(ext_idx):
                hit = ext_idx[self.rng.random(len(ext_idx)) < ext_p]
                if len(hit):
                    self.v[hit] = c.v_thresh + 0.5

            fired = np.flatnonzero((self.v >= c.v_thresh) & (self.refr <= 0)).astype(np.int32)
            if len(fired):
                total_spikes += len(fired)
                spike_counts[fired] += 1
                self.v[fired] = c.v_reset
                self.g[fired] = 0.0
                self.refr[fired] = self.ref_steps
                self.delay_ring[(self.ring_pos + self.delay_steps) % len(self.delay_ring)] = fired

                if len(names):
                    lab = labels[fired]
                    lab = lab[lab >= 0]
                    if len(lab):
                        read_counts += np.bincount(lab, minlength=len(names))

            self.refr[self.refr > 0] -= 1
            self.v[self.refr > 0] = c.v_reset
            self.ring_pos = (self.ring_pos + 1) % len(self.delay_ring)

        seconds = steps * c.dt_ms / 1000.0
        read_hz = {
            name: read_counts[j] / max(1, int(sizes[j])) / seconds
            for j, name in enumerate(names)
        }
        network_hz = total_spikes / self.n / seconds

        rates_hz = spike_counts.astype(np.float32) / max(seconds, 1e-9)
        self.last_rates_hz = rates_hz.copy()
        expected = self.activity_ema_hz.astype(np.float32)
        surprise = np.maximum(0.0, rates_hz - expected)
        active_surprise = surprise[rates_hz > 0]
        novelty_hz = float(np.mean(active_surprise)) if len(active_surprise) else 0.0

        touched = 0
        if learn:
            touched = self._consolidate_epoch(spike_counts, rates_hz, surprise)
            a = float(c.familiarity_alpha)
            self.activity_ema_hz = ((1.0-a)*expected + a*rates_hz).astype(np.float16)

        pd = self.plastic_delta.astype(np.float32)
        return EpochReport(
            neural_ms=float(neural_ms),
            network_hz=float(network_hz),
            readout_hz={k: float(v) for k, v in read_hz.items()},
            active_neurons=int(np.count_nonzero(spike_counts)),
            plastic_edges_touched=int(touched),
            novelty_hz=float(novelty_hz),
            familiar_neurons=int(np.count_nonzero(self.activity_ema_hz.astype(np.float32) > 0)),
            mean_abs_plastic_delta=float(np.mean(np.abs(pd))),
            max_abs_plastic_delta=float(np.max(np.abs(pd))),
        )

    def _consolidate_epoch(self, spike_counts: np.ndarray, rates_hz: np.ndarray, surprise: np.ndarray) -> int:
        """Bounded unsupervised correlation learning on edges used this epoch.

        For each sufficiently active presynaptic neuron, only its existing outgoing
        synapses are eligible. Postsynaptic activity above the local outgoing mean
        potentiates that branch; activity below the mean depresses it. A slow global
        decay pulls all plastic changes back toward the anatomical baseline.

        Therefore no new semantic connection is invented and no sign is flipped.
        """
        c = self.config

        # Slow forgetting/homeostasis toward the immutable anatomical baseline.
        if c.forgetting_per_epoch > 0:
            self.plastic_delta = (
                self.plastic_delta.astype(np.float32) * (1.0 - c.forgetting_per_epoch)
            ).astype(np.float16)

        pre_ids = np.flatnonzero((spike_counts >= c.min_pre_spikes) & (surprise >= c.min_surprise_hz)).astype(np.int32)
        if len(pre_ids) == 0:
            return 0

        if len(pre_ids) > c.max_plastic_presynaptic_neurons:
            # Deterministic selection by strongest activity, not RNG.
            order = np.argsort(spike_counts[pre_ids])[-c.max_plastic_presynaptic_neurons:]
            pre_ids = pre_ids[order]

        touched = 0
        for pre in pre_ids:
            start, end = int(self.indptr[pre]), int(self.indptr[pre + 1])
            if end <= start:
                continue
            edge_idx = np.arange(start, end, dtype=np.int64)
            posts = self.indices[start:end]
            post_sp = spike_counts[posts].astype(np.float32)
            post_surprise = surprise[posts].astype(np.float32)
            eligible = (post_sp >= c.min_post_spikes) & (post_surprise >= c.min_surprise_hz)
            if not np.any(eligible):
                continue

            # Pre activity changes how strongly this epoch is consolidated but is
            # bounded so high-frequency cells do not dominate indefinitely.
            pre_strength = min(1.0, math.log1p(float(spike_counts[pre])) / math.log(8.0))
            active_post = post_surprise[eligible]
            scale = max(1.0, float(np.mean(active_post)))
            post_norm = active_post / scale
            # Competition within the pre-neuron's active outgoing branches.
            centered = post_norm - float(np.mean(post_norm))
            # If only one active branch exists, permit weak Hebbian potentiation.
            if len(active_post) == 1:
                centered[:] = c.target_post_activity

            upd = c.learning_rate * pre_strength * centered
            upd = np.clip(upd, -c.update_clip_per_epoch, c.update_clip_per_epoch)
            e = edge_idx[eligible]
            old = self.plastic_delta[e].astype(np.float32)
            new = np.clip(old + upd, -c.max_fractional_change, c.max_fractional_change)
            self.plastic_delta[e] = new.astype(np.float16)
            touched += len(e)

        return int(touched)

    # ---------- persistent memory ----------
    def memory_summary(self) -> dict:
        d = self.plastic_delta.astype(np.float32)
        nz = np.flatnonzero(np.abs(d) > 1e-6)
        ema = self.activity_ema_hz.astype(np.float32)
        return {
            "neurons": self.n,
            "edges": self.edge_count,
            "modified_edges": int(len(nz)),
            "modified_fraction": float(len(nz) / max(1, self.edge_count)),
            "mean_abs_delta_all_edges": float(np.mean(np.abs(d))),
            "mean_abs_delta_modified": float(np.mean(np.abs(d[nz]))) if len(nz) else 0.0,
            "max_abs_delta": float(np.max(np.abs(d))) if len(d) else 0.0,
            "familiar_neurons": int(np.count_nonzero(ema > 0)),
            "mean_expected_hz_familiar": float(np.mean(ema[ema > 0])) if np.any(ema > 0) else 0.0,
        }

    def save_memory(self, path: str | Path):
        """Save only learned deviations; baseline graph remains a separate immutable file."""
        path = Path(path)
        d = self.plastic_delta.astype(np.float32)
        nz = np.flatnonzero(np.abs(d) > 1e-6).astype(np.int32)
        meta = {
            "format": "Island-of-Dr-hell-no-plastic-memory-v1",
            "graph_name": self.graph_path.name,
            "edge_count": self.edge_count,
            "neuron_count": self.n,
            "config": asdict(self.config),
            "summary": self.memory_summary(),
        }
        ema = self.activity_ema_hz.astype(np.float32)
        fam = np.flatnonzero(ema > 1e-4).astype(np.int32)
        np.savez_compressed(
            path,
            edge_index=nz,
            delta=d[nz].astype(np.float16),
            familiar_neuron_index=fam,
            expected_hz=ema[fam].astype(np.float16),
            metadata=np.array(json.dumps(meta)),
        )
        return path

    def load_memory(self, path: str | Path):
        mem = np.load(path, allow_pickle=False)
        meta = json.loads(str(mem["metadata"]))
        if int(meta["edge_count"]) != self.edge_count or int(meta["neuron_count"]) != self.n:
            raise ValueError("Plastic memory does not match this graph")
        self.plastic_delta.fill(0)
        self.activity_ema_hz.fill(0)
        idx = mem["edge_index"].astype(np.int64)
        self.plastic_delta[idx] = mem["delta"].astype(np.float16)
        if "familiar_neuron_index" in mem and "expected_hz" in mem:
            fi = mem["familiar_neuron_index"].astype(np.int64)
            self.activity_ema_hz[fi] = mem["expected_hz"].astype(np.float16)
        return meta

    def save_fast_state(self, path: str | Path):
        """Checkpoint moment-to-moment electrical/RNG state.

        This is intentionally separate from :meth:`save_memory`: learned state can
        be reused without preserving an in-flight spike pattern, while a continuous
        specimen checkpoint can save both files.
        """
        path = Path(path)
        meta = {
            "format": "Island-of-Dr-hell-no-fast-neural-state-v1",
            "neuron_count": self.n,
            "delay_steps": self.delay_steps,
            "ring_pos": int(self.ring_pos),
            "rng_state": self.rng.bit_generator.state,
        }
        payload = {
            "v": self.v.astype(np.float32),
            "g": self.g.astype(np.float32),
            "refr": self.refr.astype(np.int16),
            "last_rates_hz": self.last_rates_hz.astype(np.float32),
            "metadata": np.array(json.dumps(meta)),
        }
        for i, arr in enumerate(self.delay_ring):
            payload[f"delay_{i:03d}"] = np.asarray(arr, dtype=np.int32)
        np.savez_compressed(path, **payload)
        return path

    def load_fast_state(self, path: str | Path):
        """Restore an exact electrical/RNG continuation checkpoint."""
        st = np.load(path, allow_pickle=False)
        meta = json.loads(str(st["metadata"]))
        if meta.get("format") != "Island-of-Dr-hell-no-fast-neural-state-v1":
            raise ValueError("Unknown fast neural state format")
        if int(meta["neuron_count"]) != self.n or int(meta["delay_steps"]) != self.delay_steps:
            raise ValueError("Fast neural state does not match this graph/config")
        v = st["v"].astype(np.float32)
        g = st["g"].astype(np.float32)
        refr = st["refr"].astype(np.int16)
        if v.shape != (self.n,) or g.shape != (self.n,) or refr.shape != (self.n,):
            raise ValueError("Fast neural state arrays have wrong shape")
        self.v[:] = v
        self.g[:] = g
        self.refr[:] = refr
        if "last_rates_hz" in st:
            self.last_rates_hz[:] = st["last_rates_hz"].astype(np.float32)
        else:
            self.last_rates_hz.fill(0.0)
        self.delay_ring = []
        for i in range(self.delay_steps + 1):
            key = f"delay_{i:03d}"
            if key not in st:
                raise ValueError(f"Fast neural state missing {key}")
            self.delay_ring.append(st[key].astype(np.int32))
        self.ring_pos = int(meta["ring_pos"]) % len(self.delay_ring)
        self.rng.bit_generator.state = meta["rng_state"]
        return meta


class PassiveVisualPorts:
    """Very conservative v0 visual projection ports for observatory experiments.

    We currently know hemisphere identity but not a faithful retinotopic coordinate
    for every photoreceptor in the compact NPZ. Therefore this class does NOT fake
    a biological pixel map. It exposes broad real visual populations only.

    A future graph export should retain optic-column coordinates and replace this.
    """

    def __init__(self, core: PlasticMaleCNSCore):
        self.core = core
        self.left_brightness = core.ids(neuron_type="R1-R6", side="left")
        self.right_brightness = core.ids(neuron_type="R1-R6", side="right")
        self.left_uv = np.concatenate([
            core.ids(neuron_type="R7p", side="left"),
            core.ids(neuron_type="R7y", side="left"),
        ])
        self.right_uv = np.concatenate([
            core.ids(neuron_type="R7p", side="right"),
            core.ids(neuron_type="R7y", side="right"),
        ])
        self.left_color = np.concatenate([
            core.ids(neuron_type="R8p", side="left"),
            core.ids(neuron_type="R8y", side="left"),
        ])
        self.right_color = np.concatenate([
            core.ids(neuron_type="R8p", side="right"),
            core.ids(neuron_type="R8y", side="right"),
        ])

    def drives(self, left: float, right: float, *, uv: float = 0.0, color: float = 0.0,
               dark_hz: float = 0.0, max_hz: float = 140.0):
        clip = lambda x: float(np.clip(x, 0.0, 1.0))
        l, r, uv, color = map(clip, (left, right, uv, color))
        return [
            (self.left_brightness, dark_hz + max_hz*l),
            (self.right_brightness, dark_hz + max_hz*r),
            (self.left_uv, max_hz*uv*l),
            (self.right_uv, max_hz*uv*r),
            (self.left_color, max_hz*color*l),
            (self.right_color, max_hz*color*r),
        ]


def _demo(graph: str, epochs: int, seed: int, save: str | None):
    core = PlasticMaleCNSCore(graph, seed=seed)
    vis = PassiveVisualPorts(core)
    # Broad downstream visual/central-complex readouts for a non-semantic fingerprint.
    readouts = {
        "MeTu1": core.ids(neuron_type="MeTu1"),
        "TuBu03": core.ids(neuron_type="TuBu03"),
        "ER4d": core.ids(neuron_type="ER4d"),
        "PFL3": core.ids(neuron_type="PFL3"),
    }
    print("Passive repeated-light demonstration")
    print("No reward, punishment, or forced choice. Neural and synaptic state persist.\n")
    for i in range(epochs):
        # Same asymmetric star/light repeatedly appears on the left.
        rep = core.run_epoch(vis.drives(0.85, 0.10, uv=0.25, color=0.5), readouts,
                             neural_ms=80.0, learn=True)
        # Quiet gap: fast neural state continues; plastic memory does not reset.
        core.run_epoch([], readouts, neural_ms=40.0, learn=False)
        print(f"{i+1:02d}: net={rep.network_hz:6.3f}Hz  " +
              " ".join(f"{k}={v:6.2f}" for k,v in rep.readout_hz.items()) +
              f"  novelty={rep.novelty_hz:6.2f}  plastic_edges={core.memory_summary()['modified_edges']}")
    print("\nmemory:", json.dumps(core.memory_summary(), indent=2))
    if save:
        core.save_memory(save)
        print("saved:", save)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-memory")
    args = ap.parse_args()
    _demo(args.graph, args.epochs, args.seed, args.save_memory)


if __name__ == "__main__":
    main()
