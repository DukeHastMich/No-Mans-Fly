#!/usr/bin/env python3
"""Island of Dr hell no — Space Fly v0.

A deterministic, streamable procedural universe wrapped around the persistent
plastic MaleCNS core.

The experiment intentionally has no reward, punishment, hunger, damage, fuel
scarcity, death, target destination, or task.  Stars are simply there.  The
connectome receives visual state, learns from repeated activity, and its real
motor-neuron populations are reinterpreted as spacecraft thruster banks.

Important modeling boundary
---------------------------
The MaleCNS graph is biological.  The spacecraft is not.  Mapping fly motor
populations to thruster forces/torques is an explicit synthetic embodiment.
Likewise, until a graph with full retinotopic coordinates is exported, the
visual bridge is coarse and should not be described as a faithful fly retina.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from plastic_connectome import PlasticMaleCNSCore, PassiveVisualPorts


# ---------------------------------------------------------------------------
# deterministic procedural universe

@dataclass(frozen=True)
class Star:
    star_id: str
    position: np.ndarray
    luminosity: float
    temperature: float
    pulse_phase: float


class ProceduralUniverse:
    """Infinite-address deterministic sector generator.

    Only sectors near the spacecraft are instantiated.  A sector is regenerated
    byte-for-byte deterministically from universe seed + integer coordinates.
    """

    def __init__(
        self,
        seed: str = "no-mans-fly-001",
        *,
        sector_size: float = 30.0,
        mean_stars_per_sector: float = 4.0,
        view_sector_radius: int = 2,
        max_cached_sectors: int = 512,
    ):
        self.seed = str(seed)
        self.sector_size = float(sector_size)
        self.mean_stars_per_sector = float(mean_stars_per_sector)
        self.view_sector_radius = int(view_sector_radius)
        self.max_cached_sectors = int(max_cached_sectors)
        self._cache: OrderedDict[tuple[int, int, int], tuple[Star, ...]] = OrderedDict()

    def _seed_for(self, coord: tuple[int, int, int]) -> int:
        b = f"{self.seed}|{coord[0]}|{coord[1]}|{coord[2]}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(b).digest()[:8], "little", signed=False)

    def sector(self, coord: tuple[int, int, int]) -> tuple[Star, ...]:
        coord = tuple(int(x) for x in coord)
        if coord in self._cache:
            out = self._cache.pop(coord)
            self._cache[coord] = out
            return out
        rng = np.random.default_rng(self._seed_for(coord))
        n = max(0, int(rng.poisson(self.mean_stars_per_sector)))
        if n == 0:
            out: tuple[Star, ...] = ()
        else:
            xyz = (np.asarray(coord, dtype=np.float64) + rng.random((n, 3))) * self.sector_size
            # Mostly modest stars, with rare bright ones.  Values are rendering units,
            # not claims about astrophysical luminosity distributions.
            lum = np.clip(rng.lognormal(mean=0.0, sigma=1.0, size=n), 0.08, 22.0)
            temp = np.clip(rng.normal(loc=6100.0, scale=2300.0, size=n), 2200.0, 18000.0)
            phase = rng.random(n) * (2.0 * math.pi)
            items = []
            for i in range(n):
                sid = hashlib.sha1(f"{self.seed}:{coord}:{i}".encode()).hexdigest()[:12]
                items.append(Star(sid, xyz[i].astype(np.float64), float(lum[i]), float(temp[i]), float(phase[i])))
            out = tuple(items)
        self._cache[coord] = out
        while len(self._cache) > self.max_cached_sectors:
            self._cache.popitem(last=False)
        return out

    def local_stars(self, position: np.ndarray) -> list[Star]:
        c = np.floor(np.asarray(position, dtype=np.float64) / self.sector_size).astype(np.int64)
        r = self.view_sector_radius
        out: list[Star] = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    out.extend(self.sector((int(c[0] + dx), int(c[1] + dy), int(c[2] + dz))))
        return out

    def identity_probe(self, coord=(0, 0, 0)) -> str:
        """Digest proving regeneration of a sector is deterministic."""
        h = hashlib.sha256()
        for s in self.sector(tuple(coord)):
            h.update(s.star_id.encode())
            h.update(np.asarray(s.position, dtype="<f8").tobytes())
            h.update(np.asarray([s.luminosity, s.temperature], dtype="<f8").tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# spacecraft math / synthetic embodiment

def q_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dtype=np.float64)


def q_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) from body to world with quaternion q."""
    v = np.asarray(v, dtype=np.float64)
    single = v.ndim == 1
    vv = v.reshape((-1, 3))
    qv = q[1:]
    t = 2.0 * np.cross(np.broadcast_to(qv, vv.shape), vv)
    out = vv + q[0] * t + np.cross(np.broadcast_to(qv, vv.shape), t)
    return out[0] if single else out


def q_inverse_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return q_rotate(q_conj(q), v)


@dataclass
class SpacecraftState:
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray       # quaternion body -> world [w,x,y,z]
    angular_velocity: np.ndarray  # body radians/sec

    @classmethod
    def initial(cls):
        # A tiny harmless inertial drift/tumble makes the sky evolve even before
        # the motor system discovers how to actuate the synthetic body.
        return cls(
            position=np.array([3.0, 5.0, 7.0], dtype=np.float64),
            velocity=np.array([0.030, 0.008, -0.004], dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            angular_velocity=np.array([0.006, 0.011, 0.004], dtype=np.float64),
        )

    def advance(self, force_body: np.ndarray, torque_body: np.ndarray, dt: float):
        # Arbitrary benign simulation units.  No mass/fuel/damage model exists.
        force_world = q_rotate(self.orientation, np.asarray(force_body, dtype=np.float64))
        self.velocity += force_world * dt
        self.position += self.velocity * dt

        self.angular_velocity += np.asarray(torque_body, dtype=np.float64) * dt
        # Mild damping only prevents numerical runaway; no penalty is attached.
        self.angular_velocity *= math.exp(-0.025 * dt)
        w = self.angular_velocity
        mag = float(np.linalg.norm(w))
        if mag > 0:
            half = 0.5 * mag * dt
            dq = np.array([math.cos(half), *(math.sin(half) * w / mag)], dtype=np.float64)
            self.orientation = q_normalize(q_mul(self.orientation, dq))


class MotorThrusterBridge:
    """Map real motor-neuron anatomical groups onto synthetic thruster banks."""

    def __init__(self, graph_path: str | Path):
        z = np.load(graph_path, allow_pickle=True)
        self.superclass = np.asarray(z["superclass"], dtype=str)
        self.subclass = np.asarray(z["subclass"], dtype=str)
        self.side = np.asarray(z["side"], dtype=str)
        self.motor_mask = np.isin(self.superclass, ["vnc_motor", "cb_motor"])
        self.groups: dict[str, np.ndarray] = {}
        for sub in ("wm", "fl", "ml", "hl", "nm", "hm", "ad", "pm", "am", "rm", "xm"):
            for side in ("L", "R"):
                ids = np.flatnonzero(self.motor_mask & (self.subclass == sub) & (self.side == side)).astype(np.int32)
                if len(ids):
                    self.groups[f"{sub}_{side}"] = ids

    def readouts(self) -> dict[str, np.ndarray]:
        return self.groups

    @staticmethod
    def _sat(hz: float, scale: float = 35.0) -> float:
        return math.tanh(max(0.0, float(hz)) / scale)

    def actuate(self, rates: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Convert population activity to force/torque.

        This mapping is intentionally mechanical rather than semantic.  It does
        not claim that a foreleg motor neuron biologically means 'pitch'.
        """
        def lr(sub: str):
            L = self._sat(rates.get(f"{sub}_L", 0.0))
            R = self._sat(rates.get(f"{sub}_R", 0.0))
            return L, R, 0.5*(L+R), (R-L)

        wL,wR,wA,wD = lr("wm")   # wing motors: strongest axial bank
        fL,fR,fA,fD = lr("fl")   # fore banks
        mL,mR,mA,mD = lr("ml")   # middle banks
        hL,hR,hA,hD = lr("hl")   # hind banks
        nL,nR,nA,nD = lr("nm")   # neck: fine attitude
        aL,aR,aA,aD = lr("ad")   # abdominal: retro/roll bank
        halL,halR,halA,halD = lr("hm")

        # body axes: +x forward, +y right, +z up
        force = np.array([
            0.030*wA + 0.010*(fA+mA+hA) - 0.012*aA,
            0.012*(mD + 0.5*fD + 0.5*hD),
            0.010*(fA-hA),
        ], dtype=np.float64)
        torque = np.array([
            0.020*(wD + aD),                         # roll
            0.018*(fA-hA) + 0.006*nA,               # pitch
            0.022*(fD+mD+hD) + 0.010*nD + 0.005*halD, # yaw
        ], dtype=np.float64)
        diag = {
            "motor_mean": float(np.mean(list(rates.values()))) if rates else 0.0,
            "wing": wA,
            "fore": fA,
            "mid": mA,
            "hind": hA,
            "neck": nA,
            "abdomen": aA,
        }
        return force, torque, diag


# ---------------------------------------------------------------------------
# radiant thermal sensing -> real antennal thermoreceptor populations

SOLAR_CONSTANT_W_M2 = 1361.0  # bolometric irradiance of 1 L_sun at 1 AU


@dataclass
class ThermalFrame:
    """Physical state of the two synthetic arista thermal elements.

    The antennae do not detect "IR photons" as an image.  They are tiny thermal
    masses attached to a maintained body temperature.  Directional stellar
    radiation changes each element's temperature, and the biological hot/cool
    receptor populations receive only that temperature state/change.
    """
    left_temp_c: float
    right_temp_c: float
    left_dtemp_c_s: float
    right_dtemp_c_s: float
    left_flux_w_m2: float
    right_flux_w_m2: float
    left_ir_fraction: float
    right_ir_fraction: float
    dominant_thermal_id: str | None
    dominant_bearing_deg: float | None
    dominant_flux_w_m2: float


def _planck_ir_fraction(temp_k: float, lo_um: float = 0.75, hi_um: float = 14.0) -> float:
    """Fraction of blackbody radiant power in a broad near/mid-IR band.

    This is diagnostic/spectral weighting only; the arista ultimately senses
    temperature produced by absorbed *total* radiant energy.  Numerical
    integration of Planck's law is cheap here because star counts are small.
    """
    T = max(100.0, float(temp_k))
    lam = np.geomspace(lo_um * 1e-6, hi_um * 1e-6, 96)
    c2 = 1.438776877e-2  # h*c/k [m*K]
    x = np.clip(c2 / (lam * T), 1e-8, 700.0)
    spectral = 1.0 / (lam**5 * np.expm1(x))
    band = float(np.trapezoid(spectral, lam))
    # Integral over a broad reference wavelength range captures effectively all
    # stellar blackbody power for the temperatures generated by this universe.
    all_lam = np.geomspace(0.05e-6, 200e-6, 192)
    xa = np.clip(c2 / (all_lam * T), 1e-8, 700.0)
    total = float(np.trapezoid(1.0 / (all_lam**5 * np.expm1(xa)), all_lam))
    return float(np.clip(band / max(total, 1e-300), 0.0, 1.0))


class RadiantThermalSensor:
    """Two directional, inertial arista thermal elements.

    Geometry is synthetic spacecraft embodiment; thermal transduction is kept
    separate from the connectome.  One world distance unit is treated as 1 AU
    for radiometry.  Star ``luminosity`` is interpreted as solar luminosities.

    The temperature model is a per-area RC approximation:
        C_A dT/dt = alpha*F_abs - H*(T - T_body)
    where H lumps conductive return to the body plus small-signal radiative
    cooling.  It deliberately has no damage, pain, or avoidance term.
    """

    def __init__(
        self,
        *,
        body_temp_c: float = 25.0,
        absorptivity: float = 0.72,
        heat_capacity_area_j_m2_k: float = 4.0,
        thermal_conductance_w_m2_k: float = 18.0,
        antenna_yaw_deg: float = 48.0,
        body_shadow: float = 0.04,
    ):
        self.body_temp_c = float(body_temp_c)
        self.absorptivity = float(absorptivity)
        self.heat_capacity_area = float(heat_capacity_area_j_m2_k)
        self.thermal_conductance = float(thermal_conductance_w_m2_k)
        self.body_shadow = float(body_shadow)
        yaw = math.radians(float(antenna_yaw_deg))
        # body +x forward, +y right; left antenna normal points forward-left.
        self.left_normal = np.array([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float64)
        self.right_normal = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
        self.left_temp_c = self.body_temp_c
        self.right_temp_c = self.body_temp_c
        self._ir_cache: dict[int, float] = {}

    def _irfrac(self, temp_k: float) -> float:
        key = int(round(float(temp_k) / 25.0) * 25)
        if key not in self._ir_cache:
            self._ir_cache[key] = _planck_ir_fraction(float(key))
        return self._ir_cache[key]

    @staticmethod
    def _incidence(dirs_body: np.ndarray, normal: np.ndarray, body_shadow: float) -> np.ndarray:
        # Frontal cosine response plus a tiny diffuse/background contribution.
        # Sources behind the head are substantially shadowed rather than simply
        # reflected into a left/right magic channel.
        c = dirs_body @ normal
        front = np.clip(c, 0.0, 1.0)
        back = np.clip(-c, 0.0, 1.0) * body_shadow
        return front + back

    def observe(self, universe: ProceduralUniverse, craft: SpacecraftState, dt: float) -> ThermalFrame:
        stars = universe.local_stars(craft.position)
        if not stars:
            return self._advance_empty(dt)

        pos = np.vstack([s.position for s in stars])
        rel_world = pos - craft.position[None, :]
        dist_au = np.linalg.norm(rel_world, axis=1)
        good = dist_au > 1e-6
        if not np.any(good):
            return self._advance_empty(dt)

        rel_world = rel_world[good]
        dist_au = dist_au[good]
        ss = [s for i, s in enumerate(stars) if good[i]]
        dirs_world = rel_world / dist_au[:, None]
        dirs_body = q_inverse_rotate(craft.orientation, dirs_world)

        lum_solar = np.asarray([s.luminosity for s in ss], dtype=np.float64)
        bolometric = SOLAR_CONSTANT_W_M2 * lum_solar / np.maximum(dist_au * dist_au, 1e-12)
        li = self._incidence(dirs_body, self.left_normal, self.body_shadow)
        ri = self._incidence(dirs_body, self.right_normal, self.body_shadow)
        left_contrib = bolometric * li
        right_contrib = bolometric * ri
        left_flux = float(np.sum(left_contrib))
        right_flux = float(np.sum(right_contrib))

        temps = np.asarray([s.temperature for s in ss], dtype=np.float64)
        irf = np.asarray([self._irfrac(t) for t in temps], dtype=np.float64)
        left_ir = float(np.sum(left_contrib * irf) / max(left_flux, 1e-12)) if left_flux else 0.0
        right_ir = float(np.sum(right_contrib * irf) / max(right_flux, 1e-12)) if right_flux else 0.0

        # Dominant directional thermal source is based on whichever antenna gets
        # the stronger contribution from that source.
        dom_score = np.maximum(left_contrib, right_contrib)
        k = int(np.argmax(dom_score))
        dom = ss[k]
        bearing = math.degrees(math.atan2(float(dirs_body[k, 1]), float(dirs_body[k, 0])))

        old_l, old_r = self.left_temp_c, self.right_temp_c
        dt = max(1e-6, float(dt))
        # Exact first-order RC update for constant flux during this world step.
        tau = self.heat_capacity_area / max(self.thermal_conductance, 1e-9)
        eq_l = self.body_temp_c + self.absorptivity * left_flux / max(self.thermal_conductance, 1e-9)
        eq_r = self.body_temp_c + self.absorptivity * right_flux / max(self.thermal_conductance, 1e-9)
        a = 1.0 - math.exp(-dt / max(tau, 1e-9))
        self.left_temp_c += a * (eq_l - self.left_temp_c)
        self.right_temp_c += a * (eq_r - self.right_temp_c)
        dl = (self.left_temp_c - old_l) / dt
        dr = (self.right_temp_c - old_r) / dt

        return ThermalFrame(
            left_temp_c=float(self.left_temp_c), right_temp_c=float(self.right_temp_c),
            left_dtemp_c_s=float(dl), right_dtemp_c_s=float(dr),
            left_flux_w_m2=left_flux, right_flux_w_m2=right_flux,
            left_ir_fraction=left_ir, right_ir_fraction=right_ir,
            dominant_thermal_id=dom.star_id, dominant_bearing_deg=float(bearing),
            dominant_flux_w_m2=float(bolometric[k]),
        )

    def _advance_empty(self, dt: float) -> ThermalFrame:
        old_l, old_r = self.left_temp_c, self.right_temp_c
        tau = self.heat_capacity_area / max(self.thermal_conductance, 1e-9)
        a = 1.0 - math.exp(-float(dt) / max(tau, 1e-9))
        self.left_temp_c += a * (self.body_temp_c - self.left_temp_c)
        self.right_temp_c += a * (self.body_temp_c - self.right_temp_c)
        return ThermalFrame(
            float(self.left_temp_c), float(self.right_temp_c),
            float((self.left_temp_c-old_l)/max(float(dt),1e-9)),
            float((self.right_temp_c-old_r)/max(float(dt),1e-9)),
            0.0,0.0,0.0,0.0,None,None,0.0,
        )

    def snapshot(self) -> dict[str, float]:
        return {"left_temp_c": float(self.left_temp_c), "right_temp_c": float(self.right_temp_c)}

    def restore(self, state: dict):
        self.left_temp_c = float(state.get("left_temp_c", self.body_temp_c))
        self.right_temp_c = float(state.get("right_temp_c", self.body_temp_c))


class AntennalThermoBridge:
    """Temperature dynamics -> real MaleCNS thermoreceptor populations.

    VP2 TRNs are the hot/warming channel. VP3a/b TRNs are the cool channel.
    The mapping intentionally does not assign a value judgement: hot is not
    punishment and cool is not reward.
    """

    def __init__(self, core: PlasticMaleCNSCore):
        self.hot_l = core.ids(neuron_type="TRN_VP2", side="L")
        self.hot_r = core.ids(neuron_type="TRN_VP2", side="R")
        cool_l = [core.ids(neuron_type=t, side="L") for t in ("TRN_VP3a", "TRN_VP3b")]
        cool_r = [core.ids(neuron_type=t, side="R") for t in ("TRN_VP3a", "TRN_VP3b")]
        self.cool_l = np.unique(np.concatenate([x for x in cool_l if len(x)])).astype(np.int32) if any(len(x) for x in cool_l) else np.empty(0,dtype=np.int32)
        self.cool_r = np.unique(np.concatenate([x for x in cool_r if len(x)])).astype(np.int32) if any(len(x) for x in cool_r) else np.empty(0,dtype=np.int32)

    @staticmethod
    def _hot_drive(temp_c: float, dtemp_c_s: float) -> float:
        # Gr28b.d's reported warmth threshold is around 26 C.  Hot-cell firing is
        # slow-adapting and strongly temperature dependent (Q10 ~4.4); the capped
        # curve below preserves those qualitative properties without claiming a
        # fitted electrophysiological transfer function.
        if temp_c <= 25.8:
            return 0.0
        q10 = 4.4
        steady = max(0.0, q10 ** ((temp_c - 26.0) / 10.0) - 1.0)
        ref = q10 ** ((34.0 - 26.0) / 10.0) - 1.0
        norm = np.clip(steady / max(ref, 1e-9), 0.0, 1.0)
        warming = np.clip(max(0.0, dtemp_c_s) / 4.0, 0.0, 1.0)
        cooling_inhibition = np.clip(max(0.0, -dtemp_c_s) / 3.0, 0.0, 1.0)
        return float(np.clip((0.78*norm + 0.22*warming) * (1.0 - 0.7*cooling_inhibition), 0.0, 1.0) * 90.0)

    @staticmethod
    def _cool_drive(temp_c: float, dtemp_c_s: float) -> float:
        # Aristal cool cells have a strong transient response to cooling and are
        # inhibited by warming.  A small sustained component appears below the
        # maintained body reference temperature.
        transient = np.clip(max(0.0, -dtemp_c_s) / 2.0, 0.0, 1.0)
        sustained = np.clip(max(0.0, 25.0 - temp_c) / 5.0, 0.0, 1.0)
        warming_inhibition = np.clip(max(0.0, dtemp_c_s) / 2.0, 0.0, 1.0)
        return float(np.clip((0.82*transient + 0.18*sustained) * (1.0 - 0.85*warming_inhibition), 0.0, 1.0) * 90.0)

    def drives(self, frame: ThermalFrame) -> list[tuple[np.ndarray, float]]:
        vals = [
            (self.hot_l, self._hot_drive(frame.left_temp_c, frame.left_dtemp_c_s)),
            (self.hot_r, self._hot_drive(frame.right_temp_c, frame.right_dtemp_c_s)),
            (self.cool_l, self._cool_drive(frame.left_temp_c, frame.left_dtemp_c_s)),
            (self.cool_r, self._cool_drive(frame.right_temp_c, frame.right_dtemp_c_s)),
        ]
        return [(ids,hz) for ids,hz in vals if len(ids) and hz > 0.0]

# ---------------------------------------------------------------------------
# visual world -> coarse biological/synthetic visual ports

@dataclass
class VisualFrame:
    left: float
    right: float
    uv: float
    color: float
    motion: float
    expansion: float
    brightest_id: str | None
    brightest_strength: float
    visible_count: int


class StarfieldSensor:
    """Coarse all-sky sensor.

    It computes visual statistics from actual nearby procedural stars.  The left
    and right brightness channels feed broad photoreceptor populations.  Motion
    and expansion may additionally feed identified feature populations through
    SyntheticFeatureBridge because full retinal column geometry is not retained
    in the compact graph.
    """

    def __init__(self):
        self.prev_dirs: dict[str, np.ndarray] = {}
        self.prev_strength: dict[str, float] = {}

    def observe(self, universe: ProceduralUniverse, craft: SpacecraftState, time_s: float) -> VisualFrame:
        stars = universe.local_stars(craft.position)
        if not stars:
            return VisualFrame(0,0,0,0,0,0,None,0,0)
        pos = np.vstack([s.position for s in stars])
        rel_world = pos - craft.position[None, :]
        dist = np.linalg.norm(rel_world, axis=1)
        good = dist > 1e-6
        if not np.any(good):
            return VisualFrame(0,0,0,0,0,0,None,0,0)
        rel_world = rel_world[good]
        dist = dist[good]
        ss = [s for i,s in enumerate(stars) if good[i]]
        dirs_world = rel_world / dist[:,None]
        dirs_body = q_inverse_rotate(craft.orientation, dirs_world)

        # Mild front weighting but nearly panoramic, suitable for a fly-like body.
        front = 0.18 + 0.82*np.clip((dirs_body[:,0]+1.0)*0.5, 0.0, 1.0)
        lum = np.asarray([s.luminosity for s in ss])
        # Tiny deterministic pulse makes rare bright stars shimmer, not random flicker.
        pulse = 1.0 + 0.035*np.sin(0.17*time_s + np.asarray([s.pulse_phase for s in ss]))
        strength = lum*pulse*front/(0.20 + (dist/12.0)**2)

        # Robust exposure curve means one nearby star cannot numerically blind the sensor.
        total = float(np.sum(strength))
        exposure = 1.0 - math.exp(-total/14.0)
        left_raw = float(np.sum(strength[dirs_body[:,1] < 0]))
        right_raw = float(np.sum(strength[dirs_body[:,1] >= 0]))
        denom = max(1e-9, left_raw + right_raw)
        left = exposure * left_raw/denom * 2.0
        right = exposure * right_raw/denom * 2.0
        left = float(np.clip(left,0,1)); right=float(np.clip(right,0,1))

        temps = np.asarray([s.temperature for s in ss])
        weighted_temp = float(np.sum(temps*strength)/max(np.sum(strength),1e-9))
        uv = float(np.clip((weighted_temp-6500.0)/7000.0, 0.0, 1.0))
        # 'color' is simply chromatic richness away from solar-ish white.
        color = float(np.clip(abs(weighted_temp-5800.0)/7000.0, 0.0, 1.0))

        motion_terms=[]; expansion_terms=[]
        current_dirs={}; current_strength={}
        for s,d,st in zip(ss,dirs_body,strength):
            current_dirs[s.star_id]=d
            current_strength[s.star_id]=float(st)
            if s.star_id in self.prev_dirs:
                dot=float(np.clip(np.dot(d,self.prev_dirs[s.star_id]),-1,1))
                motion_terms.append(math.acos(dot)*min(1.0,float(st)))
                prev=self.prev_strength.get(s.star_id,float(st))
                if st>prev:
                    expansion_terms.append((float(st)-prev)/max(0.1,prev))
        self.prev_dirs=current_dirs; self.prev_strength=current_strength
        motion=float(np.clip(np.mean(motion_terms)*8.0 if motion_terms else 0.0,0,1))
        expansion=float(np.clip(np.mean(expansion_terms) if expansion_terms else 0.0,0,1))

        bi=int(np.argmax(strength))
        return VisualFrame(left,right,uv,color,motion,expansion,ss[bi].star_id,float(strength[bi]),len(ss))


class SyntheticFeatureBridge:
    """Temporary feature-level bridge for motion until retinotopy is retained."""

    def __init__(self, core: PlasticMaleCNSCore):
        self.lplc2 = core.ids(neuron_type="LPLC2")
        self.lc4 = core.ids(neuron_type="LC4")
        self.metu1 = core.ids(neuron_type="MeTu1")

    def drives(self, frame: VisualFrame) -> list[tuple[np.ndarray,float]]:
        # The bridge is deliberately mild; broad photoreceptors remain the primary input.
        brightness=max(frame.left,frame.right)
        return [
            (self.metu1, 18.0*brightness),
            (self.lplc2, 22.0*frame.expansion),
            (self.lc4, 28.0*frame.motion*frame.expansion),
        ]


# ---------------------------------------------------------------------------
# experiment runner

@dataclass
class FrameLog:
    step: int
    time_s: float
    position: list[float]
    velocity: list[float]
    left_light: float
    right_light: float
    left_antenna_c: float
    right_antenna_c: float
    left_thermal_flux_w_m2: float
    right_thermal_flux_w_m2: float
    thermal_bearing_deg: float | None
    thermal_source_id: str | None
    motion: float
    expansion: float
    brightest_id: str | None
    brightest_strength: float
    network_hz: float
    novelty_hz: float
    modified_edges: int
    motor_mean_hz: float
    force_body: list[float]
    torque_body: list[float]


class SpaceFlyExperiment:
    def __init__(self, graph: str | Path, *, universe_seed="no-mans-fly-001", neural_seed=7):
        self.graph=Path(graph)
        self.universe=ProceduralUniverse(universe_seed)
        self.core=PlasticMaleCNSCore(self.graph, seed=neural_seed)
        self.visual=PassiveVisualPorts(self.core)
        self.features=SyntheticFeatureBridge(self.core)
        self.sensor=StarfieldSensor()
        self.thermal=RadiantThermalSensor()
        self.thermo_bridge=AntennalThermoBridge(self.core)
        self.thrusters=MotorThrusterBridge(self.graph)
        self.craft=SpacecraftState.initial()
        self.time_s=0.0
        self.logs:list[FrameLog]=[]

    def step(self, *, world_dt=0.50, neural_ms=80.0, learn=True) -> FrameLog:
        frame=self.sensor.observe(self.universe,self.craft,self.time_s)
        thermal=self.thermal.observe(self.universe,self.craft,world_dt)
        drives=self.visual.drives(frame.left,frame.right,uv=frame.uv,color=frame.color,max_hz=120.0)
        drives.extend(self.features.drives(frame))
        drives.extend(self.thermo_bridge.drives(thermal))
        report=self.core.run_epoch(drives,self.thrusters.readouts(),neural_ms=neural_ms,learn=learn)
        force,torque,diag=self.thrusters.actuate(report.readout_hz)
        self.craft.advance(force,torque,world_dt)
        self.time_s += world_dt
        mem=self.core.memory_summary()
        log=FrameLog(
            step=len(self.logs), time_s=self.time_s,
            position=[float(x) for x in self.craft.position],
            velocity=[float(x) for x in self.craft.velocity],
            left_light=frame.left,right_light=frame.right,
            left_antenna_c=thermal.left_temp_c,right_antenna_c=thermal.right_temp_c,
            left_thermal_flux_w_m2=thermal.left_flux_w_m2,right_thermal_flux_w_m2=thermal.right_flux_w_m2,
            thermal_bearing_deg=thermal.dominant_bearing_deg,thermal_source_id=thermal.dominant_thermal_id,
            motion=frame.motion,expansion=frame.expansion,
            brightest_id=frame.brightest_id,brightest_strength=frame.brightest_strength,
            network_hz=report.network_hz,novelty_hz=report.novelty_hz,
            modified_edges=int(mem["modified_edges"]),
            motor_mean_hz=float(diag["motor_mean"]),
            force_body=[float(x) for x in force],torque_body=[float(x) for x in torque],
        )
        self.logs.append(log)
        return log

    def save(self, folder: str | Path):
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        mem_path=folder/"space_fly_memory.npz"
        self.core.save_memory(mem_path)
        state={
            "format":"Island-of-Dr-hell-no-space-fly-v0",
            "universe_seed":self.universe.seed,
            "time_s":self.time_s,
            "position":self.craft.position.tolist(),
            "velocity":self.craft.velocity.tolist(),
            "orientation":self.craft.orientation.tolist(),
            "angular_velocity":self.craft.angular_velocity.tolist(),
            "thermal_state":self.thermal.snapshot(),
            "universe_sector_000_digest":self.universe.identity_probe((0,0,0)),
            "memory_file":mem_path.name,
            "logs":[asdict(x) for x in self.logs],
        }
        (folder/"space_fly_state.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
        return state

    @classmethod
    def resume(cls, graph: str | Path, folder: str | Path, *, neural_seed: int = 7):
        """Restore learned memory plus spacecraft/world state from a checkpoint."""
        folder = Path(folder)
        state = json.loads((folder/"space_fly_state.json").read_text(encoding="utf-8"))
        exp = cls(graph, universe_seed=state["universe_seed"], neural_seed=neural_seed)
        exp.core.load_memory(folder/state["memory_file"])
        exp.time_s = float(state["time_s"])
        exp.craft.position = np.asarray(state["position"], dtype=np.float64)
        exp.craft.velocity = np.asarray(state["velocity"], dtype=np.float64)
        exp.craft.orientation = q_normalize(np.asarray(state["orientation"], dtype=np.float64))
        exp.craft.angular_velocity = np.asarray(state["angular_velocity"], dtype=np.float64)
        if "thermal_state" in state:
            exp.thermal.restore(state["thermal_state"])
        # The previous visual-frame cache is intentionally rebuilt. Therefore the
        # first resumed frame reports zero optical motion; long-term neural memory is
        # preserved, but one-frame visual differentiation is not serialized in v0.
        exp.logs = []
        expected = state.get("universe_sector_000_digest")
        if expected and exp.universe.identity_probe((0,0,0)) != expected:
            raise ValueError("Universe seed no longer regenerates the checkpointed sky")
        return exp


def run_cli(args):
    if args.resume_dir:
        exp=SpaceFlyExperiment.resume(args.graph,args.resume_dir,neural_seed=args.neural_seed)
        print("resumed:",args.resume_dir)
    else:
        exp=SpaceFlyExperiment(args.graph,universe_seed=args.universe_seed,neural_seed=args.neural_seed)
    print("No Man's Fly — Space Fly v0")
    print("No goal. No reward. No punishment. Infinite-address deterministic starfield.\n")
    print("sector(0,0,0):",exp.universe.identity_probe((0,0,0))[:20])
    print("motor neurons mapped:",sum(len(v) for v in exp.thrusters.groups.values()),"across",len(exp.thrusters.groups),"banks")
    for _ in range(args.steps):
        x=exp.step(world_dt=args.world_dt,neural_ms=args.neural_ms,learn=not args.no_learn)
        print(f"{x.step:03d} t={x.time_s:5.1f}s pos=({x.position[0]:7.3f},{x.position[1]:7.3f},{x.position[2]:7.3f}) "
              f"light={x.left_light:.3f}/{x.right_light:.3f} thermal={x.left_antenna_c:.2f}/{x.right_antenna_c:.2f}C "
              f"bearing={x.thermal_bearing_deg if x.thermal_bearing_deg is not None else float('nan'):+6.1f}deg bright={x.brightest_id} "
              f"net={x.network_hz:6.3f}Hz novelty={x.novelty_hz:6.2f} motor={x.motor_mean_hz:6.2f}Hz "
              f"learned={x.modified_edges}")
    if args.save_dir:
        exp.save(args.save_dir)
        print("saved:",args.save_dir)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--graph",required=True)
    ap.add_argument("--steps",type=int,default=12)
    ap.add_argument("--world-dt",type=float,default=0.5)
    ap.add_argument("--neural-ms",type=float,default=80.0)
    ap.add_argument("--universe-seed",default="no-mans-fly-001")
    ap.add_argument("--neural-seed",type=int,default=7)
    ap.add_argument("--save-dir")
    ap.add_argument("--resume-dir")
    ap.add_argument("--no-learn",action="store_true")
    run_cli(ap.parse_args())


if __name__ == "__main__":
    main()
