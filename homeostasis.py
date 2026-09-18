#!/usr/bin/env python3
"""Bounded homeostasis and real-population interoceptive bridge for Specimen-003.

The internal variables remain bounded and reversible.  There is no starvation damage,
pain channel, death, tissue damage, or panic escalation.  Hunger is deliberately split
into a slow nutritional state and a transient expressed hunger drive so that the neural
signal is not a permanently pegged fuel gauge.

Food approach never restores nutritional satiety.  Improving food evidence can only
briefly suppress a bounded fraction of the expressed hunger drive, and that relief fades
again if feeding does not occur.  This is engineering scaffolding, not a claim of exact
Drosophila endocrine kinetics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class HomeostasisConfig:
    # Slow nutritional state.
    satiety_decay_per_s: float = 0.00075
    energy_decay_idle_per_s: float = 0.00012
    energy_cost_flight_per_s: float = 0.00055
    energy_cost_walk_per_s: float = 0.00020
    nutrient_to_satiety: float = 0.50
    nutrient_to_energy: float = 0.38

    # ``energy`` is short-term activity reserve, not total body calories.  A fly at
    # zero reserve is fatigued, not metabolically impossible.  Recovery is permitted
    # only when activity is genuinely low; Patch 7 accidentally let the reserve refill
    # during hard flight.  Feeding remains the only way to restore nutritional satiety.
    energy_rest_recovery_tau_s: float = 18.0
    energy_rest_ceiling_min: float = 0.30
    energy_emergency_ceiling: float = 0.12
    energy_emergency_recovery_tau_s: float = 90.0
    # No perpetual motor floor.  Mechanical power now falls continuously to zero as
    # short-term reserve is exhausted; true rest can mobilize the emergency reserve.
    energy_motor_power_floor: float = 0.0

    thermal_preferred_c: float = 27.0
    thermal_comfort_half_band_c: float = 3.0
    thermal_soft_scale_c: float = 9.0

    hunger_weight: float = 0.52
    thermal_weight: float = 0.35
    energy_weight: float = 0.13
    valence_gain: float = 7.0

    # Expressed hunger dynamics.  Nutritional hunger itself is still 1-satiety.
    # The pang oscillator only modulates the neural drive; it is intentionally excluded
    # from reinforcement error so a free-running oscillator cannot reward arbitrary acts.
    hunger_pang_period_s: float = 38.0
    hunger_pang_depth: float = 0.08

    # Anticipatory food relief.  Even at maximum relief, at least 82% of nutritional
    # hunger remains.  Thus smelling/approaching food can take the edge off but can
    # never produce the absurd "never mind, not hungry" fly-by state.
    approach_relief_max_fraction: float = 0.18
    approach_relief_tau_s: float = 7.5
    odor_relief_rate_per_s: float = 0.035
    odor_rise_relief_gain: float = 1.8
    distance_progress_relief_gain: float = 1.2
    contact_relief_rate_per_s: float = 0.10
    ingestion_relief_gain: float = 3.0

    # Neural transfer caps.  These are intentionally modest and bounded.
    npf_hunger_max_hz: float = 28.0
    hugin_satiety_max_hz: float = 18.0
    lk_satiety_max_hz: float = 14.0
    dh44_ingestion_max_hz: float = 24.0
    ingestion_trace_tau_s: float = 2.5


@dataclass
class HomeostaticState:
    satiety: float = 0.72
    energy: float = 0.82
    thermal_c: float = 27.0

    # nutritional_hunger is the slow need state. hunger is the expressed neural drive.
    nutritional_hunger: float = 0.28
    hunger: float = 0.28
    hunger_pang: float = 1.0
    hunger_suppression: float = 0.0
    approach_relief: float = 0.0
    hunger_phase_rad: float = math.pi / 2.0

    # Previous food evidence is persisted so approach is a consequence of real sensory
    # progress across steps, not a hidden steering vector.
    last_food_odor: float = 0.0
    last_food_distance: float = 1.0e30
    last_food_resource_id: str = ""

    thermal_error: float = 0.0
    energy_deficit: float = 0.18
    total_error: float = 0.0
    valence: float = 0.0
    ingestion_trace: float = 0.0
    consumed_total: float = 0.0


class HomeostaticSystem:
    """Persistent bounded internal state.

    ``valence`` means change in homeostatic error during the last world step.  Neither
    the hunger-pang oscillator nor anticipatory food-cue relief alters nutritional error.
    Smelling/touching food may transiently suppress expressed hunger, but only actual
    ingestion restores satiety/energy and can create nutritional positive reinforcement.
    """

    FORMAT = "no-mans-fly-homeostasis-v1"

    def __init__(self, config: HomeostasisConfig | None = None):
        self.config = config or HomeostasisConfig()
        self.state = HomeostaticState()
        self._recompute(thermal_c=self.state.thermal_c, previous_error=None)

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(float(x), 0.0, 1.0))

    def energy_recovery_ceiling(self) -> float:
        """Maximum short-term reserve recoverable without eating.

        Even a nutritionally empty animal can restore some immediate muscle reserve
        from internal stores, but cannot recover to a fully fueled state by resting.
        """
        c, s = self.config, self.state
        lo = self._clip01(c.energy_rest_ceiling_min)
        return self._clip01(lo + (1.0-lo)*self._clip01(s.satiety))

    def motor_power_scale(self) -> float:
        """Mechanical power available to expressed motor output.

        Zero reserve means the short-term mechanical reserve is exhausted.  This does
        not erase neural intent: the motor neurons may keep firing, but muscle power
        falls to zero until rest or ingestion restores reserve.  Emergency recovery is
        handled by ``advance`` only when actual expressed activity is low.
        """
        c, s = self.config, self.state
        floor = self._clip01(c.energy_motor_power_floor)
        return self._clip01(floor + (1.0-floor)*math.sqrt(self._clip01(s.energy)))

    def _thermal_error(self, temp_c: float) -> float:
        c = self.config
        d = abs(float(temp_c) - c.thermal_preferred_c)
        excess = max(0.0, d - c.thermal_comfort_half_band_c)
        return float(1.0 - math.exp(-excess / max(c.thermal_soft_scale_c, 1e-9)))

    def _express_hunger(self) -> tuple[float, float, float]:
        c, s = self.config, self.state
        base = self._clip01(1.0 - s.satiety)
        relief = self._clip01(s.approach_relief)
        pang = self._clip01(0.5 * (1.0 + math.sin(float(s.hunger_phase_rad))))

        # The oscillator fades the signal between pangs; approach provides an additional
        # bounded suppression.  Both are multiplicative, so hunger can never be driven
        # below zero and food approach can never erase a severe need state.
        pang_factor = 1.0 - c.hunger_pang_depth * (1.0 - pang)
        relief_factor = 1.0 - c.approach_relief_max_fraction * relief
        expressed = self._clip01(base * pang_factor * relief_factor)
        suppression = 0.0 if base <= 1e-12 else self._clip01(1.0 - expressed / base)
        return base, pang, suppression

    def _recompute(self, *, thermal_c: float, previous_error: float | None):
        c, s = self.config, self.state
        s.thermal_c = float(thermal_c)
        base, pang, suppression = self._express_hunger()
        s.nutritional_hunger = base
        s.hunger_pang = pang
        s.hunger_suppression = suppression
        s.hunger = self._clip01(base * (1.0 - suppression))
        s.energy_deficit = self._clip01(1.0 - s.energy)
        s.thermal_error = self._thermal_error(s.thermal_c)

        # Patch19: reinforcement uses the slow nutritional need itself.  Odor, approach
        # and contact can change the *expressed* hunger signal but cannot make the
        # organism nutritionally better.  This prevents the pathological attractor where
        # standing on food was positively reinforced despite zero ingestion.
        hunger_for_error = base
        err = (c.hunger_weight*hunger_for_error
               + c.thermal_weight*s.thermal_error
               + c.energy_weight*s.energy_deficit)
        s.total_error = self._clip01(err)
        if previous_error is None:
            s.valence = 0.0
        else:
            s.valence = float(np.clip((float(previous_error)-s.total_error)*c.valence_gain, -1.0, 1.0))

    def _update_food_relief(self, dt: float, *, food_odor: float, food_distance: float,
                            food_resource_id: str | None, food_contact: bool,
                            nutrient_consumed: float):
        c, s = self.config, self.state
        if dt > 0.0 and c.approach_relief_tau_s > 0.0:
            s.approach_relief *= math.exp(-dt / c.approach_relief_tau_s)

        odor = self._clip01(food_odor)
        rid = "" if food_resource_id is None else str(food_resource_id)
        dist = float(food_distance)
        if not math.isfinite(dist):
            dist = 1.0e30

        same = bool(rid and rid == s.last_food_resource_id)
        odor_rise = max(0.0, odor - float(s.last_food_odor)) if same else 0.0
        progress = 0.0
        # Distance is only allowed to contribute while a food odor is actually
        # present. This avoids smuggling a hidden resource waypoint into homeostasis.
        # The distance term merely stabilizes an already sensed odor gradient.
        odor_visible = max(odor, float(s.last_food_odor)) >= 0.01
        if same and odor_visible and math.isfinite(float(s.last_food_distance)) and dist < float(s.last_food_distance):
            progress = min(1.0, (float(s.last_food_distance) - dist) / max(float(s.last_food_distance), 0.25))

        # Existing odor provides a very small tonic easing; rising odor and actual
        # approach provide stronger transient evidence.  None of these alter satiety.
        add = c.odor_relief_rate_per_s * odor * dt
        add += c.odor_rise_relief_gain * odor_rise
        add += c.distance_progress_relief_gain * progress
        if food_contact:
            add += c.contact_relief_rate_per_s * dt
        if nutrient_consumed > 0.0:
            add += c.ingestion_relief_gain * float(nutrient_consumed)
        s.approach_relief = self._clip01(s.approach_relief + add)

        s.last_food_odor = odor
        s.last_food_distance = dist
        s.last_food_resource_id = rid

    def advance(self, dt_s: float, *, thermal_c: float, flight_command: float = 0.0,
                walking_command: float = 0.0, nutrient_consumed: float = 0.0,
                food_odor: float = 0.0, food_distance: float = float("inf"),
                food_resource_id: str | None = None, food_contact: bool = False) -> HomeostaticState:
        c, s = self.config, self.state
        dt = max(0.0, float(dt_s))
        prev = float(s.total_error)

        s.satiety = self._clip01(s.satiety - c.satiety_decay_per_s*dt)

        # Short-term activity reserve: exertion drains it and *actual low activity*
        # restores it toward a satiety-dependent ceiling.  Patch 7 used
        # ``(1-activity)^2`` plus an unconditional emergency term, which could make a
        # starving fly recharge while still flying at 70-80% command.  Recovery is now
        # explicitly gated off during meaningful locomotion.
        flight = self._clip01(flight_command)
        walk = self._clip01(walking_command)
        activity_load = self._clip01(max(flight, 0.70*walk))
        activity_cost = (c.energy_decay_idle_per_s
                         + c.energy_cost_flight_per_s*flight
                         + c.energy_cost_walk_per_s*walk)
        s.energy = self._clip01(s.energy - activity_cost*dt)

        if dt > 0.0:
            ceiling = self.energy_recovery_ceiling()

            # Full recovery only at rest; it fades rapidly and reaches exactly zero
            # by 30% activity.  This is intentionally a gate, not a soft tail that can
            # overpower exertion at high flight command.
            rest_gate = self._clip01((0.30-activity_load)/0.30)
            rest_factor = rest_gate*rest_gate
            if c.energy_rest_recovery_tau_s > 0.0 and rest_factor > 0.0 and s.energy < ceiling:
                a = 1.0-math.exp(-dt*rest_factor/c.energy_rest_recovery_tau_s)
                s.energy += a*(ceiling-s.energy)

            # Emergency reserve mobilization exists only at very low activity.  It can
            # help an exhausted animal recover from the rail while resting, but can no
            # longer act as an alternator during flight.
            emergency_ceiling = min(ceiling, self._clip01(c.energy_emergency_ceiling))
            emergency_gate = self._clip01((0.18-activity_load)/0.18)
            if (c.energy_emergency_recovery_tau_s > 0.0 and emergency_gate > 0.0
                    and s.energy < emergency_ceiling):
                a = 1.0-math.exp(-dt*emergency_gate/c.energy_emergency_recovery_tau_s)
                s.energy += a*(emergency_ceiling-s.energy)
            s.energy = self._clip01(s.energy)

        eaten = max(0.0, float(nutrient_consumed))
        if eaten:
            s.satiety = self._clip01(s.satiety + c.nutrient_to_satiety*eaten)
            s.energy = self._clip01(s.energy + c.nutrient_to_energy*eaten)
            s.consumed_total += eaten
            s.ingestion_trace = self._clip01(s.ingestion_trace + eaten)

        if c.ingestion_trace_tau_s > 0 and dt > 0:
            s.ingestion_trace *= math.exp(-dt/c.ingestion_trace_tau_s)

        if c.hunger_pang_period_s > 0 and dt > 0:
            s.hunger_phase_rad = float((s.hunger_phase_rad + 2.0*math.pi*dt/c.hunger_pang_period_s) % (2.0*math.pi))

        self._update_food_relief(dt, food_odor=food_odor, food_distance=food_distance,
                                 food_resource_id=food_resource_id, food_contact=bool(food_contact),
                                 nutrient_consumed=eaten)
        self._recompute(thermal_c=thermal_c, previous_error=prev)
        return s

    def snapshot(self) -> dict[str, Any]:
        return {"format": self.FORMAT, "config": asdict(self.config), "state": asdict(self.state)}

    def restore(self, payload: dict[str, Any]):
        if payload.get("format") not in (None, self.FORMAT):
            raise ValueError("Unknown homeostasis snapshot format")
        if "config" in payload:
            cfg=dict(payload["config"])
            # Migrate the old Patch7/19 emergency 45% motor floor.  It could turn a
            # zero-reserve brain command into perpetual mechanical flight.  This is an
            # embodiment migration only; homeostatic state and neural state are intact.
            if abs(float(cfg.get("energy_motor_power_floor",0.0))-0.45) < 1e-12:
                cfg["energy_motor_power_floor"]=0.0
            self.config = HomeostasisConfig(**cfg)
        raw = dict(payload.get("state", {}))
        self.state = HomeostaticState(**raw)

        # Old checkpoints predate the two-layer hunger model.  Preserve their exact
        # first-epoch hunger semantics instead of briefly resetting a starved specimen
        # to the dataclass default before the first world step.
        if "nutritional_hunger" not in raw:
            base = self._clip01(1.0 - self.state.satiety)
            self.state.nutritional_hunger = base
            self.state.hunger = self._clip01(float(raw.get("hunger", base)))
            self.state.hunger_phase_rad = math.pi / 2.0
            self.state.hunger_pang = 1.0
            self.state.approach_relief = 0.0
            self.state.hunger_suppression = 0.0 if base <= 1e-12 else self._clip01(1.0 - self.state.hunger/base)
            self.state.last_food_odor = 0.0
            self.state.last_food_distance = 1.0e30
            self.state.last_food_resource_id = ""

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8"); return p

    @classmethod
    def load(cls, path: str | Path) -> "HomeostaticSystem":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        x = cls(HomeostasisConfig(**d.get("config", {}))); x.restore(d); return x


class InteroceptiveNutrientBridge:
    """Conservative internal nutrient-state drive into identified sensor/modulator cells.

    This is *not* an action script and is kept separate from environmental sensory
    input.  It approximates missing hemolymph/gut physiology at known state-sensitive
    populations while avoiding Patch19's broad name matching.

    - NPFL1-I: a retained NPF population whose activity/signaling is hunger sensitive;
      receives a modest signal from slow nutritional deficit only.
    - DH44: the six retained PI endocrine neurons are post-ingestive nutritive-sugar
      sensors; drive requires both recent ingestion and nutritional need.
    - Hugin-RG: retained endocrine Hugin cells get only a recent-circulating-nutrient
      signal, consistent with glucose-responsive fed-state Hugin circuitry.

    No LK, AstA, descending neuron, motor neuron, or arbitrary substring match is
    stimulated.  Downstream effects must propagate through the connectome.
    """
    NPF_MAX_HZ=16.0
    DH44_MAX_HZ=22.0
    HUGIN_MAX_HZ=14.0

    def __init__(self,core):
        self.core=core
        self.npf_l1=core.ids(neuron_type="NPFL1-I")
        self.dh44=core.ids(neuron_type="DH44")
        self.hugin_rg=core.ids(neuron_type="Hugin-RG")

    def drives(self,state: HomeostaticState) -> list[tuple[np.ndarray,float]]:
        hunger=float(np.clip(state.nutritional_hunger,0.0,1.0))
        recent=float(np.clip(state.ingestion_trace,0.0,1.0))
        out=[]
        # Slow nutritional deficit only: odor/contact 'relief' cannot fake satiety here.
        if len(self.npf_l1) and hunger>0.0:
            out.append((self.npf_l1,self.NPF_MAX_HZ*hunger))
        # DH44 glucose/nutrient sensing is strongest in starved animals and only after
        # nutrient actually enters the body.
        if len(self.dh44) and recent>0.0 and hunger>0.0:
            out.append((self.dh44,self.DH44_MAX_HZ*recent*hunger))
        # Hugin is a glucose-responsive energy/satiety sensor.  Recent ingestion is the
        # represented physical proxy; no generic host 'satiety command' is used.
        if len(self.hugin_rg) and recent>0.0:
            out.append((self.hugin_rg,self.HUGIN_MAX_HZ*recent))
        return out

    def census(self) -> dict[str,int]:
        return {"NPFL1-I":int(len(self.npf_l1)),"DH44":int(len(self.dh44)),
                "Hugin-RG":int(len(self.hugin_rg)),
                "accidental_DNp29_drive":0,"LK_direct_drive":0}


# Compatibility name for older imports.  Semantics are Patch20's exact bridge.
HomeostaticNeuralBridge = InteroceptiveNutrientBridge
