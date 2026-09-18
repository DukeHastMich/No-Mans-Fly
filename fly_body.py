#!/usr/bin/env python3
"""Anatomical motor embodiment for Specimen-003.

Rule: preserve first, interpret second.
Every MaleCNS motor neuron is read independently. Known anatomical categories are
mapped to fly-like effectors; unknown categories are retained in ``unresolved``
telemetry rather than silently discarded or converted into arbitrary steering.

This is still a mechanical abstraction. It is deliberately more faithful than the
Specimen-002 generic thruster bridge, but it does not claim muscle-level force curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Any

import numpy as np


def _sat(hz: float, scale: float = 35.0) -> float:
    return math.tanh(max(0.0, float(hz)) / scale)


def _async_flight_power(hz: float) -> float:
    """Decode generic-LIF power-MN activity without treating it as literal wingbeat rate.

    Drosophila DLM/DVM power motor neurons normally modulate asynchronous flight muscle
    at only a few to ~12 Hz, whereas this generic LIF core can report much larger spike
    rates for the same structural circuit.  A separate, deliberately broad transfer
    scale preserves graded neural control while preventing 50-150 Hz model activity
    from becoming an automatic 100% mechanical throttle.  This is a decoder calibration,
    not a neural clamp: the underlying rates and plastic state are untouched.
    """
    return math.tanh(max(0.0, float(hz)) / 110.0)


@dataclass
class EffectorState:
    # normalized observer/mechanical states [-1,1] or [0,1]
    wing_left: float = 0.0
    wing_right: float = 0.0
    takeoff_left: float = 0.0
    takeoff_right: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    antenna_left: float = 0.0
    antenna_right: float = 0.0
    abdomen_bend: float = 0.0
    proboscis_extension: float = 0.0
    pharyngeal_pump: float = 0.0
    # Mechanical fill fraction of the cibarium.  This is body state, not neural state.
    cibarium_fill: float = 0.0
    haltere_left: float = 0.0
    haltere_right: float = 0.0
    foreleg_left: float = 0.0
    foreleg_right: float = 0.0
    midleg_left: float = 0.0
    midleg_right: float = 0.0
    hindleg_left: float = 0.0
    hindleg_right: float = 0.0


@dataclass
class MotorFrame:
    force_body: np.ndarray
    torque_body: np.ndarray
    effectors: EffectorState
    neuron_rates: dict[int, float]
    subclass_rates: dict[str, dict[str, float]]
    unresolved: dict[int, dict[str, Any]]
    mean_motor_hz: float
    feeding_command: float
    walking_command: float
    flight_command: float
    takeoff_command: float


class AnatomicalMotorBridge:
    """Read all mapped motor neurons and express known motor anatomy."""

    def __init__(self, core):
        self.core = core
        mask = np.isin(core.superclass.astype(str), ["vnc_motor", "cb_motor"])
        self.motor_indices = np.flatnonzero(mask).astype(np.int32)
        self.motor_bodies = core.bodies[self.motor_indices].astype(np.int64)
        self.by_body = {int(b): int(i) for b, i in zip(self.motor_bodies, self.motor_indices)}

        # One readout per motor neuron: nothing is averaged away before observation.
        self._readouts = {f"motor:{int(core.bodies[i])}": np.asarray([i], dtype=np.int32) for i in self.motor_indices}
        self.last_frame: MotorFrame | None = None

        # Cibarial mechanics persist across neural epochs.  MN12 fills/opens the chamber;
        # a later MN11 phase empties it toward the foregut.  The old same-bin harmonic
        # mean required both phases inside one 100-ms readout window and could therefore
        # reject a perfectly valid sequential swallow.
        self.cibarium_fill = 0.0
        self._last_cibarium_empty_fraction = 0.0

    def snapshot(self) -> dict[str, float]:
        return {"cibarium_fill": float(np.clip(self.cibarium_fill, 0.0, 1.0))}

    def restore(self, payload: dict[str, Any] | None):
        d = payload or {}
        self.cibarium_fill = float(np.clip(float(d.get("cibarium_fill", 0.0)), 0.0, 1.0))
        self._last_cibarium_empty_fraction = 0.0

    def readouts(self) -> dict[str, np.ndarray]:
        return self._readouts

    @staticmethod
    def _side_sign(side: str, instance: str) -> int:
        s = str(side).strip().lower()
        if s in ("l", "left") or str(instance).lower().endswith("_l"):
            return -1
        if s in ("r", "right") or str(instance).lower().endswith("_r"):
            return +1
        return 0

    @staticmethod
    def _joint_direction(type_name: str) -> float:
        t = type_name.lower()
        # Positive roughly means extension/protraction, negative flexion/retraction.
        if any(k in t for k in ("extensor", "promotor", "anterior rotator")):
            return +1.0
        if any(k in t for k in ("flexor", "remotor", "reductor", "posterior rotator")):
            return -1.0
        if "sternotrochanter" in t or "tergotr" in t:
            return +0.35
        return 0.0

    @staticmethod
    def _wing_steering_muscle(type_name: str) -> str | None:
        """Return a documented direct wing-steering muscle name, if resolved.

        MaleCNS names are explicit (``b2 MN``, ``iii1 MN``, etc.).  Indirect-control
        muscles (tp/ps), putative iii4/MNwm35, and unknown MNwm36 stay mechanically
        unresolved here rather than being folded into a generic steering pool.
        """
        s=str(type_name).strip().lower()
        for name in ("iii1","iii3","hg1","hg2","hg3","hg4","b1","b2","b3","i1","i2"):
            if s.startswith(name+" ") or s==name or s.startswith(name+"mn"):
                return name
        return None

    def _rates(self, readout_hz: dict[str, float]) -> dict[int, float]:
        return {body: float(readout_hz.get(f"motor:{body}", 0.0)) for body in self.by_body}

    def actuate(self, readout_hz: dict[str, float], *, on_surface: bool = False,
                dt_s: float = 0.10, update_mechanics: bool = True) -> MotorFrame:
        c = self.core
        rates = self._rates(readout_hz)
        subclass_sum: dict[tuple[str, str], list[float]] = {}
        unresolved: dict[int, dict[str, Any]] = {}

        # Accumulators are normalized activity commands, not literal muscle Newtons.
        wing = {-1: [], +1: []}; flight_steer = {-1: [], +1: []}; takeoff = {-1: [], +1: []}
        steering_muscles={m:{-1:[],+1:[]} for m in ("b1","b2","b3","i1","i2","iii1","iii3","hg1","hg2","hg3","hg4")}
        legs = {"fl": {-1: [], +1: []}, "ml": {-1: [], +1: []}, "hl": {-1: [], +1: []}}
        neck = {-1: [], +1: []}; haltere = {-1: [], +1: []}; abdomen = {-1: [], +1: []}
        antenna = {-1: [], +1: []}; proboscis = {-1: [], +1: []}
        # Feeding motor phases are kept separate using the current MaleCNS cell
        # types.  The older embodiment collapsed every MN11 into an "empty" pool and
        # required MN12D alone to fill the cibarium.  That is incompatible with the
        # MaleCNS anatomy: MN11V and MN12D innervate the ventral pharyngeal/cibarial
        # dilator, MN11D participates later in the pump, and MN10 completes transfer
        # toward the esophagus.  MN12V is retained if a future graph supplies it, but
        # the 2026 MaleCNS gustatory census found no anatomical MN12V match.
        pump_expand_12=[]; pump_expand_11v=[]; pump_propulse_11d=[]; pump_transfer=[]; pump_tonic=[]

        for body, idx in self.by_body.items():
            hz = rates[body]
            a = _sat(hz)
            sub = str(c.subclass[idx]); typ = str(c.types[idx]); inst = str(c.instance[idx])
            ss = self._side_sign(str(c.side[idx]), inst)
            side_key = "L" if ss < 0 else ("R" if ss > 0 else "?")
            subclass_sum.setdefault((sub, side_key), []).append(hz)

            if sub == "wm":
                # DLM/DVM are major flight power muscles; TTM is takeoff/jump; the
                # remaining wing MNs are kept as steering/power contributors.
                if "dlmn" in typ.lower() or "dvmn" in typ.lower():
                    if ss: wing[ss].append(_async_flight_power(hz))
                elif "ttmn" in typ.lower():
                    # TTM is a takeoff/jump actuator; keep it separate from ordinary
                    # wing steering so a brief burst is not diluted by dozens of MNs.
                    if ss: takeoff[ss].append(a)
                else:
                    # Preserve every direct/indirect wing-control MN independently in
                    # telemetry, but do not collapse their scalar rates into a generic
                    # left-minus-right yaw command.  Dipteran steering muscles have
                    # muscle-specific, often phase-dependent functions: e.g. b1 turn
                    # information is encoded substantially by spike phase, while b1/b2
                    # also contribute to pitch control; i/iii/hg muscles show different
                    # ipsilateral/contralateral recruitment during turns.  This LIF
                    # embodiment does not yet resolve wingbeat phase, so a pooled yaw
                    # interpretation manufactures steering that the connectome did not
                    # specify. Keep named-muscle recruitment for the reduced hinge
                    # model below; the pooled steering activity is diagnostic only.
                    if ss:
                        flight_steer[ss].append(a)
                        sm=self._wing_steering_muscle(typ)
                        if sm is not None:
                            steering_muscles[sm][ss].append(a)
            elif sub in legs:
                d = self._joint_direction(typ)
                # Unknown named leg MNs still contribute weakly to limb activation,
                # while identified flexor/extensor labels preserve direction.
                if ss: legs[sub][ss].append(a * (d if d != 0 else 0.25))
            elif sub == "nm":
                if ss: neck[ss].append(a)
            elif sub == "hm":
                if ss: haltere[ss].append(a)
            elif sub == "ad":
                if ss: abdomen[ss].append(a)
            elif sub == "pm":
                # Cibarial phases are not interchangeable.  Preserve the typed
                # MaleCNS outputs instead of reducing "MN11" to one generic action.
                tl = typ.lower()
                if tl.startswith("mn12"):
                    pump_expand_12.append(a)
                elif tl.startswith("mn11v"):
                    pump_expand_11v.append(a)
                elif tl.startswith("mn11d"):
                    pump_propulse_11d.append(a)
                elif tl.startswith("mn10"):
                    pump_transfer.append(a)
                elif tl.startswith("mn5"):
                    pump_tonic.append(a)
                elif ss:
                    # MN1 is strongly retracting; MN9/2/6/8 are extension sequence.
                    direction = -1.0 if (tl == "mn1" or tl.startswith("mn1_")) else +1.0
                    proboscis[ss].append(direction*a)
            elif sub == "am":
                # MaleCNS AM cells fasciculate with the antennal nerve. Preserve them
                # as antennal actuator channels without pretending to know each muscle.
                if ss: antenna[ss].append(a)
            elif sub == "xm":
                # MNxm03 is independently annotated as a haltere motor neuron in
                # comparative datasets. Other xm cells remain unresolved rather than
                # being assigned an invented muscle target.
                if typ.lower().startswith("mnxm03") and ss:
                    haltere[ss].append(a)
                else:
                    unresolved[body] = {"rate_hz":hz,"subclass":sub,"type":typ,"instance":inst,"side":side_key}
            elif sub == "rm":
                # Central motor cells with currently unresolved peripheral target.
                unresolved[body] = {"rate_hz":hz,"subclass":sub,"type":typ,"instance":inst,"side":side_key}
            else:
                unresolved[body] = {"rate_hz":hz,"subclass":sub,"type":typ,"instance":inst,"side":side_key}

        mean = lambda xs: float(np.mean(xs)) if xs else 0.0
        WL,WR = mean(wing[-1]),mean(wing[+1])
        SL,SR = mean(flight_steer[-1]),mean(flight_steer[+1])
        steer_activity={m:{-1:mean(v[-1]),+1:mean(v[+1])} for m,v in steering_muscles.items()}
        # Muscle-specific turn recruitment from dipteran/Drosophila physiology.
        # +1: higher activity on a side accompanies turns toward that same (inner)
        # side.  -1: higher activity accompanies the outer side of a turn.  b1 is
        # deliberately omitted because its steering information is primarily spike
        # phase relative to the wingbeat, which this 100-ms LIF embodiment cannot
        # represent honestly.  i2/hg2 have minor/non-directional evidence and are also
        # omitted.
        turn_role={"b2":-1.0,"b3":+1.0,"i1":+1.0,"iii1":-1.0,
                   "iii3":-1.0,"hg1":+1.0,"hg3":+1.0,"hg4":+1.0}
        turn_weight={"b2":1.0,"b3":0.65,"i1":1.0,"iii1":1.0,
                     "iii3":0.65,"hg1":1.0,"hg3":1.0,"hg4":0.65}
        yaw_num=0.0; yaw_den=0.0
        for m,role in turn_role.items():
            w=float(turn_weight[m]); pair=steer_activity[m]
            yaw_num += w*role*(float(pair[+1])-float(pair[-1]))
            yaw_den += w
        steering_yaw=float(np.clip(yaw_num/max(yaw_den,1e-9),-1.0,1.0))
        TL,TR = mean(takeoff[-1]),mean(takeoff[+1])
        FL,FR = mean(legs['fl'][-1]),mean(legs['fl'][+1])
        ML,MR = mean(legs['ml'][-1]),mean(legs['ml'][+1])
        HL,HR = mean(legs['hl'][-1]),mean(legs['hl'][+1])
        NL,NR = mean(neck[-1]),mean(neck[+1])
        HalL,HalR = mean(haltere[-1]),mean(haltere[+1])
        AL,AR = mean(abdomen[-1]),mean(abdomen[+1])
        AntL,AntR = mean(antenna[-1]),mean(antenna[+1])
        PL,PR = mean(proboscis[-1]),mean(proboscis[+1])
        expand12,expand11v = mean(pump_expand_12),mean(pump_expand_11v)
        propulse11d = mean(pump_propulse_11d)
        transfer,tonic = mean(pump_transfer),mean(pump_tonic)

        # Stateful cibarial mechanics.  In the available MaleCNS graph MN11V is a
        # bona-fide pharyngeal-pumping output and shares cibarial/ventral-dilator
        # innervation with MN12D.  Either measured expansion channel can therefore
        # expand the chamber; requiring silent/uncertain MN12D as the sole gate made
        # the body mechanically deaf to active MN11V.  Later MN11D propulsion and MN10
        # transfer are combined geometrically so neither one alone fabricates a swallow.
        # Emptying still uses only material present at the *start* of this motor epoch,
        # preserving a real fill -> later propulsion sequence instead of an if-food-eat
        # shortcut.
        fill=float(np.clip(max(expand12,expand11v),0.0,1.0))
        late=float(np.sqrt(max(0.0,float(propulse11d))*max(0.0,float(transfer))))
        dt=max(0.0,float(dt_s))
        old_fill=float(np.clip(self.cibarium_fill,0.0,1.0))
        empty_capacity=float(np.clip(late,0.0,1.0))
        emptied=min(old_fill, empty_capacity * min(1.0, dt/0.10)) if update_mechanics else 0.0
        remaining=max(0.0,old_fill-emptied)
        fill_add=fill * max(0.0,1.0-remaining) * min(1.0,dt/0.10)
        new_fill=float(np.clip(remaining+fill_add,0.0,1.0))
        if update_mechanics:
            self.cibarium_fill=new_fill
            self._last_cibarium_empty_fraction=float(emptied)
        else:
            new_fill=old_fill
        PUMP=float(np.clip(emptied/max(min(1.0,dt/0.10),1e-9),0.0,1.0)) if update_mechanics else 0.0

        e = EffectorState(
            # Power and steering muscles remain distinct.  Direct steering activity
            # changes force/torque below; it no longer adds hidden symmetric throttle
            # that clips both wings to 1.0.
            wing_left=np.clip(WL,0,1), wing_right=np.clip(WR,0,1),
            takeoff_left=np.clip(TL,0,1), takeoff_right=np.clip(TR,0,1),
            head_yaw=np.clip(NR-NL,-1,1), head_pitch=np.clip(0.5*(NR+NL),-1,1),
            antenna_left=np.clip(AntL,0,1), antenna_right=np.clip(AntR,0,1),
            abdomen_bend=np.clip(AR-AL,-1,1), proboscis_extension=np.clip(0.5*(PL+PR),-1,1),
            pharyngeal_pump=np.clip(PUMP,0,1), cibarium_fill=np.clip(new_fill,0,1),
            haltere_left=np.clip(HalL,0,1), haltere_right=np.clip(HalR,0,1),
            foreleg_left=np.clip(FL,-1,1), foreleg_right=np.clip(FR,-1,1),
            midleg_left=np.clip(ML,-1,1), midleg_right=np.clip(MR,-1,1),
            hindleg_left=np.clip(HL,-1,1), hindleg_right=np.clip(HR,-1,1),
        )

        wing_avg=0.5*(e.wing_left+e.wing_right); wing_diff=e.wing_right-e.wing_left
        takeoff_avg=0.5*(e.takeoff_left+e.takeoff_right)
        # SL/SR remain observer diagnostics only.  Raw steering-MN firing is not a
        # scalar yaw command: individual steering muscles have different and sometimes
        # phase-coded effects.  Until wingbeat-phase mechanics are represented, using
        # their pooled side difference as yaw would add a host-authored behavior.
        leg_forward=0.0
        leg_pitch=0.0
        leg_yaw=0.0
        if on_surface:
            # Stance-like retraction/flexion propels the body forward; protraction
            # can pull backward. Zero leg activity produces zero translational force.
            leg_forward=0.055*np.clip(-(FL+FR+ML+MR+HL+HR)/6.0, -1, 1)
            leg_pitch=0.018*((FL+FR)-(HL+HR))
            leg_yaw=0.035*np.clip((FR-FL)+(MR-ML)+(HR-HL),-2,2)

        # Synthetic body mechanics driven by actual motor anatomy. Space remains a
        # benign medium for the experiment; this is not vacuum aerodynamics.
        force=np.array([
            (0.050*wing_avg + leg_forward + (0.035*takeoff_avg if on_surface else 0.0)),
            # Left/right power asymmetry remains a real mechanical asymmetry.  Direct
            # steering-MN rates are not added here until their muscle-specific hinge
            # mechanics and wingbeat phase are represented.
            0.012*wing_diff,
            0.030*wing_avg + (0.090*takeoff_avg if on_surface else 0.0),
        ],dtype=np.float64)
        # Reduced wing-hinge mechanics: b1/b2 advance the basalare, b3
        # opposes it. Activity is a bounded recruitment proxy, NOT measured
        # spike phase. Neutral antagonist balance leaves the established force
        # envelope unchanged. Calibration is documented in FLIGHT_MODEL.md.
        stroke={side:float(np.clip(
            0.5*(steer_activity['b1'][side]+steer_activity['b2'][side])
            -steer_activity['b3'][side],-1.,1.)) for side in (-1,+1)}
        powered_stroke=0.5*(WL*stroke[-1]+WR*stroke[+1])
        # x forward, z dorsal: anterior upward force gives NEGATIVE y torque
        # (nose up). No power means no aerodynamic pitch or yaw torque.
        flight_pitch=-0.018*powered_stroke
        torque=np.array([
            0.030*wing_diff + 0.008*e.abdomen_bend,
            # Neck effectors move the head; they do not directly rotate the thorax.
            # Leg posture contributes through contact; the wing hinge supplies flight pitch.
            leg_pitch + flight_pitch,
            # Free-flight yaw comes only from documented muscle-specific steering
            # recruitment, never from the old pooled left-minus-right steering mean.
            # Surface leg asymmetry can additionally yaw through real ground contact.
            leg_yaw + 0.018*wing_avg*steering_yaw,
        ],dtype=np.float64)

        subclass_rates: dict[str,dict[str,float]]={}
        for (sub,side),vals in subclass_sum.items():
            subclass_rates.setdefault(sub,{})[side]=float(np.mean(vals)) if vals else 0.0

        motor_values=list(rates.values())
        frame=MotorFrame(
            force_body=force, torque_body=torque, effectors=e, neuron_rates=rates,
            subclass_rates=subclass_rates, unresolved=unresolved,
            mean_motor_hz=float(np.mean(motor_values)) if motor_values else 0.0,
            feeding_command=float(np.clip(max(0.0,e.proboscis_extension)*0.45 + e.pharyngeal_pump*0.55,0,1)),
            walking_command=float(np.clip(np.mean(np.abs([FL,FR,ML,MR,HL,HR])),0,1)),
            flight_command=float(np.clip(max(wing_avg,takeoff_avg),0,1)),
            takeoff_command=float(np.clip(takeoff_avg,0,1)),
        )
        self.last_frame=frame
        return frame

    def census(self) -> dict[str,Any]:
        c=self.core
        out={"motor_neurons":int(len(self.motor_indices)),"by_subclass":{},"unresolved_subclasses":[],
             "partially_resolved_subclasses":[],"known_resolved_motor_neurons":0,"preserved_unresolved_motor_neurons":0}
        known={"wm","fl","ml","hl","nm","hm","ad","pm","am"}
        for sub in sorted(set(str(c.subclass[i]) for i in self.motor_indices)):
            ids=[i for i in self.motor_indices if str(c.subclass[i])==sub]
            out["by_subclass"][sub]=len(ids)
            if sub=="xm":
                resolved=sum(1 for i in ids if str(c.types[i]).lower().startswith("mnxm03"))
                out["known_resolved_motor_neurons"] += resolved
                out["preserved_unresolved_motor_neurons"] += len(ids)-resolved
                if resolved and resolved<len(ids): out["partially_resolved_subclasses"].append(sub)
                elif resolved==0: out["unresolved_subclasses"].append(sub)
            elif sub in known:
                out["known_resolved_motor_neurons"] += len(ids)
            else:
                out["preserved_unresolved_motor_neurons"] += len(ids)
                out["unresolved_subclasses"].append(sub)
        return out
