#!/usr/bin/env python3
"""No Man's Fly — Specimen-003 motivated full-brain life experiment.

This is the branch where the connectome gets *a life* rather than a waypoint.
No downstream steering command is injected.  The environment supplies photons,
thermal state, odor/taste/contact and bounded internal physiology.  MaleCNS motor
neurons drive fly-like effectors.  Environmental drives are restricted to retained
sensory neurons; interoceptive nutrient signals use explicitly identified endogenous
state-sensitive cells.  There is no host steering/action script or global reward wire.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from full_plastic_connectome import FullMaleCNSCore, FullPlasticityConfig
from full_connectome_evolution import GrowthPressureTracker
from compound_eye import (EyeOpticsConfig, SyntheticHexEyeMap, CompoundEyeRenderer,
                          HofbauerBuchnerEyeletBridge, summarize_eye_map)
from space_fly_universe import AntennalThermoBridge, q_normalize, q_rotate, q_mul
from living_universe import LivingUniverse, FlyBodyState, LivingThermalSensor, ContactState
from life_sensory import FoodSensoryBridge, PlanetarySensoryBridge, HygroThermoSupplementBridge
from homeostasis import HomeostaticSystem, InteroceptiveNutrientBridge
from fly_body import AnatomicalMotorBridge, MotorFrame
from proprioception import ProprioceptiveBridge


@dataclass
class LifeFrameLog:
    step: int
    time_s: float
    position: list[float]
    velocity: list[float]
    on_surface: bool
    body_id: str | None
    altitude: float
    altitude_body_id: str | None
    atmosphere_fraction: float
    gravity_body_id: str | None
    gravity_accel: float
    slip_angle_deg: float
    forward_speed: float
    nearest_resource_id: str | None
    nearest_resource_distance: float
    odor: float
    world_scent: float
    dominant_world_scent: float
    sensed_worlds: int
    sensory_world_id: str | None
    sensory_world_distance: float
    sensory_world_bearing_yaw_deg: float | None
    resource_contact: bool
    mouth_contact: bool
    proboscis_contact: float
    nutrient_consumed: float
    consumed_total: float
    starvation_rescue_s: float
    divine_fruit_drops: int
    satiety: float
    nutritional_hunger: float
    hunger: float
    hunger_pang: float
    hunger_suppression: float
    approach_relief: float
    energy: float
    motor_energy_scale: float
    thermal_c: float
    thermal_error: float
    homeostatic_error: float
    valence_for_next_epoch: float
    network_hz: float
    novelty_hz: float
    keepalive_wave_phase: float
    keepalive_wave_active_neurons: int
    keepalive_wave_peak_mv: float
    motor_mean_hz: float
    flight_command: float
    takeoff_command: float
    walking_command: float
    feeding_command: float
    modified_edges: int
    dopamine_mean: float
    octopamine_mean: float
    serotonin_mean: float
    growth_candidate_count: int
    top_growth_candidate_body: int | None
    eye_peak_side: str
    eye_peak_yaw_deg: float
    eye_peak_elevation_deg: float
    visible_sources: int
    antenna_left_c: float
    antenna_right_c: float
    relative_humidity: float
    dry_sense: float
    moist_sense: float
    proboscis_extension: float
    pharyngeal_pump: float
    cibarium_fill: float
    force_body: list[float]
    torque_body: list[float]


class Specimen003Experiment:
    FORMAT = "no-mans-fly-specimen003-life-v1"
    MECHANICS_REVISION = "organism-health1-turn-bias1"
    FEEDING_RECOVERY_REVISION = "feeding-path1-electrical-discharge-v1"

    # Environmental starvation safeguard. Nothing here drives neurons, rewards an
    # action, or feeds the animal directly: after sustained severe nutritional need,
    # the universe merely materializes a rich fruit patch at the nearest surface.
    DIVINE_FRUIT_STARVATION_DELAY_S = 60.0
    DIVINE_FRUIT_COOLDOWN_S = 180.0
    DIVINE_FRUIT_NEARBY_RADIUS = 0.90

    def __init__(self, graph: str | Path, *, universe_seed: str="no-mans-fly-003",
                 neural_seed: int=11, optics: EyeOpticsConfig | None=None,
                 core_config: FullPlasticityConfig | None=None, auto_tune_threads: bool=True,
                 resolved_eye_map: SyntheticHexEyeMap | None=None):
        self.graph=Path(graph)
        self.universe=LivingUniverse(universe_seed)
        self.core=FullMaleCNSCore(self.graph,seed=neural_seed,config=core_config,auto_tune=auto_tune_threads)
        if resolved_eye_map is None:
            self.eye_map=SyntheticHexEyeMap(self.graph,config=optics)
        else:
            # Structural growth appends neurons and must not recalibrate existing
            # photoreceptors. Validate the already-resolved calibration directly
            # against the new core, then reuse it without loading the graph again.
            em=resolved_eye_map
            idx=np.asarray(em.neuron_index,dtype=np.int64)
            if np.any(idx<0) or np.any(idx>=self.core.n):
                raise ValueError('resolved eye map contains neuron index outside new graph')
            if not np.array_equal(self.core.bodies[idx],np.asarray(em.body_id,dtype=np.int64)):
                raise ValueError('resolved eye map photoreceptor bodies do not match new graph')
            if not np.array_equal(self.core.types[idx].astype(str),np.asarray(em.receptor_type).astype(str)):
                raise ValueError('resolved eye map receptor types do not match new graph')
            self.eye_map=em; self.eye_map.graph_path=self.graph
        self.eye=CompoundEyeRenderer(self.eye_map)
        self.eyelet=HofbauerBuchnerEyeletBridge(self.core)
        self.thermal=LivingThermalSensor()
        self.thermo_bridge=AntennalThermoBridge(self.core)
        self.food=FoodSensoryBridge(self.core)
        self.planetary=PlanetarySensoryBridge(self.core)
        self.hygro=HygroThermoSupplementBridge(self.core)
        self.homeostasis=HomeostaticSystem()
        # New specimens start on the current recovery revision.  Legacy W925+
        # checkpoints may receive one narrowly-scoped transient electrical discharge
        # on resume; learned memory and slow neuromodulatory state are never reset.
        self.feeding_recovery_revision=self.FEEDING_RECOVERY_REVISION
        # Patch20: exact, conservative interoception replaces Patch19's broad
        # NPF/Hugin/DH44/LK substring stimulation.  No behavior command is encoded.
        self.homeo_bridge=InteroceptiveNutrientBridge(self.core)
        self.motor=AnatomicalMotorBridge(self.core)
        self.proprio=ProprioceptiveBridge(self.core)
        self.body=FlyBodyState.initial()
        self.body._set_flight_diagnostics(self.universe)
        self.growth_pressure=GrowthPressureTracker(self.core.n)
        self.time_s=0.0
        self.step_count=0
        self.pending_valence=0.0
        self.logs:list[LifeFrameLog]=[]
        self.events:list[dict[str,Any]]=[]

        # Observer-only most recent frame state.
        self.last_visual=None; self.last_thermal=None; self.last_report=None
        self.last_motor:MotorFrame|None=None; self.last_food={}; self.last_growth_candidates=[]
        self.last_consumed=0.0; self.last_force=np.zeros(3); self.last_torque=np.zeros(3)
        self.last_hygro={"relative_humidity":0.52,"atmosphere_fraction":0.0,"dry_drive":0.0,"moist_drive":0.0,"cool_drive":0.0}
        self.last_planetary={"dominant_world_id":None,"dominant_world_distance":float("inf"),"dominant_world_bearing_yaw_deg":None,"world_scent":0.0,"dominant_world_scent":0.0,"sensed_worlds":0,"world_carrier_flow_body":[0.0,0.0,0.0]}
        self.starvation_rescue_s=0.0
        self.divine_fruit_drops=0
        self.last_divine_fruit_time_s=-1.0e30

    EXTERNAL_SENSORY_SUPERCLASSES = {
        "vnc_sensory", "cb_sensory", "ol_sensory", "sensory_ascending",
        "sensory_descending", "vnc_sensory_tbc", "cb_sensory_tbc",
        "sensory_ascending_tbc",
    }

    def _validate_external_drives(self, drives):
        """Fail closed if an environmental/body bridge tries to stimulate non-sensory CNS cells."""
        out=[]
        sc=self.core.superclass.astype(str)
        allowed=self.EXTERNAL_SENSORY_SUPERCLASSES
        for ids,hz in drives:
            ids=np.asarray(ids,dtype=np.int32)
            if not len(ids) or float(hz)<=0.0:
                continue
            bad=ids[np.asarray([sc[int(i)] not in allowed for i in ids],dtype=bool)]
            if len(bad):
                sample=[(int(i),str(self.core.types[int(i)]),str(sc[int(i)])) for i in bad[:8]]
                raise RuntimeError(f"Patch20 rejected external drive into non-sensory neurons: {sample}")
            out.append((ids,float(hz)))
        return out

    def eye_mapping_digest(self) -> str:
        h=hashlib.sha256(); h.update(self.eye_map.mapping_mode.encode())
        h.update(np.asarray(self.eye_map.axes,dtype="<f4").tobytes())
        h.update(np.asarray(self.eye_map.body_id,dtype="<i8").tobytes())
        h.update(np.asarray(self.eye_map.neuron_facet,dtype="<i4").tobytes())
        h.update("|".join(self.eye_map.receptor_type.tolist()).encode())
        return h.hexdigest()

    def _event(self, kind:str, **payload):
        x={"time_s":float(self.time_s),"kind":str(kind),**payload}; self.events.append(x)
        if len(self.events)>1000: self.events=self.events[-1000:]

    def drop_emergency_care_package(self) -> dict[str,Any] | None:
        """Drop one rescue fruit into the physical world at Yar's current location.

        This is deliberately environmental intervention only.  It does not touch
        neural activity, reward/valence, motor commands, learning, or growth.
        Yar must still detect and consume the resource through the ordinary sensory
        and feeding path.  The starvation timer is reset so the automatic safeguard
        does not immediately duplicate a manually requested care package.
        """
        dropped=self.universe.drop_divine_fruit(self.body.position,self.body.contact.body_id)
        if dropped is None:
            return None
        db,dr,dd=dropped
        self.divine_fruit_drops += 1
        self.last_divine_fruit_time_s=float(self.time_s)
        self.starvation_rescue_s=0.0
        info={
            "resource_id":dr.patch_id,
            "body_id":db.body_id,
            "distance":float(dd),
            "nutrient_density":float(dr.nutrient_density),
            "fermentation":float(dr.fermentation),
            "amount":float(dr.amount),
        }
        self._event("emergency_care_package_drop",**info)
        return info











    def _refresh_body_spatial_frame(self):
        b,alt,norm=self.universe.nearest_body(self.body.position)
        _,rr,rd=self.universe.nearest_resource(self.body.position)
        self.body.contact=ContactState(
            on_surface=False,body_id=None,surface_temp_c=None,normal_world=None,
            altitude=float(alt),altitude_body_id=None if b is None else b.body_id,
            atmosphere_fraction=float(self.universe.atmosphere_fraction(self.body.position,b)),
            nearest_resource_id=None if rr is None else rr.patch_id,nearest_resource_distance=float(rd))
        self.body._set_flight_diagnostics(self.universe)


    def step(self, *, world_dt:float=0.10, neural_ms:float=100.0, learn:bool=True) -> LifeFrameLog:
        dt=max(1e-6,float(world_dt))
        was_surface=bool(self.body.contact.on_surface)
        old_body_id=self.body.contact.body_id

        # 1) Current external/internal sensory state enters real retained populations.
        visual=self.eye.observe_sequence(self.universe,self.body,self.time_s,max(1e-4,float(neural_ms)/1000.0))
        thermal=self.thermal.observe_living(self.universe,self.body,dt)
        food_drives,food_pre=self.food.observe(self.universe,self.body,self.last_motor)
        planetary_drives,planetary_pre=self.planetary.observe(self.universe,self.body)
        hygro_drives,hygro=self.hygro.drives(self.universe,self.body,thermal,food_pre)
        drives=self.eye.drives(visual)
        drives.extend(self.eyelet.drives(visual))
        drives.extend(self.thermo_bridge.drives(thermal))
        drives.extend(hygro_drives)
        drives.extend(food_drives)
        drives.extend(planetary_drives)
        # Body-state feedback is from the previous expressed motor frame. It carries
        # no target or reward semantics; it tells retained mechanosensory populations
        # what the animal actually did. Odor cues remain in the chemical pathways.
        # Odor gradients are concentration fields, not air velocity. They continue
        # to drive ORNs, but cannot bend an antenna without modeled carrier motion.
        ambient_flow=np.zeros(3,dtype=float)
        drives.extend(self.proprio.drives(self.body,self.last_motor,
                                         ambient_flow_body=ambient_flow,dt_s=dt))
        # Environmental/body drive is restricted to anatomically sensory populations.
        drives=self._validate_external_drives(drives)
        # Internal nutrient sensing is a separate physiological path: only exact
        # NPF-L1, DH44 and Hugin-RG populations are eligible, never motor/action cells.
        drives.extend(self.homeo_bridge.drives(self.homeostasis.state))

        # 2) No host-level reward wire.  The connectome retains its endogenous
        # monoaminergic wiring and unsupervised plasticity, but body error is not injected
        # as a global reinforcement scalar.
        report=self.core.run_epoch(drives,self.motor.readouts(),neural_ms=neural_ms,
                                   learn=learn,homeostatic_valence=0.0)
        growth_candidates=self.growth_pressure.observe(self.core) if getattr(self.core,'_learning_window',None) is None else []

        # 3) Preserve every motor neuron independently; resolved effectors act on body.
        # Commands completed at the end of this neural slice apply to the NEXT
        # physics interval. Continue the already expressed motor activity meanwhile.
        pending = getattr(self, 'pending_motor_rates', None)
        if pending is None:
            pending = {f'motor:{k}':float(v) for k,v in self.last_motor.neuron_rates.items()} if self.last_motor is not None else {}
        motor=self.motor.actuate(pending,on_surface=self.body.contact.on_surface,dt_s=dt)
        self.pending_motor_rates=dict(report.readout_hz)
        motor_energy_scale=self.homeostasis.motor_power_scale()
        self.body.advance_living(self.universe,motor.force_body,motor.torque_body,dt,
                                 flight_command=motor.flight_command,walking_command=motor.walking_command,
                                 motor_power_scale=motor_energy_scale,
                                 flight_force_body=motor.flight_force_body,
                                 contact_force_body=motor.contact_force_body,
                                 flight_torque_body=motor.flight_torque_body,
                                 contact_torque_body=motor.contact_torque_body)
        # 4) Re-observe contact after motion. Feeding only succeeds if real motor output
        # expresses a feeding command while a nutrient patch is actually under the fly.
        _,food_post=self.food.observe(self.universe,self.body,motor)
        consumed=self.food.consume_if_feeding(self.universe,self.body,motor,food_post,dt)

        # 5) Needs drift/recover.  No injury or death state exists.  The resulting
        # valence is used on the next neural epoch rather than commanding behavior.
        thermal_c=0.5*(float(thermal.left_temp_c)+float(thermal.right_temp_c))
        # Reserve tracks expressed mechanical work, not an impossible neural command
        # that exhausted muscles cannot execute.  This also lets a completely fatigued
        # animal genuinely rest and mobilize emergency reserve without rewriting its
        # motor-neuron activity.
        hs=self.homeostasis.advance(dt,thermal_c=thermal_c,
                                    flight_command=motor.flight_command*motor_energy_scale,
                                    walking_command=motor.walking_command*motor_energy_scale,
                                    nutrient_consumed=consumed,
                                    food_odor=float(food_post.get("odor",0.0)),
                                    food_distance=float(food_post.get("distance",float("inf"))),
                                    food_resource_id=food_post.get("resource_id"),
                                    food_contact=bool(food_post.get("contact",False) or food_post.get("mouth_contact",False)))
        # Homeostatic improvement/worsening remains diagnostic only.  Keeping the
        # persisted field at zero prevents an old checkpoint's pending scalar from being
        # accidentally reintroduced as host reinforcement.
        self.pending_valence=0.0
        self.time_s += dt

        # Persistent travel atlas.  Only entering a world's modeled local atmosphere
        # (or touching its surface) counts as an encounter, so deep-space nearest-body
        # bookkeeping does not fill the atlas with worlds Yar never actually visited.
        atlas_bid=self.body.contact.body_id if self.body.contact.on_surface else self.body.contact.altitude_body_id
        if atlas_bid is not None and (self.body.contact.on_surface or float(self.body.contact.atmosphere_fraction)>1.0e-9):
            atlas_body=self.universe.body_by_id(str(atlas_bid))
            self.universe.record_world_visit(atlas_body,time_s=self.time_s,altitude=float(self.body.contact.altitude),
                                             on_surface=bool(self.body.contact.on_surface),position=self.body.position)
        else:
            self.universe.record_world_visit(None,time_s=self.time_s,altitude=float('inf'),on_surface=False,position=self.body.position)

        # Environmental fail-safe against an indefinitely starved specimen. This is
        # deliberately world-side rather than neural-side: no novelty pulse, valence,
        # feeding command, or steering is injected. A maximally nutritious/fermented
        # resource simply appears on the nearest physical surface after sustained
        # severe hunger. If Yar is standing on a body it lands under his current feet.
        severe_starvation=bool(float(hs.satiety)<=0.02 and float(hs.nutritional_hunger)>=0.98)
        if consumed>0.0 or not severe_starvation:
            self.starvation_rescue_s=0.0
        else:
            self.starvation_rescue_s += dt
        can_drop=(self.starvation_rescue_s>=self.DIVINE_FRUIT_STARVATION_DELAY_S and
                  (self.time_s-self.last_divine_fruit_time_s)>=self.DIVINE_FRUIT_COOLDOWN_S and
                  self.universe.active_divine_fruit_distance(self.body.position)>self.DIVINE_FRUIT_NEARBY_RADIUS)
        if can_drop:
            dropped=self.universe.drop_divine_fruit(self.body.position,self.body.contact.body_id)
            if dropped is not None:
                db,dr,dd=dropped
                self.divine_fruit_drops += 1
                self.last_divine_fruit_time_s=float(self.time_s)
                self.starvation_rescue_s=0.0
                self._event("divine_fruit_drop",resource_id=dr.patch_id,body_id=db.body_id,
                            distance=float(dd),nutrient_density=float(dr.nutrient_density),
                            fermentation=float(dr.fermentation),amount=float(dr.amount))

        if self.body.contact.on_surface and (not was_surface or self.body.contact.body_id!=old_body_id):
            self._event("landed",body_id=self.body.contact.body_id,surface_temp_c=self.body.contact.surface_temp_c)
        if was_surface and not self.body.contact.on_surface:
            self._event("takeoff",body_id=old_body_id)
        if consumed>0:
            self._event("fed",resource_id=food_post.get("resource_id"),amount=float(consumed),satiety=float(hs.satiety))

        mem=self.core.memory_summary()
        mod=mem.get("modulator_mean",{})
        log=LifeFrameLog(
            step=int(self.step_count),time_s=float(self.time_s),
            position=[float(x) for x in self.body.position],velocity=[float(x) for x in self.body.velocity],
            on_surface=bool(self.body.contact.on_surface),body_id=self.body.contact.body_id,
            altitude=float(self.body.contact.altitude),altitude_body_id=self.body.contact.altitude_body_id,
            atmosphere_fraction=float(self.body.contact.atmosphere_fraction),gravity_body_id=self.body.gravity_body_id,
            gravity_accel=float(np.linalg.norm(self.body.gravity_world)),slip_angle_deg=float(self.body.slip_angle_deg),
            forward_speed=float(self.body.forward_speed),nearest_resource_id=food_post.get("resource_id"),
            nearest_resource_distance=float(food_post.get("distance",float("inf"))),odor=float(food_post.get("odor",0.0)),
            world_scent=float(planetary_pre.get("world_scent",0.0)),
            dominant_world_scent=float(planetary_pre.get("dominant_world_scent",0.0)),
            sensed_worlds=int(planetary_pre.get("sensed_worlds",0)),
            sensory_world_id=planetary_pre.get("dominant_world_id"),
            sensory_world_distance=float(planetary_pre.get("dominant_world_distance",float("inf"))),
            sensory_world_bearing_yaw_deg=planetary_pre.get("dominant_world_bearing_yaw_deg"),
            resource_contact=bool(food_post.get("contact",False)),mouth_contact=bool(food_post.get("mouth_contact",False)),
            proboscis_contact=float(food_post.get("proboscis_contact",0.0)),nutrient_consumed=float(consumed),
            consumed_total=float(hs.consumed_total),starvation_rescue_s=float(self.starvation_rescue_s),
            divine_fruit_drops=int(self.divine_fruit_drops),satiety=float(hs.satiety),
            nutritional_hunger=float(hs.nutritional_hunger),hunger=float(hs.hunger),
            hunger_pang=float(hs.hunger_pang),hunger_suppression=float(hs.hunger_suppression),
            approach_relief=float(hs.approach_relief),energy=float(hs.energy),
            motor_energy_scale=float(motor_energy_scale),
            thermal_c=float(hs.thermal_c),thermal_error=float(hs.thermal_error),homeostatic_error=float(hs.total_error),
            valence_for_next_epoch=float(hs.valence),network_hz=float(report.network_hz),novelty_hz=float(report.novelty_hz),
            keepalive_wave_phase=float(report.keepalive_wave_phase),
            keepalive_wave_active_neurons=int(report.keepalive_wave_active_neurons),
            keepalive_wave_peak_mv=float(report.keepalive_wave_peak_mv),
            motor_mean_hz=float(motor.mean_motor_hz),flight_command=float(motor.flight_command),
            takeoff_command=float(motor.takeoff_command),walking_command=float(motor.walking_command),feeding_command=float(motor.feeding_command),
            modified_edges=int(mem["modified_edges"]),dopamine_mean=float(mod.get("dopamine",0.0)),
            octopamine_mean=float(mod.get("octopamine",0.0)),serotonin_mean=float(mod.get("serotonin",0.0)),
            growth_candidate_count=len(growth_candidates),
            top_growth_candidate_body=None if not growth_candidates else int(growth_candidates[0]["body_id"]),
            eye_peak_side=str(visual.peak_side),eye_peak_yaw_deg=float(visual.peak_yaw_deg),
            eye_peak_elevation_deg=float(visual.peak_elevation_deg),visible_sources=int(visual.visible_star_count),
            antenna_left_c=float(thermal.left_temp_c),antenna_right_c=float(thermal.right_temp_c),
            relative_humidity=float(hygro.get("relative_humidity",0.0)),
            dry_sense=float(hygro.get("dry_drive",0.0)),moist_sense=float(hygro.get("moist_drive",0.0)),
            proboscis_extension=float(max(0.0,motor.effectors.proboscis_extension)),
            pharyngeal_pump=float(motor.effectors.pharyngeal_pump),
            cibarium_fill=float(getattr(motor.effectors,"cibarium_fill",0.0)),
            force_body=[float(x)*float(motor_energy_scale) for x in motor.force_body],
            torque_body=[float(x)*float(motor_energy_scale) for x in motor.torque_body],
        )
        self.logs.append(log)
        self.step_count += 1
        if len(self.logs)>2000: self.logs=self.logs[-2000:]
        self.last_visual=visual; self.last_thermal=thermal; self.last_report=report; self.last_motor=motor
        self.last_food=dict(food_post); self.last_planetary=dict(planetary_pre); self.last_hygro=dict(hygro); self.last_growth_candidates=list(growth_candidates); self.last_consumed=float(consumed)
        self.last_force=motor.force_body.copy(); self.last_torque=motor.torque_body.copy()
        return log

    def census(self) -> dict[str,Any]:
        return {
            "neurons":int(self.core.n),"edges":int(self.core.edge_count),
            "eye":{**summarize_eye_map(self.eye_map),**self.eyelet.census()},"motor":self.motor.census(),
            "food_sensory":self.food.census(),"planetary_sensory":self.planetary.census(),"hygro_thermo_sensory":self.hygro.census(),
            "proprioceptive_populations":self.proprio.census(),
            "homeostatic_populations":{**self.homeo_bridge.census(),"global_host_reinforcement":0},
            "monoamine_neurons":{x:int(np.count_nonzero(self.core.nt==x)) for x in ("dopamine","octopamine","serotonin")},
        }

    @staticmethod
    def _contact_payload(c:ContactState) -> dict[str,Any]:
        return {"on_surface":bool(c.on_surface),"body_id":c.body_id,"surface_temp_c":c.surface_temp_c,
                "normal_world":None if c.normal_world is None else np.asarray(c.normal_world,float).tolist(),
                "altitude":float(c.altitude),"altitude_body_id":c.altitude_body_id,
                "atmosphere_fraction":float(c.atmosphere_fraction),
                "nearest_resource_id":c.nearest_resource_id,
                "nearest_resource_distance":float(c.nearest_resource_distance)}

    def save(self,folder:str|Path) -> dict[str,Any]:
        folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
        self.core.save_memory(folder/"specimen003_memory.npz")
        self.core.save_fast_state(folder/"specimen003_fast_state.npz")
        self.eye_map.save_npz(folder/"compound_eye_map.npz")
        self.growth_pressure.save(folder/"growth_pressure.npz")
        self.homeostasis.save(folder/"homeostasis.json")
        (folder/"living_universe.json").write_text(json.dumps(self.universe.snapshot(),indent=2),encoding="utf-8")
        state={
            "format":self.FORMAT,"specimen":"Specimen-003","graph_name":self.graph.name,
            "mechanics_revision":self.MECHANICS_REVISION,
            "feeding_recovery_revision":self.feeding_recovery_revision,
            "universe_seed":self.universe.seed,"time_s":float(self.time_s),"step_count":int(self.step_count),"pending_valence":float(self.pending_valence),
            "starvation_rescue_s":float(self.starvation_rescue_s),"divine_fruit_drops":int(self.divine_fruit_drops),
            "last_divine_fruit_time_s":float(self.last_divine_fruit_time_s),
            "position":self.body.position.tolist(),"velocity":self.body.velocity.tolist(),
            "orientation":self.body.orientation.tolist(),"angular_velocity":self.body.angular_velocity.tolist(),
            "contact":self._contact_payload(self.body.contact),"thermal_state":self.thermal.snapshot(),
            "motor_mechanical_state":self.motor.snapshot(),
            "proprioceptive_state":self.proprio.snapshot(),
            "retinal_temporal_state":self.eye.snapshot_temporal(),
            "pending_motor_rates":getattr(self,"pending_motor_rates",None),
            "expressed_motor_rates":({f"motor:{k}":float(v) for k,v in self.last_motor.neuron_rates.items()} if self.last_motor is not None else None),
            "eye_mapping_mode":self.eye_map.mapping_mode,"eye_mapping_digest":self.eye_mapping_digest(),
            "eye_spatial_calibration":summarize_eye_map(self.eye_map),
            "files":{"memory":"specimen003_memory.npz","fast":"specimen003_fast_state.npz","eye":"compound_eye_map.npz",
                     "growth":"growth_pressure.npz","homeostasis":"homeostasis.json","universe":"living_universe.json"},
            "census":self.census(),"recent_logs":[asdict(x) for x in self.logs[-240:]],"recent_events":self.events[-400:],
        }
        (folder/"specimen003_state.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
        return state

    @classmethod
    def resume(cls,graph:str|Path,folder:str|Path,*,neural_seed:int=11,auto_tune_threads:bool=True,optics:EyeOpticsConfig|None=None):
        folder=Path(folder); state=json.loads((folder/"specimen003_state.json").read_text(encoding="utf-8"))
        if state.get("format")!=cls.FORMAT: raise ValueError("not a Specimen-003 checkpoint")
        files=state["files"]
        # A checkpoint already carries the resolved eye calibration. Load that tiny
        # map first and validate it against the core as the experiment is constructed,
        # rather than rereading the 25M-edge graph solely to rebuild the same eye.
        resolved_eye=None; saved_eye=folder/files.get("eye","compound_eye_map.npz")
        if saved_eye.exists():
            try:
                resolved_eye=SyntheticHexEyeMap.load_resolved_npz(
                    saved_eye,config=optics,graph_path=graph,calibration=state.get("eye_spatial_calibration",{}))
            except Exception:
                resolved_eye=None
        exp=cls(graph,universe_seed=state["universe_seed"],neural_seed=neural_seed,
                auto_tune_threads=auto_tune_threads,optics=optics,resolved_eye_map=resolved_eye)
        old_digest=str(state.get("eye_mapping_digest", "")); new_digest=exp.eye_mapping_digest()
        old_mode=str(state.get("eye_mapping_mode", "")); new_mode=str(exp.eye_map.mapping_mode)
        eye_migration=None
        if new_digest!=old_digest:
            # A checkpoint is self-contained: first prefer its exact persisted eye map.
            # The official construction workbook may have lived in a global data folder
            # that is absent on another machine or after creating another specimen.
            saved_eye=folder/files.get("eye","compound_eye_map.npz")
            if saved_eye.exists():
                restored=SyntheticHexEyeMap.load_npz(graph,saved_eye,config=optics,calibration=state.get("eye_spatial_calibration",{}))
                exp.eye_map=restored; exp.eye=CompoundEyeRenderer(restored)
                new_digest=exp.eye_mapping_digest(); new_mode=str(restored.mapping_mode)
            if new_digest!=old_digest:
                # Patch11 is a sensory calibration migration, not a neural-state reset.
                # Only the explicitly known v1 synthetic eye is eligible. Unknown eye
                # mismatches still fail closed so learned state cannot silently move
                # between unrelated sensory coordinate systems.
                if old_mode not in getattr(SyntheticHexEyeMap,"OLD_MAPPING_MODES",set()):
                    raise ValueError(f"eye map changed ({old_mode} -> {new_mode}); refusing to transplant learned state")
                eye_migration={
                    "from_mode":old_mode,"from_digest":old_digest,
                    "to_mode":new_mode,"to_digest":new_digest,
                    "neural_state_preserved":True,
                    "note":"Patch11 recalibrated photoreceptor spatial geometry; learned/fast neural state was preserved without reinterpretation.",
                }
        exp.core.load_memory(folder/files["memory"]); exp.core.load_fast_state(folder/files["fast"])

        # FEEDING_PATH1: W925 was forensically shown to carry a persistent transient
        # electrical attractor that suppresses the anatomically present MN11V/MN11D/MN10
        # swallowing circuit.  This is *not* learned-memory repair: plastic_delta,
        # activity familiarity, monoamine fields, last-rate history, RNG chronology and
        # ER5/ExR1 slow-wave phase are all preserved.  For descendants of the audited
        # W925 state, and only when the persisted recent record independently shows the
        # same starvation + no-pump + sustained-flight syndrome, discharge momentary
        # membrane/conductance/refractory state plus already-in-flight delayed spikes.
        # The migration is one-time and is persisted on the next normal checkpoint save.
        feeding_recovery_migration=None
        saved_recovery=str(state.get("feeding_recovery_revision", ""))
        exp.feeding_recovery_revision=saved_recovery
        if saved_recovery != cls.FEEDING_RECOVERY_REVISION:
            import re as _re
            _m=_re.search(r"-W(\d+)\.npz$", str(state.get("graph_name", exp.graph.name)))
            _generation=int(_m.group(1)) if _m else -1
            _recent=list(state.get("recent_logs", []))[-60:]
            _h=[float(x.get("nutritional_hunger", 0.0)) for x in _recent if x.get("nutritional_hunger") is not None]
            _p=[float(x.get("pharyngeal_pump", 0.0)) for x in _recent if x.get("pharyngeal_pump") is not None]
            _c=[float(x.get("nutrient_consumed", 0.0)) for x in _recent if x.get("nutrient_consumed") is not None]
            _f=[float(x.get("flight_command", 0.0)) for x in _recent if x.get("flight_command") is not None]
            _persistent=bool(len(_recent)>=10 and _h and _p and _c and _f and
                             float(np.mean(_h))>=0.95 and max(_p)<=1.0e-9 and
                             sum(_c)<=1.0e-9 and float(np.mean(_f))>=0.50)
            if _generation>=925 and _persistent:
                _delay_spikes=int(sum(len(a) for a in exp.core.delay_ring))
                _mod_before=exp.core.mod_state.copy()
                _rates_before=exp.core.last_rates_hz.copy()
                _rng_before=json.dumps(exp.core.rng.bit_generator.state,sort_keys=True)
                _wave_time=float(exp.core._keepalive_time_ms)
                exp.core.v.fill(exp.core.config.v_rest)
                exp.core.g.fill(0.0)
                exp.core.refr.fill(0)
                exp.core.delay_ring=[np.empty(0,dtype=np.int32) for _ in exp.core.delay_ring]
                feeding_recovery_migration={
                    "type":"feeding_transient_electrical_recovery",
                    "revision":cls.FEEDING_RECOVERY_REVISION,
                    "generation":int(_generation),
                    "cleared":["membrane_voltage","synaptic_conductance","refractory_counters","pending_delayed_spikes"],
                    "pending_delayed_spikes_cleared":_delay_spikes,
                    "preserved":{
                        "plastic_memory":True,
                        "neuromodulator_state":bool(np.array_equal(_mod_before,exp.core.mod_state)),
                        "last_rate_history":bool(np.array_equal(_rates_before,exp.core.last_rates_hz)),
                        "rng_chronology":bool(_rng_before==json.dumps(exp.core.rng.bit_generator.state,sort_keys=True)),
                        "state_wave_phase":bool(abs(_wave_time-float(exp.core._keepalive_time_ms))<=1.0e-12),
                    },
                    "eligibility":{
                        "recent_frames":len(_recent),
                        "mean_nutritional_hunger":float(np.mean(_h)),
                        "max_pharyngeal_pump":max(_p),
                        "total_recent_consumption":sum(_c),
                        "mean_flight_command":float(np.mean(_f)),
                    },
                    "reason":"forensically reproduced W925 transient attractor suppressed feeding outputs despite intact graph and learned memory",
                }
            # Whether intervention was necessary or not, a newly saved checkpoint has
            # now been evaluated under this revision and must not be reconsidered.
            exp.feeding_recovery_revision=cls.FEEDING_RECOVERY_REVISION

        exp.motor.restore(state.get("motor_mechanical_state",{}))
        gp=folder/files["growth"]
        if gp.exists(): exp.growth_pressure=GrowthPressureTracker.load(gp)
        hp=folder/files["homeostasis"]
        if hp.exists(): exp.homeostasis=HomeostaticSystem.load(hp); exp.homeo_bridge=InteroceptiveNutrientBridge(exp.core)
        up=folder/files["universe"]
        if up.exists(): exp.universe.restore(json.loads(up.read_text(encoding="utf-8")))
        exp.time_s=float(state["time_s"]); exp.step_count=int(state.get("step_count", len(state.get("recent_logs",[])))); exp.pending_valence=0.0
        exp.starvation_rescue_s=float(state.get("starvation_rescue_s",0.0))
        exp.divine_fruit_drops=int(state.get("divine_fruit_drops",0))
        exp.last_divine_fruit_time_s=float(state.get("last_divine_fruit_time_s",-1.0e30))
        exp.body.position=np.asarray(state["position"],float); exp.body.velocity=np.asarray(state["velocity"],float)
        exp.body.orientation=q_normalize(np.asarray(state["orientation"],float)); exp.body.angular_velocity=np.asarray(state["angular_velocity"],float)
        mechanics_migration=None
        if str(state.get("mechanics_revision", "")) != cls.MECHANICS_REVISION:
            # One-time mechanical-state migration only.  Earlier embodiments could
            # manufacture pitch/yaw angular velocity by pooling unrelated steering MNs,
            # coupling neck motion directly to thorax yaw/pitch, and applying leg posture
            # torque in free flight.  Those angular components are host-created body
            # artifacts, not neural memory.  Preserve roll (which can arise from real
            # left/right power asymmetry) while clearing the contaminated pitch/yaw
            # momentum.  Neural fast state, plastic memory, RNG and orientation remain
            # untouched.
            old_w=exp.body.angular_velocity.copy()
            exp.body.angular_velocity[1]=0.0
            exp.body.angular_velocity[2]=0.0
            mechanics_migration={
                "type":"mechanical_state_migration",
                "from_revision":state.get("mechanics_revision"),
                "to_revision":cls.MECHANICS_REVISION,
                "old_angular_velocity":[float(x) for x in old_w],
                "new_angular_velocity":[float(x) for x in exp.body.angular_velocity],
                "neural_state_changed":False,
                "reason":"clear legacy host-generated free-flight pitch/yaw momentum after steering decoder correction",
            }
        # Reconstruct the spatial contact/altitude frame from the restored physical
        # world rather than trusting a legacy unreferenced altitude scalar.  The exact
        # radial clearance is always retained; altitude_body_id tells observers which
        # surface owns that number even in free flight.
        c=state.get("contact",{})
        nb,nalt,nnorm=exp.universe.nearest_body(exp.body.position)
        _,nr,nrd=exp.universe.nearest_resource(exp.body.position)
        old_on_surface=bool(c.get("on_surface",False))
        saved_contact_id=c.get("body_id")
        # A saved surface attachment is accepted only when the restored position is
        # actually on that nearest body's surface.  This prevents a stale folder/file
        # transplant from manufacturing contact with an unrelated world.
        contact_ok=bool(old_on_surface and nb is not None and
                        (saved_contact_id is None or saved_contact_id==nb.body_id) and
                        float(nalt) <= max(float(getattr(exp.body,"CONTACT_EPSILON",0.0)),1e-6))
        exp.body.contact=ContactState(
            on_surface=contact_ok,
            body_id=nb.body_id if contact_ok else None,
            surface_temp_c=nb.surface_temp_c if contact_ok else None,
            normal_world=np.asarray(nnorm,float) if contact_ok and nnorm is not None else None,
            altitude=float(nalt),
            altitude_body_id=None if nb is None else nb.body_id,
            atmosphere_fraction=float(exp.universe.atmosphere_fraction(exp.body.position,nb)),
            nearest_resource_id=None if nr is None else nr.patch_id,
            nearest_resource_distance=float(nrd),
        )
        exp.body._set_flight_diagnostics(exp.universe)
        exp.thermal.restore(state.get("thermal_state",{})); exp.eye.restore_temporal(state.get('retinal_temporal_state'))
        # Proprioception on the next epoch depends on the previously expressed motor
        # frame. Reconstruct it deterministically from the persisted last-rates vector
        # and restored contact state so save/resume does not create a one-step sensory
        # discontinuity.
        _rates={}
        for _name,_ids in exp.motor.readouts().items():
            _ii=np.asarray(_ids,dtype=np.int32)
            _rates[_name]=float(np.mean(exp.core.last_rates_hz[_ii])) if len(_ii) else 0.0
        if state.get('expressed_motor_rates') is not None:
            _rates={str(k):float(v) for k,v in state['expressed_motor_rates'].items()}
        exp.pending_motor_rates=state.get('pending_motor_rates')
        exp.last_motor=exp.motor.actuate(_rates,on_surface=exp.body.contact.on_surface,
                                          dt_s=0.0,update_mechanics=False)
        exp.proprio.restore(state.get('proprioceptive_state'),body=exp.body,motor_frame=exp.last_motor)
        exp.events=list(state.get("recent_events",[])); exp.logs=[]
        if mechanics_migration is not None:
            mechanics_migration["time_s"]=float(exp.time_s)
            exp.events.append(mechanics_migration)
        if feeding_recovery_migration is not None:
            feeding_recovery_migration["time_s"]=float(exp.time_s)
            exp.events.append(feeding_recovery_migration)
        # Migrate whatever pre-atlas landing history survives in the checkpoint event
        # ring, then register the currently restored local world without double-counting
        # an already-recorded persisted landing.
        exp.universe.backfill_atlas_from_events(exp.events)
        _abid=exp.body.contact.body_id if exp.body.contact.on_surface else exp.body.contact.altitude_body_id
        if _abid is not None and (exp.body.contact.on_surface or float(exp.body.contact.atmosphere_fraction)>1.0e-9):
            _ab=exp.universe.body_by_id(str(_abid))
            exp.universe.record_world_visit(_ab,time_s=exp.time_s,altitude=float(exp.body.contact.altitude),
                                            on_surface=bool(exp.body.contact.on_surface),position=exp.body.position)
        if eye_migration is not None:
            exp._event("eye_spatial_calibration_migration",**eye_migration)
        return exp


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--graph",required=True); ap.add_argument("--steps",type=int,default=20)
    ap.add_argument("--world-dt",type=float,default=.10); ap.add_argument("--neural-ms",type=float,default=100.0)
    ap.add_argument("--save-dir"); a=ap.parse_args()
    e=Specimen003Experiment(a.graph)
    print(json.dumps(e.census(),indent=2))
    for _ in range(a.steps):
        x=e.step(world_dt=a.world_dt,neural_ms=a.neural_ms)
        print(f"t={x.time_s:7.2f}s net={x.network_hz:5.2f}Hz motor={x.motor_mean_hz:5.1f} hunger={x.hunger:.3f} "
              f"err={x.homeostatic_error:.3f} val={x.valence_for_next_epoch:+.3f} surface={x.on_surface} "
              f"food={x.nearest_resource_distance:5.2f} eat={x.nutrient_consumed:.4f}")
    if a.save_dir: e.save(a.save_dir)
