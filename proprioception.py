#!/usr/bin/env python3
"""Mechanical/proprioceptive sensory feedback for Specimen-003.

This closes part of the loop that a normal fly gets for free: motor output changes the
body, and retained MaleCNS mechanosensory populations receive a bounded description of
what the body actually did.  No desired heading, reward, avoidance command, or target
coordinate is encoded here.

The transfer functions are engineering approximations.  The population identities are
real MaleCNS annotations; their exact physiological tuning is not claimed.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
from space_fly_universe import q_inverse_rotate


def _clip_hz(x: float, cap: float = 85.0) -> float:
    return float(np.clip(float(x), 0.0, cap))


class ProprioceptiveBridge:
    """Conservative mechanosensory embodiment using physical body state only.

    Important Patch20 boundaries:
    * JO-A/B are vibration/sound channels.  A 0.1-s world step cannot represent the
      ~20-1000 Hz antennal vibrations they encode, so they are left silent until a
      proper high-rate acoustic/antenna solver exists.
    * JO-C/E are sustained antennal-deflection channels for wind/gravity and receive
      only relative-airflow + gravity-derived deflection.
    * Leg taste-bristle GRNs are NOT mechanosensors and are never driven here.
    """

    SENSORY_SUPERCLASSES = {
        "vnc_sensory", "cb_sensory", "ol_sensory", "sensory_ascending",
        "sensory_descending", "vnc_sensory_tbc", "cb_sensory_tbc",
        "sensory_ascending_tbc",
    }

    def __init__(self, core):
        self.core = core
        sc = core.superclass.astype(str)
        sub = core.subclass.astype(str)
        typ = core.types.astype(str)
        sensory = np.isin(sc, list(self.SENSORY_SUPERCLASSES))
        self.groups: dict[str, np.ndarray] = {}
        for name in (
            "haltere", "campaniform sensilla", "chordotonal organ", "leg bristle",
            "leg", "hair plate", "wing bristle", "wing", "neck", "abdomen",
            "wind_gravity", "auditory", "mechanosensory bristle",
        ):
            self.groups[name] = np.flatnonzero(sensory & (sub == name)).astype(np.int32)

        inst = np.char.lower(core.instance.astype(str))
        side = np.char.lower(core.side.astype(str))
        self._is_left = (side == "l") | (side == "left") | np.char.endswith(inst, "_l")
        self._is_right = (side == "r") | (side == "right") | np.char.endswith(inst, "_r")

        # Functionally established Johnston-organ split.
        tu=np.char.upper(typ)
        aud=self.groups["auditory"]
        wg=self.groups["wind_gravity"]
        # Coarse subclass labels overlap functional JO types in this release.
        # Classify the union by explicit subgroup identity, not subclass alone.
        alljo=np.unique(np.concatenate([aud,wg])).astype(np.int32)
        self.jo_ab=alljo[np.char.startswith(tu[alljo],"JO-A") | np.char.startswith(tu[alljo],"JO-B")]
        # JO-C and JO-E are opponent static-deflection channels, not one pooled
        # "wind/gravity magnitude" population.  C is preferentially activated by
        # anterior/pull deflection of the receiver; E by posterior/push deflection.
        # Keep the populations separate so a single physical deflection cannot excite
        # both opponent channels at once.
        self.jo_c=alljo[np.char.startswith(tu[alljo],"JO-C")]
        self.jo_e=alljo[np.char.startswith(tu[alljo],"JO-E")]
        self.jo_ce=np.unique(np.concatenate([self.jo_c,self.jo_e])).astype(np.int32) if (len(self.jo_c) or len(self.jo_e)) else np.empty(0,dtype=np.int32)
        known=np.concatenate([self.jo_ab,self.jo_ce]) if (len(self.jo_ab) or len(self.jo_ce)) else np.empty(0,dtype=np.int32)
        alljo=np.unique(np.concatenate([aud,wg])).astype(np.int32) if (len(aud) or len(wg)) else np.empty(0,dtype=np.int32)
        self.jo_uncertain=np.setdiff1d(alljo,known,assume_unique=False).astype(np.int32)

        # Adult MaleCNS aPhM cells are cholinergic pharyngeal mechanosensory neurons.
        # Their physical stimulus is represented by cibarial expansion/emptying from
        # the motor embodiment, never by food identity or a host feeding decision.
        self.aphm=np.flatnonzero(sensory & np.char.startswith(typ.astype(str),"aPhM")).astype(np.int32)

        # Peripheral nerve annotations locate the leg pair; missing laterality
        # stays unpaired rather than assigning a side by neuron ID.
        nerve=np.asarray(getattr(core,'entry_nerve',np.full(len(sc),'')),dtype=str)
        self.leg_segments={}
        self.strain_segments={name:self._split(self.groups['campaniform sensilla'][
            np.isin(nerve[self.groups['campaniform sensilla']],nerves)])
            for name,nerves in {'wing':['ADMN'],'haltere':['DMetaN'],
                               'leg':['ProLN','MesoLN','MetaLN']}.items()}
        for name in ('leg','hair plate','chordotonal organ'):
            ids=self.groups[name]
            parts=[]
            known=np.zeros(len(ids),dtype=bool)
            for pair,nerve_name in enumerate(('ProLN','MesoLN','MetaLN')):
                mask=nerve[ids]==nerve_name;known|=mask
                parts.append((pair,self._split(ids[mask])))
            self.leg_segments[name]=(parts,self._split(ids[~known]))
        self._prev_leg=np.zeros(6,dtype=float)
        self._prev_surface=False

    def _split(self, ids: np.ndarray):
        if not len(ids):
            e=np.empty(0,dtype=np.int32); return e,e,e
        l=ids[self._is_left[ids]]; r=ids[self._is_right[ids]]
        known=self._is_left[ids] | self._is_right[ids]
        return l.astype(np.int32,copy=False),r.astype(np.int32,copy=False),ids[~known].astype(np.int32,copy=False)

    @staticmethod
    def _add_bilateral(out, split, base_hz: float, lateral: float = 0.0):
        l,r,u=split; lat=float(np.clip(lateral,-1.0,1.0))
        if len(l): out.append((l,_clip_hz(base_hz*(1.0-0.35*lat))))
        if len(r): out.append((r,_clip_hz(base_hz*(1.0+0.35*lat))))
        if len(u): out.append((u,_clip_hz(base_hz)))

    @staticmethod
    def _leg_state(eff):
        if eff is None: return np.zeros(6,dtype=float)
        return np.asarray([eff.foreleg_left,eff.foreleg_right,eff.midleg_left,eff.midleg_right,
                           eff.hindleg_left,eff.hindleg_right],dtype=float)

    def drives(self, body, motor_frame=None, *, ambient_flow_body=None, dt_s=0.1) -> list[tuple[np.ndarray,float]]:
        out=[]
        omega=np.asarray(body.angular_velocity,dtype=float); omag=float(np.linalg.norm(omega))
        on_surface=bool(getattr(body.contact,"on_surface",False))
        vel_world=np.asarray(getattr(body,"velocity",np.zeros(3)),dtype=float)
        try: vel_body=q_inverse_rotate(body.orientation,vel_world.reshape(1,3))[0]
        except Exception: vel_body=vel_world
        airspeed=float(np.linalg.norm(vel_body))
        try: grav_body=q_inverse_rotate(body.orientation,np.asarray(body.gravity_world,dtype=float).reshape(1,3))[0]
        except Exception: grav_body=np.zeros(3,dtype=float)
        gmag=float(np.linalg.norm(grav_body)); gunit=grav_body/gmag if gmag>1e-12 else np.zeros(3)

        if motor_frame is None:
            flight=0.0; force=np.zeros(3); eff=None
        else:
            flight=float(np.clip(motor_frame.flight_command,0,1))
            force=np.asarray(motor_frame.force_body,dtype=float)
            eff=motor_frame.effectors

        # Haltere envelope: actual body angular motion under active flight.  This is an
        # envelope only; the sub-millisecond wingbeat phase is below world-step resolution.
        haltere_base=flight*(6.0+42.0*math.tanh(omag/0.35))
        self._add_bilateral(out,self._split(self.groups["haltere"]),haltere_base,
                            math.tanh((omega[0]+0.65*omega[2])/0.20) if omag else 0.0)

        # JO-A/B acoustic/vibration neurons deliberately receive no surrogate wind or
        # body-speed signal.  Proper stimulation requires a high-rate antenna vibration
        # waveform (tens-hundreds of Hz and above), not a 10-Hz world envelope.

        # JO-C/E: sustained opponent static deflection from relative airflow + gravity.
        # The previous bridge used |deflection| and drove C and E together.  That erased
        # the sign of antennal displacement and, in this connectome, let ordinary
        # gravity stimulate the JO-E branch that can seed a persistent flight attractor.
        #
        # Body +X is the fly's forward/anterior axis in this embodiment.  A positive
        # receiver displacement therefore drives the anterior/pull-preferring JO-C
        # population; a negative displacement drives posterior/push-preferring JO-E.
        # Half-wave rectification preserves the biological opponent code: never both
        # branches merely because one antenna is deflected.
        airflow=-vel_body
        wind=float(math.tanh(np.linalg.norm(airflow)/4.0))
        if np.linalg.norm(airflow)>1e-12: air_dir=airflow/np.linalg.norm(airflow)
        else: air_dir=np.zeros(3)

        # External laminar carrier at the antennae.  In the ordinary atmosphere this
        # can represent ambient wind; No Man's Fly also supplies the local direction
        # of its fictional vacuum-borne volatile carrier.  The vector is already in
        # body coordinates and contains no destination/behavior semantics.
        amb=np.zeros(3,dtype=float)
        if ambient_flow_body is not None:
            try:
                a=np.asarray(ambient_flow_body,dtype=float).reshape(3)
                am=float(np.linalg.norm(a))
                if am>1e-12:
                    amb=(a/am)*min(1.0,am)
            except Exception:
                amb=np.zeros(3,dtype=float)

        # Convert physical flow/gravity into the displacement of each passive antennal
        # receiver.  Wind from the front pushes both antennae; wind from one side pulls
        # the ipsilateral antenna and pushes the contralateral antenna.  Positive
        # receiver displacement is "pull" (JO-C); negative is "push" (JO-E).
        # This preserves bilateral opponent coding instead of turning bearing into a
        # host yaw signal.
        deflect=0.72*wind*air_dir + 0.42*amb + 0.28*gunit
        left_pull=float(np.clip(deflect[0]+deflect[1],-1.0,1.0))
        right_pull=float(np.clip(deflect[0]-deflect[1],-1.0,1.0))
        c_l,c_r,c_u=self._split(self.jo_c); e_l,e_r,e_u=self._split(self.jo_e)
        if len(c_l) and left_pull>0.0: out.append((c_l,_clip_hz(48.0*left_pull)))
        if len(e_l) and left_pull<0.0: out.append((e_l,_clip_hz(48.0*(-left_pull))))
        if len(c_r) and right_pull>0.0: out.append((c_r,_clip_hz(48.0*right_pull)))
        if len(e_r) and right_pull<0.0: out.append((e_r,_clip_hz(48.0*(-right_pull))))
        # Any genuinely unpaired C/E cells retain only the fore/aft displacement; do
        # not invent a side for them.
        mid_pull=float(np.clip(deflect[0],-1.0,1.0))
        if len(c_u) and mid_pull>0.0: out.append((c_u,_clip_hz(48.0*mid_pull)))
        if len(e_u) and mid_pull<0.0: out.append((e_u,_clip_hz(48.0*(-mid_pull))))
        # JO-D/F/unclear remain silent until their tuning is defensible.

        # Tactile bristles: weak airflow while airborne; substrate sweep and contact
        # onset while grounded.  No blanket 'surface=True' stimulation.
        contact_transient=1.0 if (on_surface and not self._prev_surface) else 0.0
        tactile=float(np.clip((0.18 if not on_surface else 0.55)*math.tanh(airspeed/3.0)+0.55*contact_transient,0,1))
        self._add_bilateral(out,self._split(self.groups["mechanosensory bristle"]),26.0*tactile,
                            math.tanh(float(vel_body[1])/1.5) if airspeed else 0.0)

        # Only aerodynamic force belongs in the wing envelope. Old frames with no
        # decomposition use wing power alone, never the mixed leg/body force.
        wing_force=getattr(motor_frame,'flight_force_body',None)
        wing_load=float(np.clip(np.linalg.norm(wing_force)/0.08,0,1)) if wing_force is not None else flight
        self._add_bilateral(out,self._split(self.groups["wing bristle"]),24.0*flight+18.0*wing_load,
                            math.tanh(omega[0]/0.18))
        self._add_bilateral(out,self._split(self.groups["wing"]),18.0*flight+14.0*wing_load,
                            math.tanh(omega[2]/0.18))

        # Strain/load receptors.
        support=1.0 if on_surface else 0.0
        # Organ-local envelopes remain reduced transduction models. Ground support
        # cannot load haltere/wing sensilla; wing force cannot load leg sensilla.
        self._add_bilateral(out,self.strain_segments['wing'],38.0*0.65*wing_load)
        self._add_bilateral(out,self.strain_segments['leg'],38.0*0.50*support)
        self._add_bilateral(out,self.strain_segments['haltere'],haltere_base,
                            math.tanh((omega[0]+0.65*omega[2])/0.20) if omag else 0.0)

        # Limb proprioception uses the expressed leg effectors.  Crucially, the graph's
        # 'leg bristle' subclass is gustatory (LgLG/LgAG types) and is not touched here.
        leg=self._leg_state(eff); leg_abs=float(np.mean(np.abs(leg)))
        # Movement is a rate, not a difference per arbitrary rendering slice.
        # Retain the old 100-ms calibration while making it independent of slicing.
        leg_speed=np.abs(leg-self._prev_leg)*(0.1/max(float(dt_s),1e-6))
        def segment_feedback(name,values,gain):
            parts,unknown=self.leg_segments[name]
            for pair,(left,right,unpaired) in parts:
                lv=float(values[2*pair]);rv=float(values[2*pair+1])
                for ids,value in ((left,lv),(right,rv),(unpaired,.5*(lv+rv))):
                    if len(ids):out.append((ids,_clip_hz(gain*value)))
            # An unlocalized receptor must not receive all-leg motion by default.
            # These cells remain in the graph/census pending an anatomical mapping.
        segment_feedback('leg',np.abs(leg),26.0*(1.0 if on_surface else 0.35))
        segment_feedback('hair plate',np.abs(leg),22.0)
        segment_feedback('chordotonal organ',np.clip(leg_speed/0.20,0,1),34.0)

        head=0.0 if eff is None else min(1.0,abs(float(eff.head_yaw))+abs(float(eff.head_pitch)))
        abd=0.0 if eff is None else min(1.0,abs(float(eff.abdomen_bend)))
        self._add_bilateral(out,self._split(self.groups["neck"]),26.0*head)
        self._add_bilateral(out,self._split(self.groups["abdomen"]),20.0*abd)

        # Cibarial expansion is a genuine internal mechanical stimulus.  Keep the
        # feedback modest and continuous so the connectome decides what it means.
        # We deliberately do not cherry-pick only cells that happen to connect to MN11;
        # all retained aPhM mechanosensory cells receive the same physical state.
        if eff is not None and len(self.aphm):
            fill=float(np.clip(getattr(eff,"cibarium_fill",0.0),0.0,1.0))
            emptying=float(np.clip(getattr(eff,"pharyngeal_pump",0.0),0.0,1.0))
            aphm_hz=_clip_hz(18.0*fill + 6.0*emptying,cap=28.0)
            if aphm_hz>0.0:
                out.append((self.aphm,aphm_hz))

        self._prev_leg=leg.copy(); self._prev_surface=on_surface
        return [(ids,hz) for ids,hz in out if len(ids) and hz>0.0]

    def snapshot(self):
        return {'previous_leg':self._prev_leg.tolist(),'previous_surface':bool(self._prev_surface)}

    def restore(self,payload=None,*,body=None,motor_frame=None):
        if payload is None:
            # Legacy checkpoints have no receptor history: initialize from the
            # restored physical state without manufacturing a contact/motion event.
            self._prev_leg=self._leg_state(getattr(motor_frame,'effectors',None)).copy()
            self._prev_surface=bool(getattr(getattr(body,'contact',None),'on_surface',False))
            return
        leg=np.asarray(payload['previous_leg'],dtype=float)
        if leg.shape!=(6,) or not np.all(np.isfinite(leg)):
            raise ValueError('Invalid proprioceptive checkpoint history')
        self._prev_leg=leg.copy();self._prev_surface=bool(payload['previous_surface'])

    def census(self) -> dict[str,Any]:
        out={name:int(len(ids)) for name,ids in self.groups.items()}
        out.update({"JO_AB_vibration_silent_pending_acoustic_solver":int(len(self.jo_ab)),
                    "JO_C_anterior_pull":int(len(self.jo_c)),
                    "JO_E_posterior_push":int(len(self.jo_e)),
                    "JO_CE_wind_gravity":int(len(self.jo_ce)),
                    "JO_uncertain_silent":int(len(self.jo_uncertain)),
                    "aPhM_cibarium_mechanosensory":int(len(self.aphm)),
                    "leg_taste_bristles_mechanical_drive":0,
                    "unlocalized_limb_receptors_not_driven":{name:sum(len(x) for x in unknown)
                        for name,(_,unknown) in self.leg_segments.items()}})
        return out

