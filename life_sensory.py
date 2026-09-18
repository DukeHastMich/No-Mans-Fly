#!/usr/bin/env python3
"""Native chemosensory/hygrosensory bridge for Specimen-003.

The environment supplies physical-ish fields; retained MaleCNS sensory populations do
all downstream interpretation.  Nothing here supplies a desired heading, reward,
"novelty" label, or target vector.

Patch 20 retains and audits the old three-ORN generic food scalar with a deliberately conservative
DoOR-informed fermentation mixture across multiple real ORN glomerular classes, and
uses the 2026 MaleCNS gustatory connectome assignments for appetitive leg/labellar
contact.  It is not a complete DoOR 693-odorant chemical simulator: the shipped world
has a small named mixture whose receptor response directions are literature grounded.
"""
from __future__ import annotations
import hashlib
import math
from typing import Any

import numpy as np
from space_fly_universe import q_inverse_rotate


class FoodSensoryBridge:
    ODOR_LENGTH_SCALE = 18.0

    # Curated DoOR-informed glomerular response weights for compounds represented in
    # the procedural fermentation field.  Values are normalized engineering gains,
    # not claimed firing-rate fits.  Glomerulus assignments / strong ligands follow
    # DoOR 2.0 (Muench & Galizia 2016): DM1/Or42b, VA2/Or92a, DM4/Or59b,
    # DM5/Or33b/Or85a, DM2/Or22a, VM5d/Or85b, DM3/Or47a, DL1/Or10a, DA2/Or56a.
    ORN_MIXTURE = {
        "ORN_DM1":  {"ethyl_propionate": 1.00, "three_hexanone": 0.92, "hydroxybutyrate": 0.78},
        "ORN_VA2":  {"diacetyl": 1.00, "ethyl_propionate": 0.22},
        "ORN_DM4":  {"methyl_acetate": 0.92, "ethyl_propionate": 0.20},
        "ORN_DM5":  {"hydroxybutyrate": 0.95, "ethyl_propionate": 0.35},
        "ORN_DM2":  {"ethyl_hexanoate": 1.00, "pentyl_acetate": 0.30},
        "ORN_VM5d": {"butyl_acetate": 1.00, "pentyl_acetate": 0.35},
        "ORN_DM3":  {"pentyl_acetate": 1.00, "ethyl_propionate": 0.25},
        "ORN_DL1":  {"butyric_acid": 0.95, "ethyl_benzoate": 0.45},
        "ORN_DA2":  {"geosmin": 1.00},
        "ORN_V":    {"carbon_dioxide": 1.00},
    }

    @classmethod
    def odor_concentration(cls, resource, distance: float, remaining: float) -> float:
        d=max(0.0,float(distance)); amount=max(float(resource.amount),1e-9)
        # Airborne volatile concentration depends on volatile production and dilution,
        # not on the host's semantic notion of nutritional value.  Individual compounds
        # below can still covary with substrate chemistry where biologically plausible.
        return float(np.clip(float(resource.fermentation)*
                             math.exp(-d/cls.ODOR_LENGTH_SCALE)*(max(0.0,float(remaining))/amount),0,1))

    @staticmethod
    def _resource_chemistry(resource) -> dict[str,float]:
        """Small deterministic chemistry vector derived without extra world RNG draws."""
        n=float(np.clip(resource.nutrient_density,0,1)); f=float(np.clip(resource.fermentation,0,1))
        h=hashlib.sha256(str(resource.patch_id).encode("utf-8")).digest()
        u=[x/255.0 for x in h[:8]]
        # Fermentation produces a mixture, not a semantic "food odor".  Individual
        # resource IDs vary the ester/acid balance while nutrient density supplies the
        # sweet substrate.  Geosmin is intentionally rare/weak rather than a food cue.
        return {
            "ethyl_propionate": float(np.clip(f*(0.35+0.65*u[0]),0,1)),
            "three_hexanone": float(np.clip(f*(0.18+0.55*u[1]),0,1)),
            "hydroxybutyrate": float(np.clip(f*n*(0.25+0.65*u[2]),0,1)),
            "diacetyl": float(np.clip(f*(0.20+0.70*u[3]),0,1)),
            "methyl_acetate": float(np.clip(f*(0.15+0.55*u[4]),0,1)),
            "ethyl_hexanoate": float(np.clip(f*n*(0.12+0.60*u[5]),0,1)),
            "butyl_acetate": float(np.clip(f*(0.10+0.55*u[6]),0,1)),
            "pentyl_acetate": float(np.clip(f*(0.08+0.50*u[7]),0,1)),
            "butyric_acid": float(np.clip(f*(0.16+0.55*(1.0-u[0])),0,1)),
            "ethyl_benzoate": float(np.clip(f*0.18*u[2],0,1)),
            "geosmin": float(0.10 if u[7] > 0.94 else 0.0),
            "carbon_dioxide": float(np.clip(f*(0.30+0.70*n),0,1)),
            "carbonation": float(np.clip(f*(0.25+0.65*n),0,1)),
            # Contact chemistry used by gustation.
            "sugar": n,
            "water": float(np.clip(0.30+0.55*n*(1.0-0.35*f),0,1)),
            "low_salt": float(np.clip(0.08+0.12*u[5],0,0.25)),
            "high_salt": 0.0,
            "bitter": 0.0,
        }

    def __init__(self, core):
        self.core=core
        self.orn={name:core.ids(neuron_type=name) for name in self.ORN_MIXTURE}
        # 2026 MaleCNS gustatory-connectome assignments.
        self.lglg3=core.ids(neuron_type="LgLG3")  # likely Gr5a/sugar
        self.lglg4=core.ids(neuron_type="LgLG4")  # Gr64f/Ir56b sugar + attractive low salt
        self.lb3a=core.ids(neuron_type="LB3a")    # ppk28 water
        self.lb3b=core.ids(neuron_type="LB3b")    # Gr64f sugar + Ir56b low salt overlap
        self.lb3c=core.ids(neuron_type="LB3c")    # Gr64f sugar
        self.lb3d=core.ids(neuron_type="LB3d")    # aversive high salt / Ir7c/ppk23-like
        self.lb1=np.unique(np.concatenate([core.ids(neuron_type=x) for x in ("LB1a","LB1b","LB1c","LB1d")])).astype(np.int32)
        self.phg1=np.unique(np.concatenate([core.ids(neuron_type=x) for x in ("PhG1a","PhG1b","PhG1c")])).astype(np.int32)
        self.claw=core.ids(neuron_type="claw_tpGRN")
        self.dorsal=core.ids(neuron_type="dorsal_tpGRN")
        inst=np.char.lower(core.instance.astype(str)); side=np.char.lower(core.side.astype(str))
        self._left=(side=='l')|(side=='left')|np.char.endswith(inst,'_l')
        self._right=(side=='r')|(side=='right')|np.char.endswith(inst,'_r')

    def _split(self,ids):
        ids=np.asarray(ids,dtype=np.int32)
        l=ids[self._left[ids]]; r=ids[self._right[ids]]
        u=ids[~(self._left[ids]|self._right[ids])]
        return l,r,u

    @staticmethod
    def _add_bilateral(drives,split,base_hz,lateral):
        l,r,u=split; lat=float(np.clip(lateral,-1,1))
        if len(l): drives.append((l,float(base_hz*(1.0-0.32*lat))))
        if len(r): drives.append((r,float(base_hz*(1.0+0.32*lat))))
        if len(u): drives.append((u,float(base_hz)))

    @staticmethod
    def _motor_contact_factors(motor_frame) -> tuple[float,float]:
        if motor_frame is None: return 0.0,0.0
        eff=getattr(motor_frame,"effectors",None)
        if eff is None: return 0.0,0.0
        extension=float(np.clip(max(0.0,float(eff.proboscis_extension))/0.30,0,1))
        pump=float(np.clip(float(eff.pharyngeal_pump),0,1))
        return extension,pump

    def observe(self, world, body, motor_frame=None):
        habitat,resource,d=world.nearest_resource(body.position)
        if resource is None:
            return [], {"resource_id":None,"distance":float("inf"),"odor":0.0,"contact":False,
                        "foot_contact":False,"mouth_contact":False,"mouth_distance":float("inf"),
                        "fermentation":0.0,"bearing_yaw_deg":None,"chemistry":{},
                        "volatile_flow_body":[0.0,0.0,0.0],
                        "proboscis_contact":0.0,"proboscis_extension":0.0,"pump_contact":0.0}
        remaining=world.resource_remaining(resource.patch_id,resource.amount)
        # No Man's Fly deliberately allows chemosensory fields to propagate through
        # vacuum.  The field still comes from the resource's deterministic chemistry,
        # attenuates with distance, and reaches only real ORN populations; it does not
        # expose a destination coordinate or inject steering/reward.  Atmosphere is
        # therefore not a gate on odor in this universe.
        atm=float(world.atmosphere_fraction(body.position,habitat)) if hasattr(world,"atmosphere_fraction") else 1.0
        odor=self.odor_concentration(resource,d,remaining)
        chem=self._resource_chemistry(resource)
        q=habitat.resource_position(resource); rel=np.asarray(q,float)-np.asarray(body.position,float)
        nr=max(float(np.linalg.norm(rel)),1e-12)
        dbody=q_inverse_rotate(body.orientation,(rel/nr).reshape(1,3))[0]
        lateral=float(np.clip(dbody[1],-1,1)); bearing=float(math.degrees(math.atan2(float(dbody[1]),float(dbody[0]))))
        # In No Man's Fly, volatile chemistry can propagate through vacuum.  Give that
        # field an equally physical local carrier direction: material radiates away
        # from its source, so at the fly the carrier velocity points from the source
        # toward the receiver (-dbody).  Magnitude is the local volatile concentration.
        # This is a local environmental vector, not a desired heading or steering
        # command; the only downstream consumer is the antennal mechanosensory bridge.
        volatile_flow_body=(-np.asarray(dbody,dtype=np.float64)*float(np.clip(odor,0.0,1.0)))
        drives=[]

        # Combinatorial olfactory pattern: concentration scales the per-glomerulus
        # response to the resource's actual deterministic mixture.
        if odor>0:
            for name,weights in self.ORN_MIXTURE.items():
                response=sum(float(chem.get(c,0.0))*w for c,w in weights.items())
                response=float(np.clip(response,0,1))
                if response>0:
                    self._add_bilateral(drives,self._split(self.orn[name]),72.0*odor*response,lateral)

        # Tarsal/leg contact and labellar/proboscis contact are physically distinct.
        # The previous bridge incorrectly required the body/feet to already be inside
        # the food patch before an extended proboscis could taste it.  That produced a
        # catch-22: Yar could extend toward nearby food yet his face taste neurons still
        # sampled empty air.  Keep foot contact for leg GRNs, but give the mouth its own
        # forward reach in body coordinates.
        foot_contact=bool(body.contact.on_surface and d <= resource.radius)
        extension,pump=self._motor_contact_factors(motor_frame)

        # Body +X is forward.  Convert the physical body length to local world units,
        # put the labellar base near the anterior end of the body, and let extension
        # sweep a short line segment forward.  Contact is determined geometrically by
        # the distance from the resource patch center to that segment; no food bearing
        # or desired heading is passed downstream.
        body_length_u=float(getattr(body,"BODY_LENGTH_M",2.26e-3))/max(float(getattr(body,"LOCAL_M_PER_UNIT",1.0e-2)),1.0e-12)
        head_base_x=0.45*body_length_u
        max_proboscis_reach=0.50*body_length_u
        labellum_radius=max(0.035*body_length_u,0.004)
        rel_body=q_inverse_rotate(body.orientation,rel.reshape(1,3))[0]
        mouth_distance=float("inf")
        labellar_access=0.0
        mouth_contact=False
        if extension>0.0:
            a=np.asarray([head_base_x,0.0,0.0],dtype=np.float64)
            btip=np.asarray([head_base_x+max_proboscis_reach*extension,0.0,0.0],dtype=np.float64)
            seg=btip-a; seg2=float(np.dot(seg,seg))
            if seg2<=1.0e-18:
                closest=a
            else:
                u=float(np.clip(np.dot(rel_body-a,seg)/seg2,0.0,1.0))
                closest=a+u*seg
            mouth_distance=float(np.linalg.norm(rel_body-closest))
            contact_limit=float(resource.radius)+labellum_radius
            if mouth_distance <= contact_limit:
                mouth_contact=True
                # Once a taste sensillum is physically wetted, chemistry determines
                # response strength; extension determines reach, not sweetness.  A soft
                # labellar-radius edge avoids a one-coordinate discontinuity.
                penetration=contact_limit-mouth_distance
                labellar_access=float(np.clip(penetration/max(labellum_radius,1.0e-9),0.0,1.0))

        sweet=float(chem["sugar"]); water=float(chem["water"]); lowsalt=float(chem["low_salt"])
        if foot_contact:
            # Leg taste occurs on physical resource contact; these are appetitive leg
            # GRNs from the 2026 MaleCNS gustatory connectome, not a host "food" flag.
            # Strong-contact ceiling is calibrated to the 25..200 Hz input range used
            # in published connectome/LIF taste simulations.  It is a model-input scale,
            # not a claim that every natural sweet contact fires every GRN at 200 Hz.
            if len(self.lglg3): drives.append((self.lglg3,200.0*sweet))
            if len(self.lglg4): drives.append((self.lglg4,200.0*max(sweet,0.65*lowsalt)))

        if mouth_contact:
            # Labellar receptors are driven by actual mouth contact, independent of
            # whether the feet are also on the patch.
            if len(self.lb3a): drives.append((self.lb3a,180.0*water*labellar_access))
            if len(self.lb3b): drives.append((self.lb3b,200.0*max(sweet,0.6*lowsalt)*labellar_access))
            if len(self.lb3c): drives.append((self.lb3c,200.0*sweet*labellar_access))
            if chem["high_salt"]>0 and len(self.lb3d): drives.append((self.lb3d,200.0*chem["high_salt"]*labellar_access))
            if chem["bitter"]>0 and len(self.lb1): drives.append((self.lb1,200.0*chem["bitter"]*labellar_access))
            # 2026 typing separates taste pegs: ctpGRNs match Ir56d/Gr64e
            # patterns; IR56d taste-peg neurons are established carbonation/fatty-
            # acid sensors.  dtpGRNs match Gr5a/Ir60d; only the Gr5a-supported
            # sweet component is represented here. Unsupported ligands stay silent.
            if len(self.claw): drives.append((self.claw,28.0*chem["carbonation"]*labellar_access))
            if len(self.dorsal): drives.append((self.dorsal,22.0*sweet*labellar_access))

            # PhG1 is a putative Gr64e pharyngeal class with strong feeding/endocrine
            # connectivity. It only sees food once pumping physically puts material in
            # the pharynx; no pumping = no pharyngeal taste drive.
            if pump>0 and len(self.phg1): drives.append((self.phg1,42.0*max(sweet,water)*pump))

        return drives,{"resource_id":resource.patch_id,"distance":float(d),"odor":odor,"contact":foot_contact,
                       "foot_contact":foot_contact,"mouth_contact":bool(mouth_contact),"mouth_distance":float(mouth_distance),
                       "fermentation":float(resource.fermentation),"nutrient_density":float(resource.nutrient_density),
                       "remaining":float(remaining),"bearing_yaw_deg":bearing,"chemistry":chem,
                       "volatile_flow_body":volatile_flow_body.tolist(),
                       "proboscis_contact":float(labellar_access if mouth_contact else 0.0),
                       "proboscis_extension":float(extension),
                       "pump_contact":float(pump if mouth_contact else 0.0)}

    def consume_if_feeding(self, world, body, motor_frame, sensory, dt_s:float) -> float:
        """Fluid ingestion follows actual pump output continuously while on food.

        The old ``feeding_command > .04`` gate made a starving fly with weak but real
        pump activity incapable of taking even a tiny sip.  Drosophila MN11/MN12 drive
        the cibarial/pharyngeal pump; here ingestion therefore scales continuously with
        the expressed pharyngeal-pump effector.  No motor activity -> no eating.
        """
        rid=sensory.get("resource_id")
        # Fluid can be ingested whenever the mouth is physically on the resource;
        # standing on the patch with the feet is neither necessary nor sufficient.
        mouth_contact=bool(sensory.get("mouth_contact",sensory.get("contact",False)))
        if not rid or not mouth_contact or motor_frame is None: return 0.0
        eff=getattr(motor_frame,"effectors",None)
        if eff is None: return 0.0
        pump=float(np.clip(float(eff.pharyngeal_pump),0,1))
        extension=float(np.clip(max(0.0,float(eff.proboscis_extension)),0,1))
        if pump<=1e-9: return 0.0
        # Contact quality is geometric.  Once the mouth is wetted, pump output moves
        # fluid; extension itself is not treated as a nutrient/reward multiplier.
        contact_quality=float(np.clip(float(sensory.get("proboscis_contact",1.0)),0.0,1.0))
        access=0.55+0.45*contact_quality
        rate=0.16*pump*access*float(sensory.get("nutrient_density",1.0))
        return float(world.consume(rid,rate*max(float(dt_s),0.0),current_remaining=float(sensory.get("remaining",0.0))))

    def census(self):
        out={name:int(len(ids)) for name,ids in self.orn.items()}
        out.update({"LgLG3":len(self.lglg3),"LgLG4":len(self.lglg4),"LB3a":len(self.lb3a),
                    "LB3b":len(self.lb3b),"LB3c":len(self.lb3c),"LB3d":len(self.lb3d),
                    "LB1a-d":len(self.lb1),"PhG1a-c":len(self.phg1),
                    "claw_tpGRN":len(self.claw),"dorsal_tpGRN":len(self.dorsal)})
        return out


class PlanetarySensoryBridge:
    """Weak non-nutritional world-scale volatile field for habitat salience.

    This is an alternate-universe environmental field, not a waypoint.  Every habitat
    emits a deterministic ambient chemical fingerprint independent of resources or
    nutrient value.  The field attenuates with geometry/distance, stimulates only real
    retained ORNs, and supplies a local carrier-flow vector to the existing Johnston-
    organ mechanosensory path.  No desired heading, motor command, reward, world ID, or
    visit credit is injected into the nervous system.
    """

    FIELD_MAX_ALT = 24.0
    FIELD_LENGTH_SCALE = 14.0
    FIELD_SURFACE_STRENGTH = 0.55
    ORN_RATE_HZ = 42.0

    def __init__(self, core):
        self.core=core
        self.orn={name:core.ids(neuron_type=name) for name in FoodSensoryBridge.ORN_MIXTURE}
        inst=np.char.lower(core.instance.astype(str)); side=np.char.lower(core.side.astype(str))
        self._left=(side=='l')|(side=='left')|np.char.endswith(inst,'_l')
        self._right=(side=='r')|(side=='right')|np.char.endswith(inst,'_r')

    def _split(self,ids):
        ids=np.asarray(ids,dtype=np.int32)
        l=ids[self._left[ids]]; r=ids[self._right[ids]]
        u=ids[~(self._left[ids]|self._right[ids])]
        return l,r,u

    @staticmethod
    def _world_chemistry(body) -> dict[str,float]:
        """Stable ambient fingerprint from immutable world properties/ID.

        Values are environmental mixture strengths only.  They do not encode food,
        safety, desirability, aliases, visit history, or any behavioral target.
        """
        h=hashlib.sha256((str(body.body_id)+':ambient-world-volatiles-v1').encode('utf-8')).digest()
        u=[x/255.0 for x in h[:8]]
        temp=float(np.clip((float(body.surface_temp_c)-8.0)/36.0,0.0,1.0))
        alb=float(np.clip(float(body.albedo),0.0,1.0))
        return {
            'ethyl_propionate': float(np.clip(0.05+0.16*u[0]+0.05*temp,0,0.30)),
            'three_hexanone': float(np.clip(0.04+0.15*u[1],0,0.24)),
            'methyl_acetate': float(np.clip(0.08+0.24*u[2]+0.05*alb,0,0.38)),
            'ethyl_benzoate': float(np.clip(0.05+0.22*u[3],0,0.30)),
            'butyric_acid': float(np.clip(0.04+0.18*u[4]+0.04*(1.0-alb),0,0.28)),
            'geosmin': float(np.clip(0.02+0.12*u[5],0,0.16)),
            'carbon_dioxide': float(np.clip(0.04+0.15*u[6]+0.08*temp,0,0.25)),
            'pentyl_acetate': float(np.clip(0.03+0.12*u[7],0,0.18)),
        }

    @classmethod
    def _field_strength(cls, body, distance:float) -> float:
        r=max(1e-9,float(body.radius)); d=max(r,float(distance)); alt=max(0.0,d-r)
        if alt>cls.FIELD_MAX_ALT:return 0.0
        # Spherical dilution-like geometry plus a gentle finite-range decay.  The
        # surface value is capped and the field becomes weak long before sector scale.
        ratio=float(np.clip(r/d,0.0,1.0))
        return float(np.clip(cls.FIELD_SURFACE_STRENGTH*(ratio**1.20)*
                             math.exp(-alt/cls.FIELD_LENGTH_SCALE),0.0,cls.FIELD_SURFACE_STRENGTH))

    def observe(self, world, body):
        p=np.asarray(body.position,dtype=float)
        accum={name:[0.0,0.0,0.0] for name in self.orn}  # left/right/unpaired Hz
        carrier=np.zeros(3,dtype=np.float64)
        total=0.0; dominant_id=None; dominant_strength=0.0
        dominant_distance=float('inf'); dominant_bearing=None
        sensed_worlds=0

        for habitat in world.local_bodies(p):
            rel=np.asarray(habitat.center,dtype=float)-p
            dist=float(np.linalg.norm(rel))
            if dist<=1e-12:continue
            strength=self._field_strength(habitat,dist)
            if strength<=1e-8:continue
            sensed_worlds+=1
            dbody=q_inverse_rotate(body.orientation,(rel/dist).reshape(1,3))[0]
            lateral=float(np.clip(dbody[1],-1.0,1.0))
            chemistry=self._world_chemistry(habitat)

            # Independent ambient fingerprints superpose at the antennae.  This is
            # concentration geometry, not a host selection of the nearest target.
            for name,weights in FoodSensoryBridge.ORN_MIXTURE.items():
                response=float(np.clip(sum(float(chemistry.get(comp,0.0))*float(weight)
                                           for comp,weight in weights.items()),0.0,1.0))
                if response<=0.0:continue
                base=self.ORN_RATE_HZ*strength*response
                accum[name][0]+=base*(1.0-0.26*lateral)
                accum[name][1]+=base*(1.0+0.26*lateral)
                accum[name][2]+=base

            # Volatile material radiates from habitat to receiver.  At the fly that
            # physical carrier travels source->fly, hence -dbody.  Multiple worlds
            # vector-sum naturally and can cancel; no destination vector is exposed.
            carrier += -np.asarray(dbody,dtype=np.float64)*strength
            total += strength
            if strength>dominant_strength:
                dominant_strength=float(strength);dominant_id=str(habitat.body_id)
                dominant_distance=float(max(0.0,dist-float(habitat.radius)))
                dominant_bearing=float(math.degrees(math.atan2(float(dbody[1]),float(dbody[0]))))

        drives=[]
        for name,ids in self.orn.items():
            l,r,u=self._split(ids); lh,rh,uh=accum[name]
            # Cap only the external receptor rate presented by this one environmental
            # field.  Food odor remains a separate superposed sensory source.
            if len(l) and lh>0.0:drives.append((l,float(np.clip(lh,0.0,36.0))))
            if len(r) and rh>0.0:drives.append((r,float(np.clip(rh,0.0,36.0))))
            if len(u) and uh>0.0:drives.append((u,float(np.clip(uh,0.0,36.0))))

        cmag=float(np.linalg.norm(carrier))
        if cmag>1.0:carrier/=cmag
        return drives,{
            'dominant_world_id':dominant_id,
            'dominant_world_distance':float(dominant_distance),
            'dominant_world_bearing_yaw_deg':dominant_bearing,
            'world_scent':float(np.clip(total,0.0,1.0)),
            'dominant_world_scent':float(dominant_strength),
            'sensed_worlds':int(sensed_worlds),
            'world_carrier_flow_body':carrier.tolist(),
        }

    def census(self):
        return {
            'world_orn_cells':int(len(np.unique(np.concatenate([v for v in self.orn.values() if len(v)]))) if any(len(v) for v in self.orn.values()) else 0),
            'field_max_alt':float(self.FIELD_MAX_ALT),
            'field_surface_strength':float(self.FIELD_SURFACE_STRENGTH),
            'host_steering_or_reward':0,
        }


class HygroThermoSupplementBridge:
    """Real MaleCNS sacculus populations missing from the old thermal-only bridge.

    Humidity is an environmental scalar derived from a deterministic habitat baseline
    plus local evaporative resource vapor.  It is not a reward and carries no resource
    bearing.  VP4/VP1d are dry-biased Ir40a populations; VP5/VP1m are Ir68a moist
    populations; VP1l is the Ir21a cool population below ~25 C.
    """
    def __init__(self,core):
        self.vp4=core.ids(neuron_type="HRN_VP4")
        self.vp1d=core.ids(neuron_type="HRN_VP1d")
        self.vp5=core.ids(neuron_type="HRN_VP5")
        self.vp1m=core.ids(neuron_type="TRN_VP1m")
        self.vp1l=core.ids(neuron_type="HRN_VP1l")

    @staticmethod
    def relative_humidity(world,body,food_sensory:dict[str,Any]|None=None) -> tuple[float,float]:
        # Humidity is body-local.  Previously the nearest body's humidity was applied
        # at full strength at arbitrary altitude, effectively filling space with that
        # world's air.  Keep a neutral reference for telemetry, but fade all actual
        # hygrosensory drive with the body's atmosphere occupancy.
        b,alt,_=world.nearest_body(body.position)
        atm=float(world.atmosphere_fraction(body.position,b)) if hasattr(world,"atmosphere_fraction") else 1.0
        local=0.52
        if b is not None:
            h=hashlib.sha256((str(b.body_id)+":humidity").encode()).digest()
            local=0.36+0.34*(h[0]/255.0)
            if getattr(body.contact,"on_surface",False) and body.contact.body_id==b.body_id:
                local+=0.06

        # Local evaporative vapor from the nearest resource is computed from resource
        # substrate/water potential and distance directly and cannot leak past the
        # atmosphere shell.
        rb,r,d=world.nearest_resource(body.position)
        if r is not None and rb is not None and math.isfinite(float(d)):
            remain=float(world.resource_remaining(r.patch_id,r.amount))
            frac=remain/max(float(r.amount),1e-9)
            hh=hashlib.sha256((str(r.patch_id)+":water").encode()).digest()
            substrate_water=0.38+0.42*(hh[0]/255.0)
            water_potential=float(np.clip(substrate_water*(1.0-0.22*float(r.fermentation)),0,1))
            vapor=water_potential*frac*math.exp(-max(0.0,float(d))/4.0)
            local += 0.18*vapor*float(world.atmosphere_fraction(body.position,rb))
        rh=float(np.clip(0.52+atm*(local-0.52),0.05,0.98))
        return rh,atm

    def drives(self,world,body,thermal_frame,food_sensory=None):
        rh,atm=self.relative_humidity(world,body,food_sensory)
        dry=float(np.clip((0.62-rh)/0.42,0,1)); moist=float(np.clip((rh-0.42)/0.46,0,1))
        mean_t=0.5*(float(thermal_frame.left_temp_c)+float(thermal_frame.right_temp_c))
        cooling=max(0.0,-0.5*(float(thermal_frame.left_dtemp_c_s)+float(thermal_frame.right_dtemp_c_s)))
        cool=float(np.clip((25.0-mean_t)/8.0 + 0.20*np.clip(cooling/2.0,0,1),0,1))
        out=[]
        for ids,hz in ((self.vp4,72.0*dry),(self.vp1d,58.0*np.clip(0.72*dry+0.28*cooling/2.0,0,1)),
                       (self.vp5,68.0*moist),(self.vp1m,58.0*moist),(self.vp1l,62.0*cool)):
            hz=float(hz)*atm
            if len(ids) and hz>0: out.append((ids,hz))
        return out,{"relative_humidity":rh,"atmosphere_fraction":atm,"dry_drive":dry*atm,"moist_drive":moist*atm,"cool_drive":cool*atm}

    def census(self):
        return {"HRN_VP4":len(self.vp4),"HRN_VP1d":len(self.vp1d),"HRN_VP5":len(self.vp5),
                "TRN_VP1m":len(self.vp1m),"HRN_VP1l":len(self.vp1l)}
