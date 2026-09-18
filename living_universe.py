#!/usr/bin/env python3
"""Procedural habitats/resources and fly-like contact mechanics for Specimen-003.

The old infinite star field remains intact.  This layer adds deterministic bodies to
interact with, rather than secretly giving the connectome a target coordinate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import OrderedDict
import hashlib
import math
import re
from typing import Any

import numpy as np

from space_fly_universe import (
    ProceduralUniverse, SpacecraftState, RadiantThermalSensor, ThermalFrame,
    q_rotate, q_inverse_rotate, q_mul, q_normalize,
)




@dataclass(frozen=True)
class VisualSource:
    star_id: str
    position: np.ndarray
    luminosity: float
    temperature: float = 5800.0
    pulse_phase: float = 0.0
    spectral_weights: tuple[float,float,float,float,float] | None = None
    # For surface-attached visual features (food patches), identify the sphere they
    # belong to so the compound-eye occlusion test does not hide the patch behind
    # the very surface it is sitting on.  This is geometric provenance only.
    surface_body_id: str | None = None

@dataclass(frozen=True)
class ResourcePatch:
    patch_id: str
    body_id: str
    direction: np.ndarray
    nutrient_density: float
    fermentation: float
    amount: float
    radius: float = 0.28


@dataclass(frozen=True)
class HabitatBody:
    body_id: str
    center: np.ndarray
    radius: float
    surface_temp_c: float
    albedo: float
    resources: tuple[ResourcePatch, ...]

    def resource_position(self, r: ResourcePatch) -> np.ndarray:
        return self.center + np.asarray(r.direction, dtype=np.float64)*(self.radius+0.015)


@dataclass
class ContactState:
    on_surface: bool = False
    # body_id is strictly the body actually in surface contact (None in free flight).
    body_id: str | None = None
    surface_temp_c: float | None = None
    normal_world: np.ndarray | None = None
    # altitude is always radial clearance above the nearest habitat-body surface.
    # altitude_body_id names that reference body even when the fly is not attached.
    altitude: float = float("inf")
    altitude_body_id: str | None = None
    # Smooth local-atmosphere occupancy used only by physical environmental fields
    # and display semantics.  It is not a target/reward/steering signal.
    atmosphere_fraction: float = 0.0
    nearest_resource_id: str | None = None
    nearest_resource_distance: float = float("inf")


class LivingUniverse:
    """Starfield plus deterministic planets/moons and nutrient patches."""

    # Procedural sectors are deterministic and cheap to regenerate.  Keep only a few
    # local neighborhoods resident instead of allowing long-distance travel to
    # turn the cache into a permanent materialized universe.
    BODY_CACHE_MAX_SECTORS = 96

    # Local atmosphere shell.  The original code accidentally let body humidity be
    # spatially constant throughout every surrounding sector, so a fly many body
    # radii away still received terrestrial hygrosensory drive.  Four body radii is
    # an explicit simulation-environment scale, not a claim about real planetary
    # atmospheres.  The exact numerical altitude is retained outside this shell.
    ATMOSPHERE_HEIGHT_RADII = 4.0

    def __init__(self, seed: str = "no-mans-fly-003", *, star_sector_size: float = 30.0):
        self.seed = str(seed)
        self.stars = ProceduralUniverse(self.seed, sector_size=star_sector_size)
        self.sector_size = float(star_sector_size)
        self._body_cache: OrderedDict[tuple[int,int,int], tuple[HabitatBody,...]] = OrderedDict()
        self._resource_remaining: dict[str,float] = {}
        # Runtime/environmental resources that are not part of the deterministic
        # procedural body definition.  These are persisted explicitly so a rescue
        # fruit cannot disappear/reappear across save/resume.
        self._dynamic_resources: dict[str,ResourcePatch] = {}
        self._divine_fruit_serial: int = 0
        # Persistent human atlas metadata. Procedural worlds may fall out of the
        # local body cache, but their immutable designation/coordinates and Yar's
        # visit history remain here. Aliases never replace body_id internally.
        self._world_aliases: dict[str,str] = {}
        self._world_atlas: dict[str,dict[str,Any]] = {}
        self._travel_history: list[dict[str,Any]] = []
        self._intervention_history: list[dict[str,Any]] = []
        self._atlas_surface_body_id: str|None = None

    def body_by_id(self, body_id: str) -> HabitatBody | None:
        """Reconstruct one deterministic procedural body from its designation."""
        bid=str(body_id)
        if bid=="H+0+0+0-bootstrap":
            return next((b for b in self.bodies_in_sector((0,0,0),cache=False) if b.body_id==bid),None)
        m=re.fullmatch(r"H([+-]\d+)([+-]\d+)([+-]\d+)-(\d+)",bid)
        if not m:
            return None
        coord=(int(m.group(1)),int(m.group(2)),int(m.group(3)))
        return next((b for b in self.bodies_in_sector(coord,cache=False) if b.body_id==bid),None)

    def world_alias(self, body_id: str) -> str:
        return str(self._world_aliases.get(str(body_id),""))

    def world_display_name(self, body_id: str) -> str:
        alias=self.world_alias(body_id).strip()
        return alias if alias else str(body_id)


    def _ensure_atlas_body(self, body: HabitatBody, time_s: float|None=None) -> dict[str,Any]:
        bid=str(body.body_id)
        rec=self._world_atlas.get(bid)
        if rec is None:
            rec={
                "body_id":bid,
                "center":[float(x) for x in np.asarray(body.center,float)],
                "radius":float(body.radius),
                "surface_temp_c":float(body.surface_temp_c),
                "albedo":float(body.albedo),
                "first_encounter_s":None,
                "last_encounter_s":None,
                "first_landed_s":None,
                "last_landed_s":None,
                "landing_count":0,
                "closest_altitude":None,
                "visit_order":len(self._world_atlas),
            }
            self._world_atlas[bid]=rec
        else:
            # Re-derive immutable procedural geometry rather than trusting stale UI data.
            rec["center"]=[float(x) for x in np.asarray(body.center,float)]
            rec["radius"]=float(body.radius);rec["surface_temp_c"]=float(body.surface_temp_c);rec["albedo"]=float(body.albedo)
        return rec

    def set_world_alias(self, body_id: str, alias: str|None, *, body: HabitatBody|None=None, time_s: float|None=None) -> str:
        bid=str(body_id); name=("" if alias is None else str(alias)).strip()[:80]
        if name:
            self._world_aliases[bid]=name
        else:
            self._world_aliases.pop(bid,None)
        b=body if body is not None else self.body_by_id(bid)
        if b is not None:
            self._ensure_atlas_body(b,time_s)
        return name

    def record_world_visit(self, body: HabitatBody|None, *, time_s: float, altitude: float, on_surface: bool, position: np.ndarray|None=None):
        """Persist encounter/landing history without keeping inactive worlds loaded."""
        if body is None:
            self._atlas_surface_body_id=None
            return
        rec=self._ensure_atlas_body(body,time_s)
        t=float(time_s); alt=float(altitude)
        first=rec.get("first_encounter_s") is None
        if first:
            rec["first_encounter_s"]=t
            self._travel_history.append({"time_s":t,"kind":"encountered","body_id":body.body_id,"center":[float(x) for x in body.center]})
        rec["last_encounter_s"]=t
        old=rec.get("closest_altitude")
        if math.isfinite(alt) and (old is None or alt<float(old)):
            rec["closest_altitude"]=alt
        if on_surface:
            if self._atlas_surface_body_id != body.body_id:
                rec["landing_count"]=int(rec.get("landing_count",0))+1
                if rec.get("first_landed_s") is None: rec["first_landed_s"]=t
                rec["last_landed_s"]=t
                self._travel_history.append({"time_s":t,"kind":"landed","body_id":body.body_id,"center":[float(x) for x in body.center]})
            self._atlas_surface_body_id=body.body_id
        else:
            self._atlas_surface_body_id=None
        if position is not None:
            rec["last_position"]=[float(x) for x in np.asarray(position,float)]
        if len(self._travel_history)>5000:
            self._travel_history=self._travel_history[-5000:]

    def prime_world_contact(self, body_id: str|None):
        """Resume helper: avoid counting a persisted landing twice on first step."""
        self._atlas_surface_body_id=None if body_id is None else str(body_id)

    def backfill_atlas_from_events(self, events: list[dict[str,Any]]):
        """Best-effort migration of pre-atlas landing history from checkpoint events."""
        surface=self._atlas_surface_body_id
        for ev in events or []:
            kind=str(ev.get("kind")); bid=str(ev.get("body_id",""))
            if kind=="takeoff":
                if surface==bid: surface=None
                continue
            if kind!="landed":
                continue
            b=self.body_by_id(bid)
            if b is None: continue
            t=float(ev.get("time_s",0.0)); rec=self._ensure_atlas_body(b,t)
            if rec.get("first_encounter_s") is None: rec["first_encounter_s"]=t
            rec["last_encounter_s"]=max(float(rec.get("last_encounter_s") or t),t)
            if rec.get("first_landed_s") is None:
                rec["first_landed_s"]=t; rec["landing_count"]=1
                self._travel_history.append({"time_s":t,"kind":"landed","body_id":bid,"center":[float(x) for x in b.center],"migrated":True})
            rec["last_landed_s"]=max(float(rec.get("last_landed_s") or t),t)
            surface=bid
        self._atlas_surface_body_id=surface

    def atlas_snapshot(self) -> dict[str,Any]:
        worlds=[]
        for bid,rec in self._world_atlas.items():
            q=dict(rec); q["alias"]=self.world_alias(bid); worlds.append(q)
        worlds.sort(key=lambda q:(int(q.get("visit_order",10**9)),str(q.get("body_id"))))
        return {"worlds":worlds,"travel_history":[dict(x) for x in self._travel_history],
                "interventions":[dict(x) for x in self._intervention_history],
                "aliases":dict(self._world_aliases)}

    def record_intervention(self, event:dict[str,Any]):
        q=dict(event); self._intervention_history.append(q)
        if len(self._intervention_history)>5000:self._intervention_history=self._intervention_history[-5000:]









    # Thermal sensing sees real procedural stars only.
    def local_stars(self, position: np.ndarray):
        return self.stars.local_stars(position)

    def local_visual_sources(self, position: np.ndarray):
        """Point-light sources for the compound eye: stars plus surface resources.

        Habitat *surfaces* are intentionally not converted to a 32-point cloud here.
        Patch22 moves them to continuous sphere optics inside ``CompoundEyeBridge``:
        each ommatidium tests the actual sphere silhouette and nearest surface hit.
        This removes the old popping/aliasing as a fly walks across a world.
        """
        p=np.asarray(position,dtype=float)
        out=list(self.stars.local_stars(p))
        for b in self.local_bodies(p):
            rel_obs=p-b.center; d=max(float(np.linalg.norm(rel_obs)),1e-6)
            view=rel_obs/d
            for r in self.resources_for_body(b):
                remain=self._resource_remaining.get(r.patch_id,r.amount)
                # A far-side patch is physically hidden by its own world.  The
                # compound-eye bridge performs the final ray/sphere occlusion test.
                if remain<=1e-9 or float(np.dot(np.asarray(r.direction,float),view))<=0.0:
                    continue
                q=b.resource_position(r)
                hh=hashlib.sha256((str(r.patch_id)+":reflectance").encode()).digest()
                uu=[x/255.0 for x in hh[:5]]
                rl=0.28+0.32*(sum(uu)/5.0)+0.12*r.fermentation
                sw=tuple(float(np.clip(0.22+0.58*uu[j]+(0.18*r.fermentation if j in (0,1) else 0.0),0.05,1.0)) for j in range(5))
                out.append(VisualSource(f"resource:{r.patch_id}",q,rl,5000.0,0.0,sw,b.body_id))
        return out


    def identity_probe(self, coord=(0,0,0)):
        return self.stars.identity_probe(coord)

    def _rng(self, coord: tuple[int,int,int]):
        h=hashlib.sha256((self.seed+":habitat:"+",".join(map(str,coord))).encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8],"little"))

    def _sector_coord(self, position: np.ndarray) -> tuple[int,int,int]:
        return tuple(np.floor(np.asarray(position,dtype=float)/self.sector_size).astype(int))

    @staticmethod
    def _unit(rng) -> np.ndarray:
        v=rng.normal(size=3); return v/max(np.linalg.norm(v),1e-12)

    def bodies_in_sector(self, coord: tuple[int,int,int], *, cache: bool = True) -> tuple[HabitatBody,...]:
        coord=tuple(map(int,coord))
        if cache and coord in self._body_cache:
            out=self._body_cache.pop(coord); self._body_cache[coord]=out
            return out
        rng=self._rng(coord); base=np.asarray(coord,dtype=float)*self.sector_size
        n=int(rng.integers(1,4))
        out=[]
        for j in range(n):
            center=base+rng.uniform(3.0,self.sector_size-3.0,size=3)
            radius=float(rng.uniform(0.75,1.65))
            temp=float(rng.uniform(8.0,44.0))
            # Food is commonest in the broad warm band but not guaranteed there.
            warm=np.clip(1.0-abs(temp-27.0)/18.0,0.0,1.0)
            nr=int(rng.integers(0,4)) if rng.random() < (0.20+0.65*warm) else 0
            rs=[]
            bid=f"H{coord[0]:+d}{coord[1]:+d}{coord[2]:+d}-{j}"
            for k in range(nr):
                rid=f"{bid}-R{k}"
                r=ResourcePatch(rid,bid,self._unit(rng),float(rng.uniform(.35,1.0)),
                                float(rng.uniform(0.0,1.0)),float(rng.uniform(.6,1.8)))
                rs.append(r)
            out.append(HabitatBody(bid,center,radius,temp,float(rng.uniform(.12,.75)),tuple(rs)))

        # The origin sector gets one deterministic nearby habitat so the first life
        # experiment has something reachable without the brain receiving a waypoint.
        if coord==(0,0,0):
            bid="H+0+0+0-bootstrap"
            rr=np.random.default_rng(int.from_bytes(hashlib.sha256((self.seed+":bootstrap").encode()).digest()[:8],"little"))
            rs=[]
            for k in range(3):
                rid=f"{bid}-R{k}"
                r=ResourcePatch(rid,bid,self._unit(rr),float(rr.uniform(.55,1.0)),float(rr.uniform(.35,.95)),1.4)
                rs.append(r)
            out.append(HabitatBody(bid,np.array([8.0,5.0,7.0]),1.15,28.0,.35,tuple(rs)))
        result=tuple(out)
        if cache:
            self._body_cache[coord]=result
            while len(self._body_cache)>int(self.BODY_CACHE_MAX_SECTORS):
                self._body_cache.popitem(last=False)
        return result

    def local_bodies(self, position: np.ndarray, radius_sectors: int = 1) -> list[HabitatBody]:
        c=self._sector_coord(position); out=[]
        r=int(radius_sectors)
        for x in range(c[0]-r,c[0]+r+1):
            for y in range(c[1]-r,c[1]+r+1):
                for z in range(c[2]-r,c[2]+r+1): out.extend(self.bodies_in_sector((x,y,z)))
        return out

    def nearest_body(self, position: np.ndarray) -> tuple[HabitatBody|None,float,np.ndarray|None]:
        p=np.asarray(position,dtype=float); best=None; bd=float("inf"); normal=None
        for b in self.local_bodies(p):
            rel=p-b.center; d=float(np.linalg.norm(rel)); alt=d-b.radius
            if alt<bd:
                best,bd=b,alt; normal=rel/max(d,1e-12)
        return best,bd,normal


    @classmethod
    def atmosphere_height(cls, body: HabitatBody | None) -> float:
        if body is None:
            return 0.0
        return max(0.0, float(body.radius) * float(cls.ATMOSPHERE_HEIGHT_RADII))

    def atmosphere_fraction(self, position: np.ndarray, body: HabitatBody | None = None) -> float:
        """Smooth 1->0 local-atmosphere occupancy for one physical body.

        This is an environmental field boundary, never a neural semantic.  Surface
        contact is 1.0; at/above the configured atmosphere height it is 0.0.
        """
        p=np.asarray(position,dtype=float)
        b=body
        if b is None:
            b,alt,_=self.nearest_body(p)
        else:
            alt=float(np.linalg.norm(p-np.asarray(b.center,float)))-float(b.radius)
        if b is None:
            return 0.0
        h=max(self.atmosphere_height(b),1e-12)
        x=float(np.clip(max(0.0,float(alt))/h,0.0,1.0))
        # 1-smoothstep(x): finite slope at neither boundary, avoiding sensory clicks.
        return float(1.0-(3.0*x*x-2.0*x*x*x))

    def gravity_at(self, position: np.ndarray, magnitude: float) -> tuple[HabitatBody|None,np.ndarray]:
        """Dominant local-body gravity with inverse-square falloff.

        ``magnitude`` is the calibrated acceleration at a habitat body's surface.
        Outside a body the field falls as (radius / distance)^2; inside the surface
        it is capped at the surface value because solid-body collision/contact owns
        that regime.  The dominant local field is selected by acceleration strength,
        giving neighboring bodies a natural sphere-of-influence boundary instead of
        a constant-strength nearest-body gravity cage.  No target/steering semantics
        are involved.
        """
        p=np.asarray(position,dtype=float)
        surface_g=max(0.0,float(magnitude))
        best=None; best_g=-1.0; best_vec=np.zeros(3,dtype=np.float64)
        for b in self.local_bodies(p):
            rel=p-b.center; d=max(float(np.linalg.norm(rel)),1e-12)
            normal=rel/d
            ratio=min(1.0,float(b.radius)/d)
            g=surface_g*ratio*ratio
            if g>best_g:
                best=b; best_g=g; best_vec=-normal*g
        return best,best_vec

    def resources_for_body(self, body: HabitatBody) -> tuple[ResourcePatch,...]:
        """Static procedural resources plus persisted runtime drops for one body."""
        dyn=tuple(r for r in self._dynamic_resources.values() if r.body_id==body.body_id)
        return tuple(body.resources)+dyn

    def nearest_resource(self, position: np.ndarray) -> tuple[HabitatBody|None,ResourcePatch|None,float]:
        p=np.asarray(position,dtype=float); bb=rr=None; bd=float("inf")
        for b in self.local_bodies(p):
            for r in self.resources_for_body(b):
                if self._resource_remaining.get(r.patch_id,r.amount)<=1e-9: continue
                d=float(np.linalg.norm(p-b.resource_position(r)))
                if d<bd: bb,rr,bd=b,r,d
        return bb,rr,bd

    def active_divine_fruit_distance(self, position: np.ndarray) -> float:
        """Distance to the nearest still-edible rescue fruit; observer/control only."""
        p=np.asarray(position,dtype=float); best=float("inf")
        for b in self.local_bodies(p):
            for r in self.resources_for_body(b):
                if "-DIVINE-FRUIT-" not in r.patch_id: continue
                if self._resource_remaining.get(r.patch_id,r.amount)<=1e-9: continue
                best=min(best,float(np.linalg.norm(p-b.resource_position(r))))
        return best

    def drop_divine_fruit(self, position: np.ndarray, preferred_body_id: str|None = None) -> tuple[HabitatBody,ResourcePatch,float]|None:
        """Place/refill the single active emergency food patch at Yar's surface.

        Patch18 allowed every manual care-package press to create another persistent
        coconut. Patch19 treats divine manna as one environmental rescue object: if an
        uneaten divine fruit already exists, it is relocated and refilled rather than
        multiplying. This still changes only the world; Yar must sense and eat it.
        """
        p=np.asarray(position,dtype=float)
        bodies=self.local_bodies(p)
        if not bodies: return None
        b=None
        if preferred_body_id is not None:
            b=next((x for x in bodies if x.body_id==preferred_body_id),None)
        if b is None:
            b=min(bodies,key=lambda x: abs(float(np.linalg.norm(p-x.center))-float(x.radius)))
        rel=p-np.asarray(b.center,float); nr=float(np.linalg.norm(rel))
        direction=np.array([1.0,0.0,0.0],dtype=np.float64) if nr<=1e-12 else rel/nr

        active=[]
        for rid,r0 in list(self._dynamic_resources.items()):
            if "-DIVINE-FRUIT-" in rid and self._resource_remaining.get(rid,r0.amount)>1e-9:
                active.append((rid,r0))
        if active:
            # Keep one stable ID on the same body, consume/remove any accidental
            # Patch18 duplicates. If Yar has moved to a different world, retire the old
            # package and mint one correctly named for the new body.
            active=sorted(active,key=lambda x:x[0])
            keep_rid,keep_r=active[0]
            for oldrid,_ in active[1:]:
                self._resource_remaining[oldrid]=0.0
                self._dynamic_resources.pop(oldrid,None)
            if keep_r.body_id == b.body_id:
                rid=keep_rid
            else:
                self._resource_remaining[keep_rid]=0.0
                self._dynamic_resources.pop(keep_rid,None)
                self._divine_fruit_serial += 1
                rid=f"{b.body_id}-DIVINE-FRUIT-{self._divine_fruit_serial:04d}"
        else:
            self._divine_fruit_serial += 1
            rid=f"{b.body_id}-DIVINE-FRUIT-{self._divine_fruit_serial:04d}"

        r=ResourcePatch(rid,b.body_id,np.asarray(direction,dtype=np.float64),1.0,1.0,4.0,0.70)
        self._dynamic_resources[rid]=r
        self._resource_remaining[rid]=float(r.amount)
        d=float(np.linalg.norm(p-b.resource_position(r)))
        return b,r,d

    def consume(self, patch_id: str, amount: float, *, current_remaining: float|None = None) -> float:
        """Consume resource material using sparse overrides.

        Deterministic procedural resources are not persisted until modified.  Callers
        that already resolved the physical ResourcePatch provide its current/default
        remaining amount on the first consumption.
        """
        rid=str(patch_id)
        default=0.0 if current_remaining is None else float(current_remaining)
        cur=float(self._resource_remaining.get(rid,default)); take=min(cur,max(0.0,float(amount)))
        self._resource_remaining[rid]=cur-take
        return take

    def resource_remaining(self, patch_id: str, default_amount: float = 0.0) -> float:
        return float(self._resource_remaining.get(str(patch_id),float(default_amount)))

    @staticmethod
    def _procedural_resource_coord(patch_id: str):
        m=re.fullmatch(r"H([+-]\d+)([+-]\d+)([+-]\d+)-(\d+)-R(\d+)",str(patch_id))
        if not m:return None
        return (int(m.group(1)),int(m.group(2)),int(m.group(3)))

    def _compact_legacy_resource_overrides(self) -> int:
        """Drop legacy entries that merely duplicate deterministic procedural defaults."""
        if not self._resource_remaining:return 0
        dynamic=set(self._dynamic_resources)
        groups={}
        for rid in list(self._resource_remaining):
            if rid in dynamic:continue
            coord=self._procedural_resource_coord(rid)
            if coord is not None:groups.setdefault(coord,[]).append(rid)
        removed=0
        for coord,rids in groups.items():
            defaults={}
            for b in self.bodies_in_sector(coord,cache=False):
                for r in b.resources:defaults[str(r.patch_id)]=float(r.amount)
            for rid in rids:
                default=defaults.get(rid)
                if default is None:continue
                val=float(self._resource_remaining[rid])
                if math.isclose(val,default,rel_tol=0.0,abs_tol=1.0e-12):
                    self._resource_remaining.pop(rid,None); removed+=1
        return removed

    def snapshot(self) -> dict[str,Any]:
        dynamic=[]
        for r in self._dynamic_resources.values():
            dynamic.append({"patch_id":r.patch_id,"body_id":r.body_id,
                            "direction":np.asarray(r.direction,float).tolist(),
                            "nutrient_density":float(r.nutrient_density),"fermentation":float(r.fermentation),
                            "amount":float(r.amount),"radius":float(r.radius)})
        return {"seed":self.seed,"resource_remaining":dict(self._resource_remaining),
                "dynamic_resources":dynamic,"divine_fruit_serial":int(self._divine_fruit_serial),
                "world_aliases":dict(self._world_aliases),
                "world_atlas":{k:dict(v) for k,v in self._world_atlas.items()},
                "travel_history":[dict(x) for x in self._travel_history],
                "intervention_history":[dict(x) for x in self._intervention_history]}

    def restore(self,d:dict[str,Any]):
        for k,v in d.get("resource_remaining",{}).items(): self._resource_remaining[str(k)]=float(v)
        self._dynamic_resources={}
        for x in d.get("dynamic_resources",[]):
            try:
                r=ResourcePatch(str(x["patch_id"]),str(x["body_id"]),np.asarray(x["direction"],dtype=np.float64),
                                float(x.get("nutrient_density",1.0)),float(x.get("fermentation",1.0)),
                                float(x.get("amount",4.0)),float(x.get("radius",0.70)))
            except (KeyError,TypeError,ValueError):
                continue
            self._dynamic_resources[r.patch_id]=r
            self._resource_remaining.setdefault(r.patch_id,float(r.amount))
        self._divine_fruit_serial=int(d.get("divine_fruit_serial",0))
        self._compact_legacy_resource_overrides()
        self._world_aliases={str(k):str(v) for k,v in (d.get("world_aliases",{}) or {}).items() if str(v).strip()}
        self._world_atlas={}
        for k,v in (d.get("world_atlas",{}) or {}).items():
            if isinstance(v,dict): self._world_atlas[str(k)]=dict(v)
        self._travel_history=[dict(x) for x in (d.get("travel_history",[]) or []) if isinstance(x,dict)]
        self._intervention_history=[dict(x) for x in (d.get("intervention_history",[]) or []) if isinstance(x,dict)]
        # Obsolete active-intervention checkpoint fields are intentionally ignored.
        self._atlas_surface_body_id=None
        if self._dynamic_resources:
            for rid in self._dynamic_resources:
                m=re.search(r"-DIVINE-FRUIT-(\d+)$",rid)
                if m: self._divine_fruit_serial=max(self._divine_fruit_serial,int(m.group(1)))


class FlyBodyState(SpacecraftState):
    """Fly body pose plus contact semantics.

    ``SpacecraftState`` is retained only for its quaternion/pose representation and
    checkpoint compatibility.  Local free flight is atmospheric and gravitational:
    the nearest habitat body supplies "down", wing force must continuously oppose it,
    and body-relative aerodynamics make a changed heading bend the flight path instead
    of letting the fly coast sideways on a spacecraft-like inertial rail.
    """

    # Explicit reference morphology for the embodied insect.  The procedural universe
    # still uses a centimetre-like local coordinate system, but motor command is now
    # converted through a body mass and inertia rather than being treated as a naked
    # acceleration.  The reference values are within measured adult D. melanogaster
    # morphology; they are not claimed to be Yar-specific morphometrics.
    LOCAL_CM_PER_UNIT = 1.0
    LOCAL_M_PER_UNIT = 0.01
    BODY_MASS_KG = 0.702e-6
    BODY_LENGTH_M = 2.26e-3
    BODY_WIDTH_M = 1.20e-3
    BODY_HEIGHT_M = 0.80e-3

    # Ellipsoid approximation, body axes x=longitudinal, y=lateral, z=dorsoventral.
    _A = BODY_LENGTH_M/2.0; _B = BODY_WIDTH_M/2.0; _C = BODY_HEIGHT_M/2.0
    BODY_INERTIA_KGM2 = np.asarray([
        BODY_MASS_KG*(_B*_B+_C*_C)/5.0,
        BODY_MASS_KG*(_A*_A+_C*_C)/5.0,
        BODY_MASS_KG*(_A*_A+_B*_B)/5.0,
    ],dtype=float)

    # ``fly_body.py`` emits normalized force/torque commands.  These reference actuator
    # scales preserve the established useful force envelope while making the mechanical
    # conversion explicit and mass/inertia-dependent.  They contain no target/steering
    # semantics.
    FLIGHT_FORCE_UNIT_N = BODY_MASS_KG * LOCAL_M_PER_UNIT * 3000.0
    SURFACE_FORCE_UNIT_N = BODY_MASS_KG * LOCAL_M_PER_UNIT * 30.0
    TORQUE_UNIT_NM = BODY_INERTIA_KGM2 * 1800.0

    # Calibrated surface gravity.  LivingUniverse.gravity_at() applies inverse-square
    # falloff away from each body's surface and selects the dominant local field.  This
    # preserves the established ~58-unit surface load without turning every habitat
    # into a constant-strength gravity well extending halfway to its neighbor.
    GRAVITY_ACCELERATION = 58.0

    # Free-flight integration is sub-stepped because Yar can now rotate rapidly. This
    # keeps attitude, gravity, lift and airflow coupled within one 0.1 s world tick.
    FREE_FLIGHT_MAX_SUBSTEP = 0.01
    CONTACT_EPSILON = 1.0e-8

    # Air-like translational damping. Forward motion is comparatively easy while
    # lateral/vertical slip damps harder. These values yield a broad ~20-40 local
    # unit/s cruise envelope for the motor commands observed in Yar, without a speed
    # clamp and without any semantic steering.
    LINEAR_DRAG = np.asarray([1.50, 6.00, 2.00], dtype=float)
    QUADRATIC_DRAG = np.asarray([0.015, 0.080, 0.020], dtype=float)

    # Angular aerodynamic damping: ordinary asymmetries produce visible turns quickly
    # while very large commands are self-limiting instead of creating spacecraft-like
    # persistent tumble.
    ANGULAR_LINEAR_DRAG = 2.0
    ANGULAR_QUADRATIC_DRAG = 0.18

    # Six-legged stance supplies a passive righting reaction while the feet are in
    # contact with a curved surface.  This is body mechanics, not a neural command:
    # pitch/roll are spring-damped toward local surface-normal "up" while yaw remains
    # free.  Without this, the simulated fly could walk around a sphere with its body
    # attitude detached from the surface and point wing lift sideways indefinitely.
    SURFACE_STANCE_SPRING = 90.0      # rad/s^2 per rad of tilt error
    SURFACE_STANCE_DAMPING = 14.0     # s^-1, pitch/roll only
    # Passive tarsal traction: a planted, non-walking fly should not ice-skate on
    # residual flight momentum.  Leg-driven gait reduces, but never removes, traction.
    SURFACE_STATIC_TRACTION = 12.0     # s^-1 at negligible gait
    SURFACE_WALK_TRACTION = 2.0        # s^-1 during strong gait

    # Finite tarsal/pulvillar pull-off resistance.  Contact is no longer released by
    # an infinitesimal positive normal acceleration.  This is intentionally modest
    # rather than a wall-climbing maximum: at quiet stance the six-foot contact can
    # oppose ~0.24 body-weight-equivalent of additional separating acceleration in
    # the local mechanics.  During strong leg cycling, an alternating-tripod-like
    # contact fraction reduces available adhesion without any dwell timer or semantic
    # takeoff gate.
    TARSAL_ADHESION_MAX_ACCEL = 14.0   # local units/s^2 at quiet six-foot stance
    TARSAL_WALK_CONTACT_FRACTION = 0.50

    def __init__(self, position, velocity, orientation, angular_velocity):
        super().__init__(np.asarray(position,float),np.asarray(velocity,float),np.asarray(orientation,float),np.asarray(angular_velocity,float))
        self.contact=ContactState()
        # Observer diagnostics only; not a second dynamics state and not checkpointed.
        self.gravity_body_id: str|None = None
        self.gravity_world=np.zeros(3,dtype=np.float64)
        self.slip_angle_deg=0.0
        self.forward_speed=0.0
        # Exact world-physics substep positions from the most recent advance.
        # GUI framerate is never involved.
        self.last_motion_trace=[self.position.copy()]

    @classmethod
    def initial(cls):
        # START_VECTOR1: a newborn embodied fly must not inherit the old synthetic-
        # spacecraft sideways drift/tumble.  The origin-sector bootstrap habitat is
        # deliberately placed straight ahead along body/world +X from the birth point
        # (3,5,7) -> (8,5,7).  Preserve only the tiny original drift magnitude, but
        # align it with body-forward; start with zero angular velocity.  Existing saved
        # specimens never call this path on resume.
        drift_speed=float(np.linalg.norm(np.array([0.030,0.008,-0.004],dtype=np.float64)))
        return cls(
            np.array([3.0,5.0,7.0],dtype=np.float64),
            np.array([drift_speed,0.0,0.0],dtype=np.float64),
            np.array([1.0,0.0,0.0,0.0],dtype=np.float64),
            np.zeros(3,dtype=np.float64),
        )

    @staticmethod
    def _quat_step(orientation: np.ndarray, angular_velocity: np.ndarray, dt: float) -> np.ndarray:
        w=np.asarray(angular_velocity,float); mag=float(np.linalg.norm(w))
        if mag<=0.0: return orientation
        half=.5*mag*float(dt)
        dq=np.array([math.cos(half),*(math.sin(half)*w/mag)],float)
        return q_normalize(q_mul(orientation,dq))

    @classmethod
    def _linear_accel_gain(cls, *, flight: float) -> float:
        mix=float(np.clip(flight,0.0,1.0))
        force_unit=cls.SURFACE_FORCE_UNIT_N + mix*(cls.FLIGHT_FORCE_UNIT_N-cls.SURFACE_FORCE_UNIT_N)
        return float(force_unit/(cls.BODY_MASS_KG*cls.LOCAL_M_PER_UNIT))

    @classmethod
    def _angular_accel(cls, torque_body: np.ndarray) -> np.ndarray:
        torque=np.asarray(torque_body,dtype=float)*cls.TORQUE_UNIT_NM
        return torque/np.maximum(cls.BODY_INERTIA_KGM2,1e-30)

    def _aero_basis(self, gravity_world: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Body-relative aerodynamic axes, independent of the gravity source.

        Gravity is still integrated in world coordinates. Rotating the dissipative
        drag tensor with the body cannot supply energy or turn falling speed into
        thrust. A gravity-relative tangent frame loses pitch/roll information and
        changes the drag law when the dominant habitat switches.
        """
        f=q_rotate(self.orientation,np.array([1.,0.,0.]))
        r=q_rotate(self.orientation,np.array([0.,1.,0.]))
        up=q_rotate(self.orientation,np.array([0.,0.,1.]))
        return f,r,up

    def _set_flight_diagnostics(self, world: LivingUniverse):
        gb,gvec=world.gravity_at(self.position,self.GRAVITY_ACCELERATION)
        self.gravity_body_id=None if gb is None else gb.body_id
        self.gravity_world=np.asarray(gvec,dtype=np.float64)
        f,r,up=self._aero_basis(self.gravity_world)
        vf=float(np.dot(self.velocity,f)); vs=float(np.dot(self.velocity,r)); vu=float(np.dot(self.velocity,up))
        self.forward_speed=vf
        # Signed body-frame sideslip retains backwards flight instead of folding it to zero.
        self.slip_angle_deg=float(math.degrees(math.atan2(vs,vf)))

    def advance_living(self, world: LivingUniverse, force_body: np.ndarray, torque_body: np.ndarray,
                       dt: float, *, flight_command: float, walking_command: float,
                       motor_power_scale: float = 1.0) -> ContactState:
        dt=max(1e-6,float(dt)); self.last_motion_trace=[self.position.copy()]; b,alt,norm=world.nearest_body(self.position)
        power=float(np.clip(float(motor_power_scale),0.0,1.0))
        force_body=np.asarray(force_body,float)*power
        torque_body=np.asarray(torque_body,float)*power
        flight=float(np.clip(float(flight_command),0.0,1.0))
        attached=self.contact.on_surface and self.contact.body_id is not None

        # Surface contact is now a real geometric constraint, not a sticky halo.
        # The physical surface, collision radius and attachment radius are identical.
        # A microscopic epsilon is used only for floating-point comparison; it does
        # not enlarge the body. Contact persists while the surface normal reaction can
        # cancel inward acceleration. If the actual motor-generated outward normal
        # acceleration becomes positive, the fly leaves naturally -- no magic flight
        # command threshold and no host-injected takeoff decision.
        if attached and b is not None and b.body_id==self.contact.body_id:
            n=np.asarray(norm,float)
            # Blend surface and flight force calibration continuously. force_body
            # already contains the anatomical wing/leg mixture; flight only selects
            # how strongly that command couples into free-air mechanics while the feet
            # are still touching the surface.
            contact_gain=self._linear_accel_gain(flight=flight)
            fw=q_rotate(self.orientation,np.asarray(force_body,float))*contact_gain
            gb,gvec=world.gravity_at(self.position,self.GRAVITY_ACCELERATION)
            self.gravity_body_id=None if gb is None else gb.body_id
            self.gravity_world=np.asarray(gvec,dtype=np.float64)
            net_acc=fw+self.gravity_world
            normal_acc=float(np.dot(net_acc,n))
            # Pulvilli/tarsal contact can sustain a finite tensile load.  Previously
            # *any* positive outward normal acceleration detached the fly, which made
            # marginal wing output produce stick -> pop -> recontact -> stick cycles.
            # Use actual leg-motor activity (walking_command) only to approximate how
            # much of the six-foot contact is planted: quiet stance ~= all feet; strong
            # gait ~= alternating-tripod contact.  There is no timer, target, reward,
            # or host-side takeoff decision.
            walk_gate=float(np.clip(float(walking_command)/0.35,0.0,1.0))
            planted_fraction=1.0-walk_gate*(1.0-self.TARSAL_WALK_CONTACT_FRACTION)
            adhesion_acc=self.TARSAL_ADHESION_MAX_ACCEL*planted_fraction
            # Radial velocity is not used as a takeoff trigger: on a curved surface
            # yesterday's tangent becomes slightly outward relative to today's normal
            # after position moves, which otherwise manufactures false hops.  Release
            # only when the *current physical separating load* exceeds adhesion.
            if normal_acc > adhesion_acc:
                attached=False
                self.contact=ContactState()
            else:
                # Normal reaction removes only the inward radial component. Tangential
                # force remains available for walking/sliding; the body cannot sink
                # below or float above the actual surface while contact is active.
                tangent_acc=net_acc-n*float(np.dot(net_acc,n))
                self.velocity += tangent_acc*dt
                self.velocity -= n*float(np.dot(self.velocity,n))
                walk_gate=float(np.clip(float(walking_command)/0.35,0.0,1.0))
                traction=self.SURFACE_STATIC_TRACTION + walk_gate*(self.SURFACE_WALK_TRACTION-self.SURFACE_STATIC_TRACTION)
                self.velocity *= math.exp(-traction*dt)
                self.position += self.velocity*dt
                rel=self.position-b.center; n=rel/max(np.linalg.norm(rel),1e-12)
                self.position=b.center+n*b.radius
                self.last_motion_trace.append(self.position.copy())
                # Re-project against the *new* surface normal after moving around the
                # sphere.  Without this, curvature leaves a small artificial outward
                # velocity that looks like takeoff on the following world tick.
                self.velocity -= n*float(np.dot(self.velocity,n))

                # Grounded turning uses the same calibrated motor-torque scale as free
                # flight, then the same bounded angular drag.  Contact constrains only
                # translation normal to the surface; it must not reduce steering torque
                # by three orders of magnitude.
                # Passive stance reaction: align body +Z with the new local surface
                # normal while preserving free yaw.  Work in body coordinates because
                # angular_velocity/torque_body are body-frame quantities here.
                n_body=q_inverse_rotate(self.orientation,n.reshape(1,3))[0]
                n_body=n_body/max(float(np.linalg.norm(n_body)),1e-12)
                z_body=np.array([0.0,0.0,1.0],dtype=np.float64)
                axis=np.cross(z_body,n_body); sa=float(np.linalg.norm(axis))
                ca=float(np.clip(np.dot(z_body,n_body),-1.0,1.0))
                if sa>1e-10:
                    axis/=sa; angle=math.atan2(sa,ca)
                elif ca<0.0:
                    axis=np.array([1.0,0.0,0.0],dtype=np.float64); angle=math.pi
                else:
                    axis=np.zeros(3,dtype=np.float64); angle=0.0
                stance_acc=axis*(self.SURFACE_STANCE_SPRING*angle)
                stance_acc[:2] -= self.SURFACE_STANCE_DAMPING*self.angular_velocity[:2]
                self.angular_velocity += (self._angular_accel(torque_body) + stance_acc)*dt
                ang_decay=self.ANGULAR_LINEAR_DRAG+self.ANGULAR_QUADRATIC_DRAG*np.abs(self.angular_velocity)
                self.angular_velocity *= np.exp(-ang_decay*dt)
                self.orientation=self._quat_step(self.orientation,self.angular_velocity,dt)

        if not attached:
            # Couple attitude, thrust, gravity and airflow at a finer mechanical step.
            # This matters because Yar can rotate far enough during one world tick that
            # applying drag in only the old body frame visibly preserved the old course.
            nsub=max(1,int(math.ceil(dt/self.FREE_FLIGHT_MAX_SUBSTEP)))
            h=dt/nsub
            for _ in range(nsub):
                # Attitude first: forces and airflow below use the orientation the fly
                # actually has during this sub-step, not the previous world-frame pose.
                self.angular_velocity += self._angular_accel(torque_body)*h
                ang_decay=self.ANGULAR_LINEAR_DRAG+self.ANGULAR_QUADRATIC_DRAG*np.abs(self.angular_velocity)
                self.angular_velocity *= np.exp(-ang_decay*h)
                self.orientation=self._quat_step(self.orientation,self.angular_velocity,h)

                # Ordinary wing/body force is body-relative.  Gravity comes from the
                # strongest local habitat field and falls with inverse-square distance,
                # so sustained outward flight can actually escape a body's near field.
                force_world=q_rotate(self.orientation,force_body)*self._linear_accel_gain(flight=1.0)
                gb,gvec=world.gravity_at(self.position,self.GRAVITY_ACCELERATION)
                self.gravity_body_id=None if gb is None else gb.body_id
                self.gravity_world=np.asarray(gvec,dtype=np.float64)
                self.velocity += (force_world+self.gravity_world)*h

                # Resolve actual airflow along body axes, including pitch and roll.
                # Gravity remains a separate world-space acceleration above.
                fwd,right,up=self._aero_basis(self.gravity_world)
                vcomp=np.array([float(np.dot(self.velocity,fwd)),
                                float(np.dot(self.velocity,right)),
                                float(np.dot(self.velocity,up))],dtype=np.float64)
                # Anisotropic drag only removes kinetic energy. It neither snaps
                # velocity to the nose nor supplies an upright/steering controller.
                damping=np.exp(-(self.LINEAR_DRAG+self.QUADRATIC_DRAG*np.abs(vcomp))*h)
                vcomp *= damping
                self.velocity=fwd*vcomp[0]+right*vcomp[1]+up*vcomp[2]
                self.position += self.velocity*h
                self.last_motion_trace.append(self.position.copy())

                # Collision/contact is checked per mechanical sub-step so gravity or a
                # high-speed bank cannot tunnel through a small habitat body.
                b,alt,norm=world.nearest_body(self.position)
                if b is not None and alt <= self.CONTACT_EPSILON:
                    n=np.asarray(norm,float)
                    inward=float(np.dot(self.velocity,n))
                    if inward<0: self.velocity-=n*inward
                    self.position=b.center+n*b.radius
                    self.last_motion_trace.append(self.position.copy())
                    attached=True
                    break

        b,alt,norm=world.nearest_body(self.position)
        bb,rr,rd=world.nearest_resource(self.position)
        self.contact=ContactState(
            on_surface=bool(attached and b is not None), body_id=b.body_id if attached and b is not None else None,
            surface_temp_c=b.surface_temp_c if attached and b is not None else None,
            normal_world=np.asarray(norm,float) if attached and norm is not None else None,
            altitude=float(alt), altitude_body_id=None if b is None else b.body_id,
            atmosphere_fraction=float(world.atmosphere_fraction(self.position,b)),
            nearest_resource_id=rr.patch_id if rr else None,
            nearest_resource_distance=float(rd),
        )
        self._set_flight_diagnostics(world)
        return self.contact


class LivingThermalSensor(RadiantThermalSensor):
    """Stellar arista sensing plus bounded near-field habitat temperature coupling.

    The base stellar model remains radiometric. Habitat surfaces add a separate
    *near-field* exchange approximation whose strength depends on apparent angular
    area and antenna direction. This gives the animal information about warm versus
    hot/cold surfaces before touchdown without inventing a semantic danger channel.
    The transfer is simulation scaffolding, not claimed fly heat-transfer physiology.
    """

    def observe_living(self, universe: LivingUniverse, craft: FlyBodyState, dt: float) -> ThermalFrame:
        dt=max(float(dt),1e-6)
        frame=super().observe(universe,craft,dt)
        old_l,old_r=self.left_temp_c,self.right_temp_c

        # Nearby bodies exchange heat with the antenna elements according to
        # apparent angular area. Both warm and cool surfaces are representable.
        best_id=None; best_score=0.0; best_bearing=None
        for b in universe.local_bodies(craft.position):
            rel=np.asarray(b.center,float)-np.asarray(craft.position,float)
            dist=float(np.linalg.norm(rel))
            if dist <= 1e-9: continue
            alt=max(0.0,dist-float(b.radius))
            # Solid-angle-like coverage with an additional near-field rolloff.
            coverage=float(np.clip((float(b.radius)/dist)**2,0.0,1.0))*math.exp(-alt/2.5)
            if coverage < 1e-5: continue
            dbody=q_inverse_rotate(craft.orientation,(rel/dist).reshape(1,3))[0]
            li=float(self._incidence(dbody.reshape(1,3),self.left_normal,self.body_shadow)[0])
            ri=float(self._incidence(dbody.reshape(1,3),self.right_normal,self.body_shadow)[0])
            # Coupling is deliberately capped; it changes sensory temperature but
            # cannot create damage or unbounded thermal runaway.
            kl=min(0.85,coverage*li*1.6); kr=min(0.85,coverage*ri*1.6)
            target=float(b.surface_temp_c)
            al=1.0-math.exp(-dt*kl/0.55) if kl>0 else 0.0
            ar=1.0-math.exp(-dt*kr/0.55) if kr>0 else 0.0
            self.left_temp_c += al*(target-self.left_temp_c)
            self.right_temp_c += ar*(target-self.right_temp_c)
            score=coverage*max(li,ri)*abs(target-self.body_temp_c)
            if score>best_score:
                best_score=score; best_id=b.body_id
                best_bearing=math.degrees(math.atan2(float(dbody[1]),float(dbody[0])))

        # Surface contact is the strongest local thermal context.
        if craft.contact.on_surface and craft.contact.surface_temp_c is not None:
            amb=float(craft.contact.surface_temp_c); a=1.0-math.exp(-dt/0.35)
            self.left_temp_c += a*(amb-self.left_temp_c); self.right_temp_c += a*(amb-self.right_temp_c)
            best_id=craft.contact.body_id; best_bearing=0.0; best_score=max(best_score,1.0)

        dl=(self.left_temp_c-old_l)/dt; dr=(self.right_temp_c-old_r)/dt
        return ThermalFrame(float(self.left_temp_c),float(self.right_temp_c),float(dl),float(dr),
                            frame.left_flux_w_m2,frame.right_flux_w_m2,frame.left_ir_fraction,frame.right_ir_fraction,
                            best_id if best_id is not None else frame.dominant_thermal_id,
                            float(best_bearing) if best_bearing is not None else frame.dominant_bearing_deg,
                            max(float(frame.dominant_flux_w_m2),float(best_score)))
