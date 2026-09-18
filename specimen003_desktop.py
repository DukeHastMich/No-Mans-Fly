#!/usr/bin/env python3
"""No Man's Fly — Specimen-003 Life Observatory / long-run controller.

Observer/controller only.  The GUI never supplies steering or reward commands.  It
shows what the connectome, body, homeostatic state, and world are already doing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import queue
import shutil
import struct
import sys
import threading
import time
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from world_visual import surface_rgb
from observer_delivery import SnapshotQueue

from specimen003_experiment import Specimen003Experiment
from seed_store import ensure_default_seed, read_seed, save_numbered_seed
from compound_eye import CompoundEyeRenderer
from full_connectome_evolution import (
    GrowthPressureTracker, load_full_growth_parent, seed_duplicate_neuron_full, freeze_full_baseline,
)
from space_fly_universe import q_inverse_rotate, q_rotate, q_mul, q_normalize

ROOT=Path(__file__).resolve().parent
BASE_GRAPH=ROOT/'data'/'male-cns-v1.0-full-graph-v2.npz'
RUNS_ROOT=ROOT/'Runs'
SEED_DIR=RUNS_ROOT/'Specimen-Seed'
LIVE_DIR=RUNS_ROOT/'Specimen-003-Live'
BUILD_ID='Patch23A GE2 + Organism Health 1 + Turn Bias 1 + Feeding Path 1 + New Fly 1 + Body View 3 + Life Depth 1'



class SpecimenAlreadyRunningError(RuntimeError):
    pass


class _SpecimenProcessLock:
    """OS-held single-writer lock for one live specimen directory.

    The lock is advisory but process-scoped: GUI and headless runners both acquire
    it before loading checkpoint state.  The operating system releases ownership
    automatically if a process crashes or is killed, so a stale lock file cannot
    by itself strand a specimen.
    """
    def __init__(self, live_dir: Path):
        self.path=Path(live_dir)/'.no_mans_fly.writer.lock'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self._fh=open(self.path,'a+b',buffering=0)
        self._kind=None
        try:
            self._fh.seek(0,os.SEEK_END)
            if self._fh.tell()==0:
                self._fh.write(b'0')
            self._fh.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self._fh.fileno(),msvcrt.LK_NBLCK,1)
                self._kind='windows'
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                self._kind='posix'
        except (OSError,IOError) as exc:
            owner='unknown owner'
            try:
                self._fh.seek(1); raw=self._fh.read().decode('utf-8','replace').strip()
                if raw:
                    info=json.loads(raw); owner=f"PID {info.get('pid','?')} • {info.get('mode','runner')} • {info.get('command','')}"
            except Exception:
                pass
            try:self._fh.close()
            except Exception:pass
            raise SpecimenAlreadyRunningError(
                f'{live_dir} is already open for writing by {owner}.\n'
                'Only one No Man\'s Fly GUI/headless runner may own a live specimen at a time. '
                'Stop the other runner before continuing.') from exc
        mode='headless' if '--headless' in sys.argv else 'gui'
        info={'pid':os.getpid(),'mode':mode,'started_unix':time.time(),'command':' '.join(sys.argv)}
        payload=json.dumps(info,separators=(',',':')).encode('utf-8')
        self._fh.seek(1); self._fh.truncate(); self._fh.write(payload); self._fh.flush()

    def release(self):
        fh=self._fh
        if fh is None:return
        try:
            fh.seek(0)
            if self._kind=='windows':
                import msvcrt
                msvcrt.locking(fh.fileno(),msvcrt.LK_UNLCK,1)
            elif self._kind=='posix':
                import fcntl
                fcntl.flock(fh.fileno(),fcntl.LOCK_UN)
        except Exception:
            pass
        try:fh.close()
        except Exception:pass
        self._fh=None

    def __del__(self):
        self.release()

GROWABLE_SUPERCLASSES={
    'cb_intrinsic','ol_intrinsic','vnc_intrinsic','ascending_neuron','descending_neuron',
    'visual_projection','visual_projection_tbc','visual_centrifugal',
}


def _rel(path:Path)->str:
    try:return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:return str(path.resolve())

def _resolve(text:str)->Path:
    s=str(text).replace('\\',os.sep).replace('/',os.sep); p=Path(s)
    return p if p.is_absolute() else ROOT/p

def _blackbody_rgb(kelvin:float)->str:
    t=max(1000.,min(40000.,float(kelvin)))/100.
    if t<=66:
        r=255; g=99.4708025861*math.log(t)-161.1195681661
        b=0 if t<=19 else 138.5177312231*math.log(t-10)-305.0447927307
    else:
        r=329.698727446*((t-60)**-0.1332047592)
        g=288.1221695283*((t-60)**-0.0755148492); b=255
    r,g,b=[max(0,min(255,int(x))) for x in (r,g,b)]
    return f'#{r:02x}{g:02x}{b:02x}'


FOOD_ICONS=("🍎","🍌","🍓","🍇","🍒","🍑","🍐","🍊","🍋","🍉","🥭","🍍","💩")

def _stable_digest(text:str)->bytes:
    return hashlib.sha256(str(text).encode("utf-8")).digest()

def _resource_icon(resource_id:str)->str:
    if "-DIVINE-FRUIT-" in str(resource_id): return "🥥"
    d=_stable_digest(resource_id)
    return FOOD_ICONS[int.from_bytes(d[:2],"little") % len(FOOD_ICONS)]

def _body_render_kind(body_id:str)->str:
    # Observer-only appearance. Habitat physics does not distinguish planet/moon yet.
    return "moon" if _stable_digest(body_id)[2] < 72 else "planet"

def _rgba_png_bytes(rgb:np.ndarray,mask:np.ndarray)->bytes:
    """Dependency-free RGBA PNG encoder for Tk observer sprites.

    Life/world sprites need real alpha so their square backing tiles cannot occlude
    another world.  This is display-only and has no path into the simulation.
    """
    a=np.asarray(rgb,dtype=np.uint8)
    m=np.asarray(mask,dtype=bool)
    h,w=a.shape[:2]
    rgba=np.empty((h,w,4),dtype=np.uint8)
    rgba[:,:,:3]=a
    rgba[:,:,3]=np.where(m,255,0).astype(np.uint8)
    raw=b''.join(b'\x00'+np.ascontiguousarray(rgba[y]).tobytes() for y in range(h))
    def chunk(tag:bytes,data:bytes)->bytes:
        return struct.pack('>I',len(data))+tag+data+struct.pack('>I',zlib.crc32(tag+data)&0xffffffff)
    return (b'\x89PNG\r\n\x1a\n'
            +chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,6,0,0,0))
            +chunk(b'IDAT',zlib.compress(raw,6))
            +chunk(b'IEND',b''))

def _planet_sprite_array(body_id:str, diameter:int, kind:str="planet", *,
                         albedo=.5, surface_basis=None)->tuple[np.ndarray,np.ndarray]:
    """Small orthographic sphere using the same spectral surface as retinal hits."""
    n=max(4,int(diameter));yy,xx=np.mgrid[0:n,0:n].astype(np.float32)
    c=(n-1)*.5;r=max(1.,c);nx=(xx-c)/r;ny=-(yy-c)/r
    rr=nx*nx+ny*ny;mask=rr<=1.;nz=np.sqrt(np.clip(1.-rr,0.,1.))
    basis=np.eye(3) if surface_basis is None else np.asarray(surface_basis,float)
    normals=np.column_stack((nx[mask],ny[mask],nz[mask]))@basis.T
    rgb=np.zeros((n,n,3),dtype=np.uint8)
    rgb[mask]=surface_rgb(body_id,normals,albedo,kind)
    return rgb,mask

# Observer-only equirectangular ray cache.  The old display placed each habitat as
# one centre-point sprite.  When Yar stood on a sphere, its centre lay almost exactly
# at body-frame -Z, where azimuth is undefined; tiny floating-point changes could make
# atan2() jump by ~180 degrees and the whole world appeared to hop left/right.  These
# rays let the human display render the actual sphere/horizon instead.  Nothing here is
# supplied to the compound eye or connectome.
_OBSERVER_SKY_RAY_CACHE:dict[tuple[int,int],np.ndarray]={}

def _observer_sky_rays(width:int,height:int)->np.ndarray:
    key=(max(2,int(width)),max(2,int(height)))
    q=_OBSERVER_SKY_RAY_CACHE.get(key)
    if q is not None:return q
    w,h=key
    yaw=np.linspace(-math.pi,math.pi,w,dtype=np.float64)
    elev=np.linspace(math.pi*.5,-math.pi*.5,h,dtype=np.float64)
    ce=np.cos(elev)[:,None]
    q=np.stack((ce*np.cos(yaw)[None,:],ce*np.sin(yaw)[None,:],
                np.broadcast_to(np.sin(elev)[:,None],(h,w))),axis=-1).reshape((-1,3))
    _OBSERVER_SKY_RAY_CACHE[key]=q
    # GUI resizes are rare, but do not let an interactive resize session retain an
    # unbounded pile of multi-megabyte ray grids.
    if len(_OBSERVER_SKY_RAY_CACHE)>4:
        for k in list(_OBSERVER_SKY_RAY_CACHE)[:-4]:_OBSERVER_SKY_RAY_CACHE.pop(k,None)
    return q

def _observer_surface_rgb(body_id, normals, kind='planet', albedo=.5):
    return surface_rgb(body_id,normals,albedo,kind)


def _observer_raycast_sphere(width:int,height:int,camera_world:np.ndarray,orientation:np.ndarray,
                             center_world:np.ndarray,radius:float,body_id:str,kind:str='planet',
                             delta_q:np.ndarray|None=None,basis_world:np.ndarray|None=None,albedo:float=.5):
    """Return (mask, depth, rgb) for one observer-only habitat sphere."""
    w=max(2,int(width));h=max(2,int(height));rays=_observer_sky_rays(w,h)
    if basis_world is not None:
        rays_world=rays@np.asarray(basis_world,dtype=np.float64).T
    else:
        if delta_q is not None:rays=q_rotate(np.asarray(delta_q,float),rays)
        rays_world=q_rotate(np.asarray(orientation,float),rays)
    cam=np.asarray(camera_world,dtype=np.float64);ctr=np.asarray(center_world,dtype=np.float64)
    oc=cam-ctr;r=max(1e-9,float(radius));b=rays_world@oc;c=float(oc@oc-r*r)
    disc=b*b-c;root=np.sqrt(np.clip(disc,0.,None));t0=-b-root;t1=-b+root
    eps=max(1e-9,r*1e-8);t=np.where(t0>eps,t0,np.where(t1>eps,t1,np.inf))
    valid=(disc>=0.)&np.isfinite(t)
    rgb=np.zeros((len(t),3),dtype=np.uint8)
    if np.any(valid):
        hit=cam[None,:]+rays_world[valid]*t[valid,None]
        normal=(hit-ctr[None,:])/r
        rgb[valid]=_observer_surface_rgb(body_id,normal,kind,albedo)
    return valid.reshape((h,w)),t.reshape((h,w)),rgb.reshape((h,w,3))

def _observer_level_basis_world(orientation:np.ndarray, surface_normal_world:np.ndarray|None=None)->np.ndarray:
    """Columns are display forward/right/up in world coordinates.

    In free flight this is the actual body frame. While grounded the human observer
    is gravity-levelled around Yar's real heading. This removes the sinusoidal horizon
    that an equirectangular 360° projection produces when a great-circle horizon is
    viewed with body roll/pitch. The fused retinal view below remains unstabilized.
    """
    q=q_normalize(np.asarray(orientation,dtype=np.float64))
    f=q_rotate(q,np.array([1.,0.,0.]));r=q_rotate(q,np.array([0.,1.,0.]));u=q_rotate(q,np.array([0.,0.,1.]))
    if surface_normal_world is None:
        return np.column_stack((f,r,u))
    up=np.asarray(surface_normal_world,dtype=np.float64);nu=float(np.linalg.norm(up))
    if nu<=1e-12:return np.column_stack((f,r,u))
    up/=nu;ft=f-up*float(np.dot(f,up));nf=float(np.linalg.norm(ft))
    if nf<=1e-8:
        # Degenerate only if the body forward axis points almost straight up/down.
        # Recover a tangent heading from body-right without inventing world north.
        rt=r-up*float(np.dot(r,up));nr=float(np.linalg.norm(rt))
        if nr<=1e-8:return np.column_stack((f,r,u))
        right=rt/nr;ft=np.cross(right,up);nf=float(np.linalg.norm(ft))
    forward=ft/max(nf,1e-12);right=np.cross(up,forward);right/=max(float(np.linalg.norm(right)),1e-12)
    return np.column_stack((forward,right,up))

def _observer_project_world(directions_world:np.ndarray,basis_world:np.ndarray):
    v=np.asarray(directions_world,dtype=np.float64).reshape((-1,3))@np.asarray(basis_world,dtype=np.float64)
    yaw=np.degrees(np.arctan2(v[:,1],v[:,0]));elev=np.degrees(np.arctan2(v[:,2],np.hypot(v[:,0],v[:,1])))
    return yaw,elev

def _observer_surface_extrapolate(position_world:np.ndarray, velocity_world:np.ndarray,
                                  center_world:np.ndarray, radius:float, dt_s:float):
    """Observer-only interpolation along an attached spherical surface.

    Linear xyz extrapolation is wrong for a body constrained to a sphere: a tangent
    velocity immediately puts the predicted camera above the tangent plane while the
    display's `up` vector still points along the old surface normal.  On a small world
    that mismatch can be several degrees in one 0.1-s visual epoch, bending an otherwise
    level horizon into the cyclic wave seen in Patches 21/22.

    Advance the contact point by the exponential map of the tangential velocity instead.
    This never touches simulation state; it is only a smoother between authoritative
    snapshots. Returns (surface_position, surface_normal).
    """
    p=np.asarray(position_world,dtype=np.float64);v=np.asarray(velocity_world,dtype=np.float64)
    c=np.asarray(center_world,dtype=np.float64);r=max(1e-9,float(radius));dt=max(0.,float(dt_s))
    rel=p-c;nr=float(np.linalg.norm(rel))
    if nr<=1e-12:
        return p.copy(),np.array([0.,0.,1.],dtype=np.float64)
    n0=rel/nr
    if dt<=0.:return c+n0*r,n0
    vt=v-n0*float(np.dot(v,n0));speed=float(np.linalg.norm(vt))
    if speed<=1e-12:return c+n0*r,n0
    axis=np.cross(n0,vt);na=float(np.linalg.norm(axis))
    if na<=1e-12:return c+n0*r,n0
    axis/=na;ang=(speed/r)*dt
    # Rodrigues rotation of n0 around the great-circle axis.
    ca=math.cos(ang);sa=math.sin(ang)
    n1=n0*ca+np.cross(axis,n0)*sa+axis*float(np.dot(axis,n0))*(1.-ca)
    n1/=max(float(np.linalg.norm(n1)),1e-12)
    return c+n1*r,n1

def _eye_color(v:float)->str:
    x=max(0.,min(1.,float(v)))
    return f'#{int(18+220*x**1.8):02x}{int(12+225*x**1.25):02x}{int(28+227*x**.65):02x}'

def _box_blur2d(a:np.ndarray, passes:int=1)->np.ndarray:
    """Tiny dependency-free blur used only by the observer rasterizer."""
    out=np.asarray(a,dtype=np.float32)
    for _ in range(max(0,int(passes))):
        q=np.pad(out,1,mode='edge')
        out=(q[:-2,:-2]+q[:-2,1:-1]+q[:-2,2:]+
             q[1:-1,:-2]+q[1:-1,1:-1]+q[1:-1,2:]+
             q[2:,:-2]+q[2:,1:-1]+q[2:,2:])*(1.0/9.0)
    return out

def _eye_raster(width:int,height:int,xpix:np.ndarray,ypix:np.ndarray,response:np.ndarray,
                confidence:np.ndarray|None=None,*,blur_passes:int=1)->np.ndarray:
    """Rasterize thousands of facet samples without thousands of Tk canvas items.

    This is observer-only presentation. It does not alter the optical drive delivered
    to the connectome. Confidence controls display opacity, not neural input.
    """
    w=max(2,int(width));h=max(2,int(height)); bg=np.array([3,2,10],dtype=np.float32)
    acc=np.zeros((h,w),dtype=np.float32); cov=np.zeros((h,w),dtype=np.float32)
    if len(response):
        x=np.clip(np.asarray(xpix,dtype=np.int32),0,w-1);y=np.clip(np.asarray(ypix,dtype=np.int32),0,h-1)
        r=np.clip(np.asarray(response,dtype=np.float32),0.,1.)
        c=np.ones(len(r),dtype=np.float32) if confidence is None else np.clip(np.asarray(confidence,dtype=np.float32),0.05,1.)
        # Five-point splat is enough to keep individual ommatidia visible while the
        # subsequent small blur turns the fused field into the intended fuzzy angular map.
        for dx,dy,wt in ((0,0,1.0),(-1,0,.55),(1,0,.55),(0,-1,.55),(0,1,.55)):
            xx=np.clip(x+dx,0,w-1);yy=np.clip(y+dy,0,h-1); ww=c*wt
            np.add.at(acc,(yy,xx),r*ww);np.add.at(cov,(yy,xx),ww)
    acc=_box_blur2d(acc,blur_passes);cov=_box_blur2d(cov,blur_passes)
    val=np.divide(acc,cov,out=np.zeros_like(acc),where=cov>1e-7);val=np.clip(val,0.,1.)
    # Coverage is an opacity/confidence cue. Sparse peripheral samples remain fuzzy/dim.
    alpha=np.clip(cov*1.65,0.,1.)[...,None]
    rgb=np.empty((h,w,3),dtype=np.float32)
    rgb[...,0]=18.+220.*np.power(val,1.8)
    rgb[...,1]=12.+225.*np.power(val,1.25)
    rgb[...,2]=28.+227.*np.power(val,.65)
    rgb=bg[None,None,:]*(1.-alpha)+rgb*alpha
    return np.clip(rgb,0,255).astype(np.uint8)

def _viewer_delta_quaternion(angular_velocity, dt_s:float)->np.ndarray:
    """Observer-only attitude extrapolation; never feeds the specimen."""
    w=np.asarray(angular_velocity,dtype=np.float64); mag=float(np.linalg.norm(w))
    if mag <= 1e-12 or dt_s <= 0:
        return np.array([1.,0.,0.,0.],dtype=np.float64)
    half=.5*mag*float(dt_s); return np.array([math.cos(half),*(math.sin(half)*w/mag)],dtype=np.float64)


class LifeLineageEngine:
    def __init__(self, base_graph:Path=BASE_GRAPH, live_dir:Path=LIVE_DIR, seed_dir:Path=SEED_DIR, *, reset_live=False, specimen_label:str|None=None, defer_tuning:bool=False):
        self.base_graph=Path(base_graph); self.live_dir=Path(live_dir); self.seed_dir=Path(seed_dir)
        inferred=self.live_dir.name[:-5] if self.live_dir.name.endswith('-Live') else self.live_dir.name
        self.specimen_label=str(specimen_label or inferred or 'Specimen-003')
        self.defer_tuning=bool(defer_tuning)
        self._process_lock=_SpecimenProcessLock(self.live_dir)
        self.generations_dir=self.live_dir/'Generations'
        # Patch23A: future frozen W-generation bundles are specimen-local so one
        # specimen cannot overwrite/archive dependencies belonging to another.
        # Existing legacy global archives are left untouched.
        self.generations_archive_dir=self.live_dir/'Generations - Archived'
        self.manifest_path=self.live_dir/'live_lineage.json'
        self.growth_events_path=self.live_dir/'growth_events.jsonl'
        self.lock=threading.RLock()
        # A frozen W-generation is a complete runtime graph.  If a valid live lineage
        # already exists, resume it directly; the original MaleCNS baseline is needed
        # only to create/reset a fresh specimen, not to continue an existing one.
        existing_live=(self.manifest_path.exists() and self._checkpoint_complete(self.live_dir))
        fresh_exp=None
        if existing_live and not reset_live:
            self.generations_dir.mkdir(parents=True,exist_ok=True)
        else:
            fresh_exp=self._create_fresh_lineage()
        self.manifest=json.loads(self.manifest_path.read_text(encoding='utf-8'))
        self.specimen_label=str(self.manifest.get('specimen') or self.specimen_label)
        self.seed_dir=_resolve(self.manifest.get('seed_checkpoint',_rel(self.seed_dir)))
        self.generation=int(self.manifest.get('working_generation',0))
        self.current_graph=_resolve(self.manifest.get('current_graph',_rel(self.base_graph)))
        if not self.current_graph.exists(): raise FileNotFoundError(self.current_graph)
        self.exp=(fresh_exp if fresh_exp is not None else
                  Specimen003Experiment.resume(self.current_graph,self.live_dir,auto_tune_threads=not self.defer_tuning))
        self.thread_spec='auto'
        if not self.defer_tuning:self.exp.core.configure_threads(self.thread_spec)
        self.last_save_sim_s=float(self.exp.time_s)
        self.last_status=f'Loaded {self.specimen_label} live checkpoint'
        _rep=[]
        _gr=getattr(self.exp.core,'gain_repair_report',{}) or {}
        _fr=getattr(self.exp.core,'fast_state_repair_report',{}) or {}
        if _gr.get('applied'):
            _rep.append(f"repaired {int(_gr.get('repaired_edges',0)):,} duplicated frozen gains -> structural 1.0")
        if _fr.get('repaired'):
            _rep.append('invalid transient neural state cleared to physiological rest')
        if bool(getattr(self.exp.growth_pressure,'legacy_pressure_invalidated',False)):
            _rep.append('legacy structural-growth pressure queue invalidated; fresh post-topology revalidation required')
        if _rep:self.last_status += ' | Patch23: ' + '; '.join(_rep)
        self.last_growth_event=None
        self._meta=self._load_meta()
        self._ledger=[]; self._refresh_ledger()
        self.trajectory=[]

    @staticmethod
    def _checkpoint_complete(folder:Path)->bool:
        folder=Path(folder)
        required=(
            'specimen003_state.json','specimen003_memory.npz','specimen003_fast_state.npz',
            'compound_eye_map.npz','growth_pressure.npz','homeostasis.json','living_universe.json',
        )
        if not all((folder/name).is_file() for name in required):
            return False
        try:
            state=json.loads((folder/'specimen003_state.json').read_text(encoding='utf-8'))
            files=dict(state.get('files') or {})
            if state.get('format') != Specimen003Experiment.FORMAT:
                return False
            for key in ('memory','fast','eye','growth','homeostasis','universe'):
                rel=files.get(key)
                if not rel or not (folder/rel).is_file():
                    return False
            return True
        except Exception:
            return False

    def _write_fresh_checkpoint(self,exp:Specimen003Experiment,target:Path,readme:bool=False,*,preserve_lock:bool=False):
        target=Path(target)
        if target.resolve() != self.live_dir.resolve():
            raise ValueError('Checkpoint writer may only update the live folder, never a seed')
        tmp=target.with_name(target.name+'.creating')
        if tmp.exists(): shutil.rmtree(tmp)
        tmp.mkdir(parents=True,exist_ok=True)
        exp.save(tmp)
        if readme:
            (tmp/'README.txt').write_text(
                'Immutable fresh No Man\'s Fly seed. No experience, no structural growth.\n',encoding='utf-8')
        if not self._checkpoint_complete(tmp):
            raise RuntimeError(f'fresh checkpoint verification failed: {tmp}')
        if preserve_lock:
            # The live directory already contains the OS-held writer lock.  Never unlink
            # that file while its handle is locked (especially on Windows).  Replace all
            # other newborn checkpoint contents from the verified staging directory.
            target.mkdir(parents=True,exist_ok=True)
            for child in list(target.iterdir()):
                if child.name=='.no_mans_fly.writer.lock':
                    continue
                if child.is_dir(): shutil.rmtree(child)
                else: child.unlink()
            for src in tmp.iterdir():
                dst=target/src.name
                if src.is_dir(): shutil.copytree(src,dst)
                else: shutil.copy2(src,dst)
            shutil.rmtree(tmp)
        else:
            if target.exists(): shutil.rmtree(target)
            tmp.replace(target)

    def _create_fresh_lineage(self)->Specimen003Experiment:
        generation=0
        if self.seed_dir.resolve() == SEED_DIR.resolve():
            graph=ensure_default_seed(self.base_graph, runs=RUNS_ROOT)
            # Pristine births use construction, not legacy checkpoint migration.
            exp=Specimen003Experiment(graph,auto_tune_threads=False)
        else:
            if (self.seed_dir/'seed.json').is_file():
                info,graph=read_seed(self.seed_dir)
                if info.get('kind') != 'custom':
                    raise ValueError('Expected a custom seed snapshot')
                generation=int(info.get('working_generation',0))
            else:
                # Existing pre-seed-store lineages may still name their old seed.
                graph=self.base_graph
            if not self._checkpoint_complete(self.seed_dir):
                raise ValueError(f'Seed checkpoint is incomplete: {self.seed_dir}')
            exp=Specimen003Experiment.resume(graph,self.seed_dir,auto_tune_threads=False)
        self._write_fresh_checkpoint(exp,self.live_dir,preserve_lock=True)
        self.generations_dir=self.live_dir/'Generations'
        self.generations_dir.mkdir(parents=True,exist_ok=True)
        self.manifest_path=self.live_dir/'live_lineage.json'
        self.growth_events_path=self.live_dir/'growth_events.jsonl'
        self.manifest_path.write_text(json.dumps({
            'format':'no-mans-fly-specimen003-lineage-v1','specimen':self.specimen_label,
            'working_generation':generation,'current_graph':_rel(graph),
            'seed_checkpoint':_rel(self.seed_dir),'canonical_promotion':None,
            'sim_time_s':float(exp.time_s),'step_count':int(exp.step_count),
        },indent=2),encoding='utf-8')
        return exp

    def save_as_seed(self):
        with self.lock:
            target=save_numbered_seed(RUNS_ROOT,self.current_graph,self.exp.save,
                                     self._checkpoint_complete,specimen=self.specimen_label,
                                     generation=self.generation)
            self.last_status=f'Saved reusable seed: {target.name}'
            return target

    def _load_meta(self):
        z=np.load(self.current_graph,allow_pickle=True)
        return {k:np.asarray(z[k]) for k in ('bodies','types','instance','superclass','subclass','side') if k in z}

    def _write_manifest(self):
        self.manifest.update({
            'format':'no-mans-fly-specimen003-lineage-v1','specimen':self.specimen_label,
            'working_generation':int(self.generation),'current_graph':_rel(self.current_graph),
            'sim_time_s':float(self.exp.time_s),'step_count':int(self.exp.step_count),
        })
        self.manifest_path.write_text(json.dumps(self.manifest,indent=2),encoding='utf-8')

    def _refresh_ledger(self):
        rows=[]; bodies=np.asarray(self._meta.get('bodies',[]),dtype=np.int64)
        b2i={int(b):i for i,b in enumerate(bodies)}
        if self.growth_events_path.exists():
            for line in self.growth_events_path.read_text(encoding='utf-8').splitlines():
                try:r=json.loads(line)
                except Exception:continue
                r['parent_index_now']=b2i.get(int(r.get('parent_body',0)))
                r['daughter_index_now']=b2i.get(int(r.get('daughter_body',0)))
                rows.append(r)
        self._ledger=rows

    @staticmethod
    def _file_sha256(path:Path)->str:
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()

    @classmethod
    def _files_identical(cls,a:Path,b:Path)->bool:
        """Byte-exact comparison used only to avoid duplicate archive copies."""
        try:
            return a.stat().st_size == b.stat().st_size and cls._file_sha256(a)==cls._file_sha256(b)
        except OSError:
            return False

    @staticmethod
    def _npz_is_stored(path:Path)->bool:
        try:
            with zipfile.ZipFile(path,'r') as zf:
                infos=[x for x in zf.infolist() if not x.is_dir()]
                return bool(infos) and all(x.compress_type==zipfile.ZIP_STORED for x in infos)
        except Exception:return False

    def _schedule_archive_recompression(self,path:Path):
        """Recompress a moved historical generation outside the simulation lock.

        The archived ZIP_STORED graph remains the canonical valid file until a fully
        written/CRC-verified DEFLATE temp archive is atomically swapped into place.
        A crash or process exit during this daemon job therefore leaves a readable
        generation, merely larger than necessary.
        """
        path=Path(path)
        if not path.exists() or not self._npz_is_stored(path):return None
        log_path=self.generations_archive_dir/'archive_recompression.jsonl'
        def job():
            tmp=path.with_name(path.name+'.recompress.tmp')
            rec={'time_wall':time.time(),'graph':path.name,'status':'started'}
            try:
                with zipfile.ZipFile(path,'r') as zin:
                    src_meta={i.filename:(int(i.CRC),int(i.file_size)) for i in zin.infolist() if not i.is_dir()}
                    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as zout:
                        for info in zin.infolist():
                            if info.is_dir():continue
                            with zin.open(info,'r') as fi, zout.open(info.filename,'w',force_zip64=True) as fo:
                                shutil.copyfileobj(fi,fo,length=8*1024*1024)
                with zipfile.ZipFile(tmp,'r') as zf:
                    bad=zf.testzip()
                    dst_meta={i.filename:(int(i.CRC),int(i.file_size)) for i in zf.infolist() if not i.is_dir()}
                if bad is not None or dst_meta!=src_meta:
                    raise IOError(f'archive recompression verification failed at {bad!r}')
                os.replace(tmp,path)
                rec.update(status='completed',bytes=int(path.stat().st_size))
            except Exception as exc:
                rec.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            finally:
                try:
                    if tmp.exists():tmp.unlink()
                except Exception:pass
                try:
                    with log_path.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,sort_keys=True)+'\\n')
                except Exception:pass
        t=threading.Thread(target=job,name=f'archive-compress-{path.stem}',daemon=True);t.start();return t

    def _archive_generation_bundle(self,generation:int,graph_path:Path):
        """Move a completed *previous* W-generation out of the active Generations folder.

        Archiving is deliberately post-commit: the caller invokes this only after the
        replacement generation and live checkpoint have saved successfully.  W000 is
        the immutable baseline and is never moved.  The archived bundle includes the
        frozen graph, its freeze manifest, and that generation's growth provenance files.
        """
        generation=int(generation)
        graph_path=Path(graph_path)
        if generation <= 0:return [],[]
        try:
            active_dir=self.generations_dir.resolve()
            if graph_path.resolve().parent != active_dir:return [],[]
        except OSError:
            return [],[f'could not resolve W{generation:03d} archive paths']

        tag=f'W{generation:03d}'
        bundle=[
            graph_path,
            Path(str(graph_path)+'.manifest.json'),
            self.generations_dir/f'{tag}-growth-overlay.json',
            self.generations_dir/f'{tag}-pre-growth-memory.npz',
        ]
        sources=[x for x in bundle if x.exists()]
        if not sources:return [],[]
        self.generations_archive_dir.mkdir(parents=True,exist_ok=True)
        moved=[]; errors=[]
        for src in sources:
            dst=self.generations_archive_dir/src.name
            try:
                if dst.exists():
                    if self._files_identical(src,dst):
                        src.unlink()
                        moved.append(src.name)
                        continue
                    # Never overwrite a differing historical artifact.  Preserve both.
                    h=self._file_sha256(src)[:12]
                    if src.name.endswith('.npz.manifest.json'):
                        stem=src.name[:-len('.npz.manifest.json')]
                        dst=self.generations_archive_dir/f'{stem}-conflict-{h}.npz.manifest.json'
                    else:
                        dst=self.generations_archive_dir/f'{src.stem}-conflict-{h}{src.suffix}'
                    if dst.exists() and self._files_identical(src,dst):
                        src.unlink(); moved.append(src.name); continue
                shutil.move(str(src),str(dst)); moved.append(src.name)
            except OSError as exc:
                errors.append(f'{src.name}: {exc}')
        return moved,errors

    def configure_threads(self, value='auto', *, force=False, progress=None):
        with self.lock:
            text=str(value).strip().lower() or 'auto'
            self.thread_spec=text
            return self.exp.core.configure_threads(text,force=force,progress=progress)

    def configure_gpu(self, requested=True, *, progress=None):
        with self.lock:
            return self.exp.core.probe_gpu_hybrid(requested=bool(requested),progress=progress)

    def apply_runtime_acceleration(self):
        with self.lock:return self.exp.core.apply_runtime_acceleration()

    def save(self):
        with self.lock:
            self.exp.save(self.live_dir); self._write_manifest()
            self.last_save_sim_s=float(self.exp.time_s); self.last_status=f'Saved at t={self.exp.time_s:.2f}s W{self.generation:03d}'

    def world_alias(self,body_id:str)->str:
        with self.lock:return self.exp.universe.world_alias(str(body_id))

    def atlas_snapshot(self):
        with self.lock:
            data=self.exp.universe.atlas_snapshot()
            data['body_trajectory']=[list(p) for p in self.trajectory]
            return data

    def _persist_universe_only(self):
        up=self.live_dir/'living_universe.json'; tmp=up.with_suffix(up.suffix+'.tmp')
        tmp.write_text(json.dumps(self.exp.universe.snapshot(),indent=2),encoding='utf-8');os.replace(tmp,up)





    def set_world_alias(self,body_id:str,alias:str|None):
        with self.lock:
            bid=str(body_id); body=self.exp.universe.body_by_id(bid)
            name=self.exp.universe.set_world_alias(bid,alias,body=body,time_s=float(self.exp.time_s))
            # Alias/atlas metadata belongs to living_universe.json and is cheap enough to
            # persist immediately without rewriting the 25M-edge neural checkpoint.
            self._persist_universe_only()
            self.last_status=f"World alias {'set to '+name if name else 'cleared'} for {bid}"
            return name,self.snapshot()

    def release_process_lock(self):
        lock=getattr(self,'_process_lock',None)
        if lock is not None:
            lock.release(); self._process_lock=None

    def growable_candidate(self):
        cands=list(self.exp.last_growth_candidates or [])
        sc=np.asarray(self.exp.core.superclass,dtype=str)
        for c in cands:
            i=int(c['index'])
            if 0<=i<len(sc) and (sc[i] in GROWABLE_SUPERCLASSES or
                (c.get('kind')=='connection_recruitment' and 'sensory' in sc[i])):return c
        return None

    @staticmethod
    def _clone_contact(c):
        from living_universe import ContactState
        return ContactState(
            on_surface=bool(c.on_surface),body_id=c.body_id,surface_temp_c=c.surface_temp_c,
            normal_world=None if c.normal_world is None else np.array(c.normal_world,dtype=float,copy=True),
            altitude=float(c.altitude),altitude_body_id=c.altitude_body_id,
            atmosphere_fraction=float(c.atmosphere_fraction),nearest_resource_id=c.nearest_resource_id,
            nearest_resource_distance=float(c.nearest_resource_distance))

    def _migrate_after_growth(self,new_graph:Path,parent:int):
        old=self.exp; oldcore=old.core
        # Reuse the already-resolved eye calibration. Structural growth appends a
        # neuron and cannot legitimately renumber/recalibrate existing photoreceptors.
        # A shallow copy is sufficient because resolved mapping arrays are immutable.
        preserved_eye=copy.copy(old.eye_map)
        neo=Specimen003Experiment(new_graph,universe_seed=old.universe.seed,auto_tune_threads=False,
                                  resolved_eye_map=preserved_eye)
        neo.core.inherit_acceleration_from(oldcore)
        n=oldcore.n
        if neo.core.n!=n+1: raise RuntimeError('growth rebase did not append exactly one neuron')
        if neo.eye_mapping_digest()!=old.eye_mapping_digest():
            raise RuntimeError('internal growth unexpectedly changed eye mapping')
        # Preserve bounded learned memory without re-baking it into edge_gain.
        # The incremental freezer copies untouched source ranges verbatim and only
        # merges source blocks that receive new edges.  Migrate learned deltas in the
        # same sparse ranges instead of looping through all ~167k source neurons in
        # Python while holding the engine lock.
        old_ip=np.asarray(oldcore.indptr,dtype=np.int64); new_ip=np.asarray(neo.core.indptr,dtype=np.int64)
        old_deg=np.diff(old_ip[:n+1]); new_deg=np.diff(new_ip[:n+1])
        if np.any(new_deg < old_deg):
            raise RuntimeError('growth rebase removed pre-existing edges')
        changed=np.flatnonzero(new_deg != old_deg).astype(np.int64)
        prev_src=0
        def copy_unchanged_sources(a_src:int,b_src:int):
            if b_src<=a_src:return
            oa,ob=int(old_ip[a_src]),int(old_ip[b_src]); na,nb=int(new_ip[a_src]),int(new_ip[b_src])
            if (ob-oa)!=(nb-na):raise RuntimeError('growth shifted unchanged source-range length')
            if ob>oa and not np.array_equal(oldcore.indices[oa:ob],neo.core.indices[na:nb]):
                raise RuntimeError('growth changed pre-existing ordering in unchanged source range')
            neo.core.plastic_delta[na:nb]=oldcore.plastic_delta[oa:ob]
        for src0 in changed:
            src=int(src0);copy_unchanged_sources(prev_src,src)
            os,oe=int(old_ip[src]),int(old_ip[src+1]); ns,ne=int(new_ip[src]),int(new_ip[src+1])
            oi=np.asarray(oldcore.indices[os:oe]); ni=np.asarray(neo.core.indices[ns:ne])
            if len(oi):
                if len(ni)>=len(oi) and np.array_equal(oi,ni[:len(oi)]):
                    neo.core.plastic_delta[ns:ns+len(oi)]=oldcore.plastic_delta[os:oe]
                else:
                    # Generic stable merge fallback for an overlay that inserts a new
                    # target inside an existing target-sorted source block.
                    j=0
                    for k,tgt in enumerate(oi):
                        while j<len(ni) and int(ni[j])<int(tgt):j+=1
                        if j>=len(ni) or int(ni[j])!=int(tgt):
                            raise RuntimeError(f'growth lost pre-existing edge {src}->{int(tgt)}')
                        neo.core.plastic_delta[ns+j]=oldcore.plastic_delta[os+k];j+=1
            prev_src=src+1
        copy_unchanged_sources(prev_src,n)
        neo.core.activity_ema_hz[:n]=oldcore.activity_ema_hz

        # Electrical continuity is valid only for a sane transient state.  A legacy
        # overflow attractor must not be copied into the daughter generation.
        _sanity=oldcore.transient_state_sanity() if hasattr(oldcore,'transient_state_sanity') else {'sane':True}
        if _sanity.get('sane',True):
            neo.core.v[:n]=oldcore.v; neo.core.g[:n]=oldcore.g; neo.core.refr[:n]=oldcore.refr
            neo.core.last_rates_hz[:n]=oldcore.last_rates_hz; neo.core.mod_state[:,:n]=oldcore.mod_state
            neo.core.delay_ring=[np.array(x,copy=True,dtype=np.int32) for x in oldcore.delay_ring]
            neo.core.ring_pos=int(oldcore.ring_pos)
        neo.core.rng.bit_generator.state=copy.deepcopy(oldcore.rng.bit_generator.state)
        # Internal slow/state wave is part of neural fast state. Growth must not
        # restart it at the posterior edge or phase-lock it to a generation event.
        neo.core._keepalive_time_ms=float(getattr(oldcore,'_keepalive_time_ms',0.0))
        neo.core._keepalive_ticks_until_update=int(getattr(oldcore,'_keepalive_ticks_until_update',0))
        neo.core._refresh_keepalive_bias()
        # World/body/life continuity.
        neo.time_s=float(old.time_s); neo.step_count=int(old.step_count); neo.pending_valence=0.0
        neo.body.position=old.body.position.copy(); neo.body.velocity=old.body.velocity.copy()
        neo.body.orientation=old.body.orientation.copy(); neo.body.angular_velocity=old.body.angular_velocity.copy()
        neo.body.contact=self._clone_contact(old.body.contact)
        neo.universe.restore(old.universe.snapshot()); neo.thermal.restore(old.thermal.snapshot())
        neo.homeostasis.restore(old.homeostasis.snapshot())
        neo.motor.restore(old.motor.snapshot())
        neo.proprio.restore(old.proprio.snapshot())
        if neo.eye_mapping_digest()==old.eye_mapping_digest():
            neo.eye.restore_temporal(old.eye.snapshot_temporal())
        neo.pending_motor_rates=copy.deepcopy(getattr(old,'pending_motor_rates',None))
        neo.last_motor=copy.deepcopy(old.last_motor)
        neo.events=list(old.events); neo.logs=list(old.logs[-240:])
        # A topology rewrite changes the network that produced every pre-growth
        # pressure streak.  Preserve per-neuron cooldown bookkeeping, but invalidate
        # all pressure evidence and require the new graph to demonstrate sustained
        # overload from scratch.  mark_grown() also starts a global cooldown so a
        # queue of mature candidates cannot cause one-generation-per-step cascades.
        tr=GrowthPressureTracker(neo.core.n,old.growth_pressure.config)
        tr.cooldown[:n]=old.growth_pressure.cooldown[:n]
        tr.total_epochs=old.growth_pressure.total_epochs
        tr.mark_grown(parent); neo.growth_pressure=tr
        self.exp=neo

    def _propose_growth_trial(self,candidate:dict):
        # Growth proposals are evaluated before promotion. Automatic high-rate
        # cloning is retired; no new permanent neuron is created by this trigger.
        from evolutionary_growth import propose_connections
        proposal=propose_connections(self.exp.core,candidate)
        proposal['time_s']=float(self.exp.time_s)
        folder=self.live_dir/'GrowthTrials';folder.mkdir(parents=True,exist_ok=True)
        target=folder/f"proposal-{self.exp.growth_pressure.total_epochs:09d}-{int(candidate['body_id'])}.json"
        target.write_text(json.dumps(proposal,indent=2),encoding='utf-8')
        self.exp.growth_pressure.mark_grown(int(candidate['index']))
        self.exp.last_growth_candidates=[]
        self.last_growth_trial=proposal
        self.last_status=f"Growth trial {proposal['status']}: {proposal['reason']}"
        return proposal

    def step(self,*,world_dt=.10,neural_ms=100.,learn=True,auto_growth=True,on_slice=None):
        # Twelve ~8.33-ms intervals at the default 100-ms window. Keep integer neural
        # ticks and matching world-time fractions; wall-clock speed never enters.
        with self.lock:
            ticks=max(1,int(round(float(neural_ms)/self.exp.core.config.dt_ms)))
            count=min(12,ticks,max(1,int(round(float(neural_ms)/(100./12.)))))
            chunks=[ticks//count+(1 if i<ticks%count else 0) for i in range(count)]
            self.last_growth_event=None
            self.exp.core.begin_learning_window(learn)
            started=time.perf_counter()
            try:
                for i,chunk in enumerate(chunks):
                    slice_started=time.perf_counter()
                    dt=float(world_dt)*chunk/ticks
                    ms=chunk*self.exp.core.config.dt_ms
                    log=self.exp.step(world_dt=dt,neural_ms=ms,learn=learn)
                    trace=self.exp.body.last_motion_trace
                    if not self.trajectory and trace:self.trajectory.append(trace[0].tolist())
                    self.trajectory.extend(p.tolist() for p in trace[1:])
                    # Bounded recent actual physics history; never infer a route
                    # from encountered planet centers or predict future motion.
                    if len(self.trajectory)>20000:del self.trajectory[:-20000]
                    if on_slice is not None and i<count-1:
                        snap=self.snapshot(log=log)
                        snap.update(slice_index=i+1,slice_count=count,slice_world_dt=dt,
                                    slice_wall_s=time.perf_counter()-slice_started,
                                    learning_pending=True)
                        on_slice(snap)
            finally:
                # Finish before releasing the engine lock: saves/seeds never omit
                # accumulated learning. Also retain completed slices after an error.
                learning=self.exp.core.finish_learning_window()
            if learning is not None and self.exp.last_report is not None:
                for key,value in learning.items():setattr(self.exp.last_report,key,value)
            self.exp.last_growth_candidates=list(self.exp.growth_pressure.observe(self.exp.core))
            c=self.growable_candidate()
            if auto_growth and c is not None:self._propose_growth_trial(c)
            snap=self.snapshot(log=log)
            snap.update(slice_index=count,slice_count=count,slice_world_dt=float(world_dt)*chunks[-1]/ticks,
                        slice_wall_s=time.perf_counter()-slice_started,learning_pending=False,
                        window_world_dt=float(world_dt),window_wall_s=time.perf_counter()-started)
            return snap

    def _current_motor_frame(self):
        if self.exp.last_motor is not None:return self.exp.last_motor
        # Reconstruct observer-only motor frame from the saved last-rates vector.
        rates={}
        for name,ids in self.exp.motor.readouts().items():
            rates[name]=float(np.mean(self.exp.core.last_rates_hz[np.asarray(ids,dtype=np.int32)])) if len(ids) else 0.0
        return self.exp.motor.actuate(rates,on_surface=self.exp.body.contact.on_surface)

    def _preview_visual(self):
        return self.exp.last_visual if self.exp.last_visual is not None else self.exp.eye.preview(self.exp.universe,self.exp.body,self.exp.time_s)

    def _last_log_dict(self):
        if self.exp.logs:
            from dataclasses import asdict
            return asdict(self.exp.logs[-1])
        state_path=self.live_dir/'specimen003_state.json'
        try:
            d=json.loads(state_path.read_text(encoding='utf-8')); logs=d.get('recent_logs',[])
            if logs:return logs[-1]
        except Exception:pass
        hs=self.exp.homeostasis.state; c=self.exp.body.contact; ki=self.exp.core.keepalive_info()
        return {
            'step':max(0,self.exp.step_count-1),'time_s':self.exp.time_s,'position':self.exp.body.position.tolist(),
            'velocity':self.exp.body.velocity.tolist(),'on_surface':c.on_surface,'body_id':c.body_id,'altitude':c.altitude,
            'altitude_body_id':c.altitude_body_id,'atmosphere_fraction':c.atmosphere_fraction,
            'gravity_body_id':self.exp.body.gravity_body_id,'gravity_accel':float(np.linalg.norm(self.exp.body.gravity_world)),
            'slip_angle_deg':float(self.exp.body.slip_angle_deg),'forward_speed':float(self.exp.body.forward_speed),
            'nearest_resource_id':c.nearest_resource_id,'nearest_resource_distance':c.nearest_resource_distance,
            'odor':0.,'resource_contact':False,'mouth_contact':False,'proboscis_contact':0.,'nutrient_consumed':0.,'consumed_total':hs.consumed_total,
            'starvation_rescue_s':getattr(self.exp,'starvation_rescue_s',0.0),'divine_fruit_drops':getattr(self.exp,'divine_fruit_drops',0),
            'satiety':hs.satiety,'nutritional_hunger':getattr(hs,'nutritional_hunger',hs.hunger),'hunger':hs.hunger,
            'hunger_pang':getattr(hs,'hunger_pang',1.0),'hunger_suppression':getattr(hs,'hunger_suppression',0.0),
            'approach_relief':getattr(hs,'approach_relief',0.0),'energy':hs.energy,'motor_energy_scale':self.exp.homeostasis.motor_power_scale(),'thermal_c':hs.thermal_c,
            'thermal_error':hs.thermal_error,'homeostatic_error':hs.total_error,'valence_for_next_epoch':self.exp.pending_valence,
            'network_hz':0.,'novelty_hz':0.,'keepalive_wave_phase':float(ki.get('phase',0.0)),
            'keepalive_wave_active_neurons':int(ki.get('active_neurons',0)),'keepalive_wave_peak_mv':float(ki.get('peak_mv',0.0)),
            'motor_mean_hz':float(np.mean(self.exp.core.last_rates_hz[self.exp.motor.motor_indices])) if len(self.exp.motor.motor_indices) else 0.,
            'flight_command':0.,'takeoff_command':0.,'walking_command':0.,'feeding_command':0.,'modified_edges':self.exp.core.memory_summary()['modified_edges'],
            'dopamine_mean':float(np.mean(self.exp.core.mod_state[0])),'octopamine_mean':float(np.mean(self.exp.core.mod_state[1])),
            'serotonin_mean':float(np.mean(self.exp.core.mod_state[2])),'growth_candidate_count':0,'top_growth_candidate_body':None,
            'eye_peak_side':'','eye_peak_yaw_deg':0.,'eye_peak_elevation_deg':0.,'visible_sources':0,
            'antenna_left_c':self.exp.thermal.left_temp_c,'antenna_right_c':self.exp.thermal.right_temp_c,
            'force_body':[0,0,0],'torque_body':[0,0,0],
        }

    @staticmethod
    def _altitude_display(log:dict) -> str:
        """Human-readable body-relative altitude; exact scalar stays in state/logs."""
        alt=float(log.get('altitude',float('inf')))
        ref=log.get('altitude_body_id')
        atm=float(log.get('atmosphere_fraction',0.0) or 0.0)
        if ref is None or not math.isfinite(alt):
            return "Spc [—]"
        if (not bool(log.get('on_surface',False))) and atm <= 1.0e-9:
            return f"Spc [nearest {ref}]"
        return f"{alt:.3f} [{ref}]"

    def snapshot(self,log=None):
        with self.lock:
            visual=self._preview_visual(); mf=self._current_motor_frame(); ld=self._last_log_dict() if log is None else log.__dict__.copy()
            view_pos=getattr(visual,'scene_position',None)
            view_ori=getattr(visual,'scene_orientation',None)
            pos=np.asarray(self.exp.body.position if view_pos is None else view_pos,float).copy()
            orientation=np.asarray(self.exp.body.orientation if view_ori is None else view_ori,float).copy()
            eye_pos=getattr(visual,'scene_eye_position',None)
            view_log=dict(ld)
            view_log.update(position=pos.tolist(),time_s=float(getattr(visual,'scene_time_s',self.exp.time_s)),
                            on_surface=bool(getattr(visual,'scene_on_surface',ld.get('on_surface',False))),
                            body_id=getattr(visual,'scene_body_id',ld.get('body_id')))
            view_velocity=getattr(visual,'scene_velocity',None)
            if view_velocity is not None:view_log['velocity']=view_velocity.tolist()
            stars=[]
            scene_sources=getattr(visual,'scene_sources',None)
            for o in (self.exp.universe.local_visual_sources(pos) if scene_sources is None else scene_sources):
                sid=str(o.star_id)
                # Habitat surface samples are needed by Yar's retina, but the human
                # observer renders one cached sphere per body below. Do not redundantly
                # transform/draw the 32-point optical surface cloud in the GUI.
                if sid.startswith('body:'):continue
                rel=np.asarray(o.position,float)-pos; d=float(np.linalg.norm(rel))
                if d<=1e-9:continue
                b=q_inverse_rotate(orientation,(rel/d).reshape(1,3))[0]
                yaw=math.degrees(math.atan2(float(b[1]),float(b[0]))); elev=math.degrees(math.atan2(float(b[2]),math.hypot(float(b[0]),float(b[1]))))
                kind='resource' if sid.startswith('resource:') else 'star'
                stars.append((yaw,elev,float(o.luminosity)/(0.2+(d/12.)**2),float(o.temperature),sid,kind,np.asarray(o.position,float).tolist()))
            # local bodies for observer map only
            bodies=[]; render_bodies=[]
            scene_bodies=getattr(visual,'scene_bodies',None)
            for b in (self.exp.universe.local_bodies(pos) if scene_bodies is None else scene_bodies):
                kind=_body_render_kind(b.body_id)
                resources=[{'id':r.patch_id,'position':b.resource_position(r).tolist(),
                            'remaining':self.exp.universe.resource_remaining(r.patch_id,r.amount),
                            'icon':_resource_icon(r.patch_id)} for r in self.exp.universe.resources_for_body(b)]
                bodies.append({'id':b.body_id,'alias':self.exp.universe.world_alias(b.body_id),'center':b.center.tolist(),'radius':b.radius,'temp_c':b.surface_temp_c,
                               'albedo':b.albedo,'render_kind':kind,'resources':resources})
                rel=np.asarray(b.center,float)-pos; d=float(np.linalg.norm(rel))
                if d>1e-9:
                    v=q_inverse_rotate(orientation,(rel/d).reshape(1,3))[0]
                    yaw=math.degrees(math.atan2(float(v[1]),float(v[0]))); elev=math.degrees(math.atan2(float(v[2]),math.hypot(float(v[0]),float(v[1]))))
                    angular=math.degrees(math.asin(min(0.999999,float(b.radius)/max(d,float(b.radius)+1e-12)))) if d>float(b.radius) else 89.0
                    render_bodies.append({'id':b.body_id,'alias':self.exp.universe.world_alias(b.body_id),'yaw':yaw,'elev':elev,'angular_radius_deg':angular,
                                          'distance':d,'temp_c':b.surface_temp_c,'render_kind':kind,'albedo':float(b.albedo),
                                          'center':b.center.tolist(),'radius':float(b.radius)})
            pressure=self.exp.growth_pressure.pressure_epochs
            cand=self.growable_candidate(); rates=self.exp.core.last_rates_hz
            ledger=[]
            for r in self._ledger:
                q=dict(r); pi=q.get('parent_index_now'); di=q.get('daughter_index_now')
                q['parent_live_hz']=None if pi is None else float(rates[pi]); q['daughter_live_hz']=None if di is None else float(rates[di])
                ledger.append(q)
            return {
                'specimen':self.specimen_label,'generation':self.generation,'graph':self.current_graph.name,'neurons':self.exp.core.n,'edges':self.exp.core.edge_count,
                'log':ld,'status':self.last_status,'last_growth_event':self.last_growth_event,
                'last_growth_trial':getattr(self,'last_growth_trial',None),
                'growth_threshold':self.exp.growth_pressure.config.sustain_epochs,'max_growth_pressure':int(np.max(pressure)) if len(pressure) else 0,
                'growth_cooldown_remaining':int(getattr(self.exp.growth_pressure,'global_cooldown_epochs',0)),
                'growth_cooldown_total':int(getattr(self.exp.growth_pressure.config,'cooldown_epochs',0)),
                'build_id':BUILD_ID,
                'growable_candidate':cand,'stars':stars,'bodies':bodies,'render_bodies':render_bodies,'facet_response':visual.facet_response.copy(),
                'eye_n_left':self.exp.eye_map.n_left,'eye_n_right':self.exp.eye_map.n_right,
                'eye_yaw_rad':self.exp.eye_map.yaw_rad.copy(),'eye_elevation_rad':self.exp.eye_map.elevation_rad.copy(),
                'eye_side':self.exp.eye_map.side.copy(),'eye_facet_confidence':self.exp.eye_map.facet_confidence.copy(),
                'eye_calibration':{'mapping_mode':self.exp.eye_map.mapping_mode,'official_workbook_loaded':bool(self.exp.eye_map.official_columns),'frontal_binocular_overlap_deg':float(2*self.exp.eye_map.config.frontal_cross_deg)},
                'eye_runtime':{'spectral_model':getattr(visual,'spectral_model','?'),'effective_fwhm_mean_deg':float(getattr(visual,'effective_fwhm_mean_deg',0.0)),'microsaccade_rms_deg':float(getattr(visual,'microsaccade_rms_deg',0.0)),'retinal_subsamples':int(getattr(visual,'retinal_subsamples',1)),'dither_sensor_count':int(getattr(visual,'dither_sensor_count',0))},
                'orientation':orientation.tolist(),'view_log':view_log,'eye_position':(pos if eye_pos is None else eye_pos).tolist(),'angular_velocity':self.exp.body.angular_velocity.tolist(),
                'motor_subclass':copy.deepcopy(mf.subclass_rates),'motor_effectors':mf.effectors.__dict__.copy(),
                'motor_unresolved':copy.deepcopy(mf.unresolved),'force_body':mf.force_body.tolist(),'torque_body':mf.torque_body.tolist(),
                'lineage':ledger,'events':list(self.exp.events[-100:]),'homeostasis':self.exp.homeostasis.snapshot()['state'],
                'universe_cached_sectors':len(getattr(self.exp.universe,'_body_cache',{})),
                'universe_resource_overrides':len(getattr(self.exp.universe,'_resource_remaining',{})),
                'universe_dynamic_resources':len(getattr(self.exp.universe,'_dynamic_resources',{})),
                'threads':self.exp.core.thread_info(),'gpu':self.exp.core.gpu_info(),'memory':self.exp.core.memory_summary(),
            }

    def emergency_care_package(self):
        """Thread-safe manual world-side food intervention."""
        with self.lock:
            info=self.exp.drop_emergency_care_package()
            if info is None:
                self.last_status='Emergency care package failed: no nearby habitat body'
                return None
            self.last_status=(f"Emergency care package: {info['resource_id']} "
                              f"on {info['body_id']} d={info['distance']:.3f}")
            return info

    def reset_to_seed(self):
        with self.lock:
            oldcore=self.exp.core
            self.exp=self._create_fresh_lineage(); self.manifest=json.loads(self.manifest_path.read_text(encoding='utf-8'))
            self.generation=int(self.manifest['working_generation']); self.current_graph=_resolve(self.manifest['current_graph'])
            self.exp.core.inherit_acceleration_from(oldcore)
            self._meta=self._load_meta(); self._ledger=[]; self.last_growth_event=None; self.last_status=f'Reset {self.specimen_label} to {self.seed_dir.name}'; self.last_save_sim_s=float(self.exp.time_s); self.trajectory=[]


class SimulationWorker(threading.Thread):
    def __init__(self,engine:LifeLineageEngine,out:queue.Queue):
        super().__init__(daemon=True); self.engine=engine; self.out=out
        self.running_evt=threading.Event(); self.step_evt=threading.Event(); self.stop_evt=threading.Event(); self.save_evt=threading.Event(); self.seed_evt=threading.Event(); self.care_package_evt=threading.Event()
        # One simulated clock by default: 0.1 world s == 100 neural ms.
        self.world_dt=.10; self.neural_ms=100.; self.learn=True; self.auto_growth=True; self.autosave_sim_s=30.
        self._last_thread_state=None; self.last_save_wall_s=0.; self.last_cycle_wall_s=0.
    def run(self):
        try:
            accel=self.engine.apply_runtime_acceleration()
            self.out.put(('event',f"Runtime acceleration applied: CPU {accel['threads'].get('threads')}/{accel['threads'].get('permitted')} • GPU {'on' if accel['gpu'].get('enabled') else 'off'}"))
        except Exception as e:self.out.put(('event',f'Runtime acceleration warning: {e}'))
        self.out.put(('snapshot',self.engine.snapshot()))
        while not self.stop_evt.is_set():
            if self.care_package_evt.is_set():
                self.care_package_evt.clear()
                try:
                    info=self.engine.emergency_care_package()
                    snap=self.engine.snapshot()
                    if info is None:
                        self.out.put(('error',self.engine.last_status))
                    else:
                        self.out.put(('event',self.engine.last_status))
                    self.out.put(('snapshot',snap))
                except Exception as e:
                    self.out.put(('error',f'Emergency care package failed: {e}'))
            if self.seed_evt.is_set():
                self.seed_evt.clear()
                try:
                    target=self.engine.save_as_seed()
                    self.out.put(('event',f'Saved {target.name}; available under New Fly in the loader.'))
                except Exception as e:self.out.put(('error',f'Save seed failed: {e}'))
            if self.save_evt.is_set():
                self.save_evt.clear(); st_save=time.perf_counter()
                try:
                    self.engine.save(); self.last_save_wall_s=time.perf_counter()-st_save
                    snap=self.engine.snapshot(); snap['manual_save_wall_s']=self.last_save_wall_s; snap['save_wall_s']=self.last_save_wall_s; snap['last_save_wall_s']=self.last_save_wall_s
                    self.out.put(('event',f'{self.engine.last_status}  save wall={self.last_save_wall_s:.3f}s')); self.out.put(('snapshot',snap))
                except Exception as e:self.out.put(('error',f'Save failed: {e}'))
            if not (self.running_evt.is_set() or self.step_evt.is_set()):time.sleep(.03);continue
            self.step_evt.clear(); started=time.perf_counter()
            try:
                def publish_slice(snap):
                    wall=max(1e-9,float(snap['slice_wall_s']))
                    snap['wall_step_s']=wall; snap['sim_speed']=snap['slice_world_dt']/wall
                    snap['effective_sim_speed']=snap['sim_speed']
                    self.out.put(('snapshot',snap))
                snap=self.engine.step(world_dt=self.world_dt,neural_ms=self.neural_ms,learn=self.learn,auto_growth=self.auto_growth,on_slice=publish_slice)
                compute_wall=max(1e-9,time.perf_counter()-started); save_wall=0.
                snap['wall_step_s']=compute_wall; snap['sim_speed']=self.world_dt/compute_wall
                if self.autosave_sim_s>0 and self.engine.exp.time_s-self.engine.last_save_sim_s>=self.autosave_sim_s:
                    st_save=time.perf_counter(); self.engine.save(); save_wall=time.perf_counter()-st_save; self.last_save_wall_s=save_wall
                    snap['status']=self.engine.last_status  # Saving does not advance the state; reuse this epoch's snapshot.
                    self.out.put(('event',f'{self.engine.last_status}  save wall={save_wall:.3f}s'))
                cycle_wall=compute_wall+save_wall; self.last_cycle_wall_s=cycle_wall
                snap['save_wall_s']=save_wall; snap['last_save_wall_s']=self.last_save_wall_s; snap['cycle_wall_s']=cycle_wall
                snap['effective_sim_speed']=self.world_dt/max(1e-9,cycle_wall)
                ti=snap.get('threads',{})
                ts=(ti.get('mode'),ti.get('threads'),ti.get('tuning_complete'),ti.get('best_threads'))
                if ts != self._last_thread_state:
                    self._last_thread_state=ts
                    scores=ti.get('benchmark_scores') or {}; one=scores.get(1,scores.get('1')); best=scores.get(ti.get('best_threads'),scores.get(str(ti.get('best_threads'))))
                    gain=(float(one)/float(best)) if one and best else None
                    detail=f" tuner≈{gain:.2f}x vs 1T" if gain else ''
                    trail='→'.join(str(x) for x in ti.get('candidates',[])) or '?'
                    coarse='→'.join(str(x) for x in ti.get('coarse_probes',[])) or '?'
                    reg=ti.get('first_regression_threads')
                    anchor=ti.get('refinement_anchor_threads')
                    bracket=ti.get('search_bracket') or ['?','?']
                    self.out.put(('event',f"THREADS {ti.get('backend')} mode={ti.get('mode')} using={ti.get('threads')}/{ti.get('permitted')} best={ti.get('best_threads')} tuned={ti.get('tuning_complete')}{detail}; coarse={coarse}; bracket={bracket[0]}..{bracket[1]}; probes={trail}; first-slow={reg if reg is not None else 'runtime cap/no regression'}; anchor={anchor}"))
                self.out.put(('snapshot',snap))
                if snap.get('last_growth_event'):self.out.put(('growth',snap['last_growth_event']))
            except Exception as e:
                import traceback; self.running_evt.clear(); self.out.put(('error',f'{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}'))


class ObservatoryApp:
    def __init__(self,engine:LifeLineageEngine):
        import tkinter as tk
        from tkinter import ttk,messagebox,simpledialog
        self.tk=tk;self.ttk=ttk;self.messagebox=messagebox;self.simpledialog=simpledialog;self.engine=engine
        self.root=tk.Tk();self.root.title(f"No Man's Fly — {engine.specimen_label} Life Observatory — {BUILD_ID}");self.root.geometry('1580x980');self.root.minsize(1250,780)
        self.q=SnapshotQueue();self.worker=SimulationWorker(engine,self.q)
        self.history={'net':[],'motor':[],'novelty':[],'hunger':[]};self.path=[];self.last_growth_key=None
        self._world_camera_center=None
        self._thread_setting_applied=None; self._latest_snapshot=None; self._snapshot_received_wall=None; self._last_authoritative_key=None
        self._ui_ready=False; self._viewer_target_fps=15.; self._viewer_measured_fps=0.; self._viewer_frames=0; self._viewer_window_started=time.perf_counter()
        # Observer rendering caches. Keep Tk object churn off the window thread: eyes
        # are one raster image each, sky objects are pooled, and hidden notebook tabs
        # are updated only when selected. None of this state is part of the specimen.
        self._eye_photos={};self._eye_image_items={};self._vision_overlay_size=None
        self._planet_sprite_cache={}; self._planet_tk_cache={}
        self._sky_star_items=[];self._sky_static_size=None;self._sky_status_item=None
        self._build();self.root.protocol('WM_DELETE_WINDOW',self._close)
        # Critical ordering: finish Tk geometry before the worker can publish the first
        # frame.  This prevents the startup snapshot being painted into 1x1 canvases.
        self.root.update_idletasks(); self._ui_ready=self._canvases_ready(); self.worker.start()
        self.root.after_idle(self._finish_layout); self.root.after(20,self._poll); self.root.after(25,self._viewer_tick)

    def _build(self):
        tk=self.tk;ttk=self.ttk; st=ttk.Style()
        try:st.theme_use('clam')
        except Exception:pass
        st.configure('TFrame',background='#0d1118');st.configure('TLabel',background='#0d1118',foreground='#d7e2f0');st.configure('TCheckbutton',background='#0d1118',foreground='#d7e2f0')
        top=ttk.Frame(self.root);top.pack(fill='x',padx=8,pady=(8,4))
        self.start_btn=ttk.Button(top,text='▶ Start',command=self._toggle);self.start_btn.pack(side='left',padx=3)
        ttk.Button(top,text='Single step',command=self._single_step).pack(side='left',padx=3)
        ttk.Button(top,text='Save now',command=lambda:self.worker.save_evt.set()).pack(side='left',padx=3)
        ttk.Button(top,text='Save as seed',command=lambda:self.worker.seed_evt.set()).pack(side='left',padx=3)
        self.learn_var=tk.BooleanVar(value=True);self.grow_var=tk.BooleanVar(value=True)
        ttk.Checkbutton(top,text='Learning',variable=self.learn_var,command=self._sync).pack(side='left',padx=(16,3))
        ttk.Checkbutton(top,text='Growth trials armed',variable=self.grow_var,command=self._sync).pack(side='left',padx=3)
        ttk.Label(top,text='world dt').pack(side='left',padx=(14,2));self.dt=tk.StringVar(value='.10');ttk.Entry(top,textvariable=self.dt,width=5).pack(side='left')
        ttk.Label(top,text='neural ms (locked)').pack(side='left',padx=(7,2));self.nms=tk.StringVar(value='100');ttk.Entry(top,textvariable=self.nms,width=6,state='readonly').pack(side='left')
        ttk.Label(top,text='autosave sim s').pack(side='left',padx=(7,2));self.autos=tk.StringVar(value='30');ttk.Entry(top,textvariable=self.autos,width=6).pack(side='left')
        ttk.Label(top,text='threads').pack(side='left',padx=(7,2));self.threads=tk.StringVar(value=str(self.engine.thread_spec or 'auto'));ttk.Entry(top,textvariable=self.threads,width=6).pack(side='left')
        ttk.Label(top,text='viewer fps').pack(side='left',padx=(7,2));self.viewer_fps=tk.StringVar(value='15');ttk.Entry(top,textvariable=self.viewer_fps,width=4).pack(side='left')
        ttk.Button(top,text='Apply',command=self._sync).pack(side='left',padx=3)
        main=ttk.Frame(self.root);main.pack(fill='both',expand=True,padx=8,pady=4)
        left=ttk.Frame(main);left.pack(side='left',fill='both',expand=True)
        right=ttk.Frame(main);right.pack(side='right',fill='both',padx=(8,0))
        self.sky=tk.Canvas(left,bg='#000106',height=450,highlightthickness=1,highlightbackground='#263445');self.sky.pack(fill='both',expand=True)
        ttk.Label(left,text='OBSERVER SKY / HABITATS • gravity-level while grounded, body-frame in flight • never feeds the brain').pack(anchor='w')
        worldbar=ttk.Frame(left);worldbar.pack(fill='x',pady=(2,2))
        self.current_world_id=None
        self.current_world_btn=ttk.Button(worldbar,text='World: —',command=self._edit_current_world_alias)
        self.current_world_btn.pack(side='left',padx=(0,6))
        ttk.Button(worldbar,text='🗺 Travel atlas',command=self._open_world_atlas).pack(side='left')
        ttk.Label(worldbar,text='planet labels edit aliases').pack(side='left',padx=(8,0))
        self._atlas_window=None; self._atlas_canvas=None; self._atlas_tree=None

        er=ttk.Frame(left);er.pack(fill='x',pady=(4,0))
        ttk.Label(er,text='FUSED VISUAL FIELD • receptor evidence, NOT attention').pack(anchor='w')
        # Body View 3: keep the complete compound-eye presentation packed on the
        # LEFT (fused field plus both raw-facet strips) and give every elastic pixel
        # of horizontal window growth to the live observer-only fly model on the
        # RIGHT.  No resize operation may stretch fake black space between actual
        # receptor columns.  The body model never feeds pose, steering, reward, or
        # sensory information back to Yar.
        vision_pair=ttk.Frame(er);vision_pair.pack(fill='x',expand=False)
        vision_pair.columnconfigure(0,weight=0);vision_pair.columnconfigure(1,weight=1)
        vision_pair.rowconfigure(0,weight=1)

        eye_stack=ttk.Frame(vision_pair)
        eye_stack.grid(row=0,column=0,sticky='nsw',padx=(0,2))
        self.vision=tk.Canvas(eye_stack,bg='#03020a',height=244,width=400,highlightthickness=1,highlightbackground='#263445')
        self.vision.pack(fill='x',expand=False)
        self._vision_column_pitch_px=3
        self._vision_margin_px=6

        raw=ttk.Frame(eye_stack);raw.pack(fill='x',pady=(2,0))
        lw=ttk.Frame(raw);rw=ttk.Frame(raw)
        lw.pack(side='left',fill='y',expand=False,padx=(0,1));rw.pack(side='left',fill='y',expand=False,padx=(1,0))
        ttk.Label(lw,text='L RAW FACETS',font=('TkDefaultFont',7)).pack()
        ttk.Label(rw,text='R RAW FACETS',font=('TkDefaultFont',7)).pack()
        self.eye_l=tk.Canvas(lw,bg='#03020a',height=76,width=198,highlightthickness=1,highlightbackground='#263445');self.eye_l.pack(fill='none',expand=False)
        self.eye_r=tk.Canvas(rw,bg='#03020a',height=76,width=198,highlightthickness=1,highlightbackground='#263445');self.eye_r.pack(fill='none',expand=False)

        self.fly_model=tk.Canvas(vision_pair,bg='#05070b',height=342,width=1,highlightthickness=1,highlightbackground='#263445')
        self.fly_model.grid(row=0,column=1,sticky='nsew')


        self.tabs=ttk.Notebook(right);self.tabs.pack(fill='both',expand=True)
        self.summary=ttk.Frame(self.tabs);self.motor=ttk.Frame(self.tabs);self.life=ttk.Frame(self.tabs);self.neuro=ttk.Frame(self.tabs);self.performance=ttk.Frame(self.tabs);self.eventtab=ttk.Frame(self.tabs)
        for f,n in ((self.summary,'Summary'),(self.motor,'Motor / body'),(self.life,'Life / world'),(self.neuro,'Neurogenesis'),(self.performance,'Performance'),(self.eventtab,'Events')):self.tabs.add(f,text=n)
        self.tele=tk.Text(self.summary,width=58,height=30,bg='#080b10',fg='#d7e2f0',font=('Consolas',10),relief='flat');self.tele.pack(fill='x')
        self.hist=tk.Canvas(self.summary,width=500,height=190,bg='#05070b',highlightthickness=1,highlightbackground='#263445');self.hist.pack(fill='x',pady=(6,0))

        cols=('sub','L','R','status');self.motor_tree=ttk.Treeview(self.motor,columns=cols,show='headings',height=14)
        for c,w in zip(cols,(80,90,90,190)):self.motor_tree.heading(c,text=c);self.motor_tree.column(c,width=w,anchor='center')
        self.motor_tree.pack(fill='x')
        self.motor_text=tk.Text(self.motor,height=24,bg='#080b10',fg='#d7e2f0',font=('Consolas',9),relief='flat');self.motor_text.pack(fill='both',expand=True,pady=(5,0))

        life_controls=ttk.Frame(self.life);life_controls.pack(fill='x',pady=(0,4))
        self.care_package_btn=ttk.Button(life_controls,text='🥥 Emergency care package drop',command=self._drop_care_package);self.care_package_btn.pack(side='left',padx=(0,8))
        ttk.Label(life_controls,text='World-side food only — no neural/reward/steering injection').pack(side='left')
        self.life_text=tk.Text(self.life,height=20,bg='#080b10',fg='#d7e2f0',font=('Consolas',10),relief='flat');self.life_text.pack(fill='x')
        self.world=tk.Canvas(self.life,width=500,height=380,bg='#03060a',highlightthickness=1,highlightbackground='#263445');self.world.pack(fill='both',expand=True,pady=(5,0))

        ncols=('W','time','parent','type','daughter','edges','P Hz','D Hz');self.neuro_tree=ttk.Treeview(self.neuro,columns=ncols,show='headings',height=15)
        widths=(40,70,90,130,110,55,70,70)
        for c,w in zip(ncols,widths):self.neuro_tree.heading(c,text=c);self.neuro_tree.column(c,width=w,anchor='center')
        self.neuro_tree.pack(fill='both',expand=True)
        self.neuro_text=tk.Text(self.neuro,height=10,bg='#080b10',fg='#d7e2f0',font=('Consolas',9),relief='flat');self.neuro_text.pack(fill='x',pady=(5,0))

        self.perf_live_var=tk.StringVar(value='Waiting for first completed neural epoch…')
        ttk.Label(self.performance,textvariable=self.perf_live_var,anchor='w',font=('Consolas',9)).pack(fill='x',padx=6,pady=(6,3))
        self.perf_text=tk.Text(self.performance,bg='#080b10',fg='#d7e2f0',font=('Consolas',9),relief='flat');self.perf_text.pack(fill='both',expand=True,padx=4,pady=(0,4))

        self.events=tk.Text(self.eventtab,bg='#080b10',fg='#a9bed5',font=('Consolas',9),relief='flat');self.events.pack(fill='both',expand=True)
        self.tabs.bind('<<NotebookTabChanged>>',self._on_tab_changed,add='+')
        self._event(f'{self.engine.specimen_label} Observatory opened. Observer UI has no neural control path.');self._sync()

    def _sync(self):
        self.worker.learn=bool(self.learn_var.get());self.worker.auto_growth=bool(self.grow_var.get())
        try:
            self.worker.world_dt=max(.01,float(self.dt.get()))
            self.worker.neural_ms=1000.0*self.worker.world_dt
            self.nms.set(f'{self.worker.neural_ms:.0f}')
        except Exception:pass
        try:self.worker.autosave_sim_s=max(0.,float(self.autos.get()))
        except Exception:pass
        try:self._viewer_target_fps=max(0.,min(60.,float(self.viewer_fps.get())))
        except Exception:self._viewer_target_fps=15.;self.viewer_fps.set('15')
        try:
            wanted=(self.threads.get().strip().lower() or 'auto')
            if wanted != self._thread_setting_applied:
                info=self.engine.configure_threads(wanted); self._thread_setting_applied=wanted
                scores=info.get('benchmark_scores') or {}; one=scores.get(1,scores.get('1')); best=scores.get(info.get('best_threads'),scores.get(str(info.get('best_threads'))))
                gain=(float(one)/float(best)) if one and best else None
                suffix=f"; tuner score {float(one):.3f}s→{float(best):.3f}s ≈{gain:.2f}x" if gain else ''
                self._event(f"Thread policy applied: {info['mode']} {info['threads']}/{info['permitted']} ({info['backend']}){suffix}")
        except Exception as e:
            self._event(f'Thread setting rejected: {e}')
    def _single_step(self):
        self._sync()
        self.worker.step_evt.set()
        self._event('Single step queued.')

    def _toggle(self):
        self._sync()
        if self.worker.running_evt.is_set():self.worker.running_evt.clear();self.start_btn.configure(text='▶ Start');self._event('Paused.')
        else:self.worker.running_evt.set();self.start_btn.configure(text='❚❚ Pause');self._event('Running.')
    def _reset(self):
        if not self.messagebox.askyesno(f'Reset {self.engine.specimen_label}?',f'Discard this live state and restore {self.engine.seed_dir.name}?'):return
        self.worker.running_evt.clear();self.start_btn.configure(text='▶ Start')
        try:self.engine.reset_to_seed();self.q.put(('snapshot',self.engine.snapshot()));self._event(f'Restored {self.engine.seed_dir.name}.')
        except Exception as e:self.messagebox.showerror('Reset failed',str(e))
    def _event(self,text):
        self.events.insert('end',f'[{time.strftime("%H:%M:%S")}] {text}\n');self.events.see('end')
    def _canvases_ready(self):
        try:return min(self.sky.winfo_width(),self.sky.winfo_height(),self.vision.winfo_width(),self.vision.winfo_height(),self.eye_l.winfo_width(),self.eye_r.winfo_width(),self.fly_model.winfo_width(),self.fly_model.winfo_height()) > 40
        except Exception:return False
    def _finish_layout(self):
        self.root.update_idletasks(); self._ui_ready=self._canvases_ready()
        if not self._ui_ready:self.root.after(25,self._finish_layout);return
        if self._latest_snapshot is not None:self._draw_authoritative(self._latest_snapshot)
    def _poll(self):
        # Drain the queue completely but paint only the newest snapshot. The queue
        # bridge MUST reschedule itself even if an observer-only paint routine fails;
        # otherwise the simulation can continue invisibly while telemetry freezes.
        newest=None
        try:
            while True:
                kind,p=self.q.get_nowait()
                if kind=='snapshot':
                    newest=p
                elif kind=='event':self._event(str(p))
                elif kind=='growth':
                    k=(p.get('working_generation'),p.get('parent_body'))
                    if k!=self.last_growth_key:self.last_growth_key=k;self._event(f"STRUCTURAL GROWTH → W{p['working_generation']:03d}: {p['parent_body']} {p['parent_type']} → {p['daughter_body']}")
                elif kind=='error':
                    self.worker.running_evt.clear(); self.start_btn.configure(text='▶ Start')
                    self._event('ERROR: '+str(p))
        except queue.Empty:
            pass
        except Exception as e:
            import traceback
            self._event('UI QUEUE ERROR: '+f'{type(e).__name__}: {e}')
            traceback.print_exc()
        try:
            if newest is not None:
                self._latest_snapshot=newest;self._snapshot_received_wall=time.perf_counter()
                if self._ui_ready and self._viewer_target_fps<=0:
                    try:
                        self._draw_authoritative(newest)
                    except Exception as e:
                        import traceback
                        self._event('UI DRAW ERROR: '+f'{type(e).__name__}: {e}')
                        traceback.print_exc()
        finally:
            # Never silently sever worker -> Observatory delivery.
            try:self.root.after(20,self._poll)
            except Exception:pass


    def _viewer_dt_sim(self,s):
        # Sky, retina and map share a captured pose; never predict future motion.
        return 0.
    def _selected_tab(self):
        try:return self.tabs.select()
        except Exception:return ''
    def _tab_is(self,frame):return self._selected_tab()==str(frame)
    def _draw_active_tab(self,s):
        if self._tab_is(self.summary):self._summary(s);self._history()
        elif self._tab_is(self.motor):self._motor(s)
        elif self._tab_is(self.life):self._life(s,viewer_dt=0.,draw_world=True)
        elif self._tab_is(self.neuro):self._neuro(s)
        elif self._tab_is(self.performance):self._performance(s)
    def _on_tab_changed(self,event=None):
        if self._ui_ready and self._latest_snapshot is not None:self._draw_active_tab(self._latest_snapshot)

    def _viewer_tick(self):
        started=time.perf_counter()
        try:
            if self._ui_ready and self._latest_snapshot is not None and self._viewer_target_fps>0:
                s=self._latest_snapshot
                key=(id(s),self.sky.winfo_width(),self.sky.winfo_height(),
                     self.world.winfo_width(),self.world.winfo_height(),self._selected_tab())
                if key != getattr(self,'_painted_frame_key',None):
                    paint_start=time.perf_counter()
                    self._draw_authoritative(s)
                    self._last_render_wall=time.perf_counter()-paint_start
                    previous=getattr(self,'_last_paint_wall',None)
                    if previous is not None and started>previous:self._viewer_measured_fps=1./(started-previous)
                    self._last_paint_wall=started
                    self._painted_frame_key=key
                    self._viewer_frames+=1
            elapsed=started-self._viewer_window_started
            if elapsed>=1.:
                self._viewer_frames=0;self._viewer_window_started=started
                if self._latest_snapshot is not None:self._update_perf_line(self._latest_snapshot)
        except Exception as e:
            self._event(f'UI DRAW ERROR: {type(e).__name__}: {e}')
        finally:
            period=1./self._viewer_target_fps if self._viewer_target_fps>0 else .1
            delay=max(1,int(1000*max(0.,period-(time.perf_counter()-started))))
            self.root.after(delay,self._viewer_tick)

    def _draw_authoritative(self,s):
        l=s['log'];p=s.get('view_log',l).get('position',[0,0,0]); key=(int(s.get('generation',0)),float(l.get('time_s',0.)))
        if key != self._last_authoritative_key:
            self._last_authoritative_key=key; self.path.append(tuple(p));self.path=self.path[-400:]
            for k,v in (('net',l.get('network_hz',0)),('motor',l.get('motor_mean_hz',0)),('novelty',l.get('novelty_hz',0)),('hunger',100*l.get('hunger',0))):
                self.history[k].append(float(v));self.history[k]=self.history[k][-180:]
        # Always-visible observer panels update on authoritative epochs. Hidden notebook
        # panels are lazy-rendered only when selected.
        self._sky(s);self._eyes(s);self._fly_model(s);self._update_current_world_designation(s);self._draw_active_tab(s);self._update_perf_line(s,0.)

    def _current_world_from_snapshot(self,s):
        ld=s.get('log',{}) or {}
        bid=ld.get('body_id') if bool(ld.get('on_surface',False)) else ld.get('altitude_body_id')
        return None if bid in (None,'','None') else str(bid)

    def _update_current_world_designation(self,s):
        bid=self._current_world_from_snapshot(s); self.current_world_id=bid
        if bid is None:
            self.current_world_btn.configure(text='World: Spc')
            return
        alias=''
        for rb in s.get('render_bodies',[]):
            if str(rb.get('id'))==bid: alias=str(rb.get('alias','')).strip(); break
        if not alias:
            for b in s.get('bodies',[]):
                if str(b.get('id'))==bid: alias=str(b.get('alias','')).strip(); break
        self.current_world_btn.configure(text=(f'World: {alias}\n{bid}' if alias else f'World: {bid}'))





    def _edit_current_world_alias(self):
        if self.current_world_id:self._edit_world_alias(self.current_world_id)

    def _edit_world_alias(self,body_id):
        bid=str(body_id)
        try: current=self.engine.world_alias(bid)
        except Exception: current=''
        alias=self.simpledialog.askstring('World alias',f'Alias for {bid}\n\nBlank = use procedural designation only.',initialvalue=current,parent=self.root)
        if alias is None:return
        try:
            name,snap=self.engine.set_world_alias(bid,alias)
            self.q.put(('snapshot',snap))
            self._event(f"World alias {'set to '+name if name else 'cleared'} for {bid}.")
            self._refresh_world_atlas()
        except Exception as e:self.messagebox.showerror('World alias failed',str(e),parent=self.root)

    def _open_world_atlas(self):
        if self._atlas_window is not None:
            try:
                if self._atlas_window.winfo_exists():self._atlas_window.lift();self._refresh_world_atlas();return
            except Exception:pass
        win=self.tk.Toplevel(self.root);self._atlas_window=win;win.title("Yar's Travel Atlas");win.geometry('1100x720');win.minsize(820,520)
        top=self.ttk.Frame(win);top.pack(fill='x',padx=8,pady=6)
        self.ttk.Label(top,text='Persistent travel atlas • click a world to select it').pack(side='left')
        self.ttk.Button(top,text='Refresh',command=self._refresh_world_atlas).pack(side='right')
        self.ttk.Button(top,text='Edit selected alias',command=self._atlas_edit_selected).pack(side='right',padx=4)
        body=self.ttk.Frame(win);body.pack(fill='both',expand=True,padx=8,pady=(0,8))
        self._atlas_canvas=self.tk.Canvas(body,bg='#02050a',highlightthickness=1,highlightbackground='#263445');self._atlas_canvas.pack(side='left',fill='both',expand=True)
        cols=('name','designation','landings','first','last','xyz')
        tree=self.ttk.Treeview(body,columns=cols,show='headings',height=20);self._atlas_tree=tree
        for c,w in zip(cols,(150,155,65,75,75,210)):tree.heading(c,text=c);tree.column(c,width=w,anchor='center')
        tree.pack(side='right',fill='y',padx=(8,0));tree.bind('<Button-3>',self._atlas_tree_edit)
        self.ttk.Label(win,text='Equal-scale X/Y projection. Cyan: recent actual flight in this session; dots: encountered worlds. Right-click/Edit button changes aliases.').pack(anchor='w',padx=8,pady=(0,6))
        win.protocol('WM_DELETE_WINDOW',self._close_world_atlas);self._refresh_world_atlas()

    def _close_world_atlas(self):
        try:self._atlas_window.destroy()
        except Exception:pass
        self._atlas_window=None;self._atlas_canvas=None;self._atlas_tree=None

    def _atlas_selected_body_id(self):
        if self._atlas_tree is None:return None
        sel=self._atlas_tree.selection()
        return str(sel[0]) if sel else None



    def _atlas_edit_selected(self):
        if self._atlas_tree is None:return
        sel=self._atlas_tree.selection()
        if sel:self._edit_world_alias(sel[0])


    def _atlas_tree_edit(self,event=None):
        if self._atlas_tree is None:return
        if event is not None:
            row=self._atlas_tree.identify_row(event.y)
            if row:self._atlas_tree.selection_set(row)
        sel=self._atlas_tree.selection()
        if sel:self._edit_world_alias(sel[0])

    def _refresh_world_atlas(self):
        if self._atlas_window is None or self._atlas_canvas is None or self._atlas_tree is None:return
        try:data=self.engine.atlas_snapshot()
        except Exception:return
        c=self._atlas_canvas;c.delete('all');w=max(40,c.winfo_width());h=max(40,c.winfo_height())
        worlds=list(data.get('worlds',[]));hist=list(data.get('travel_history',[]))
        for x in self._atlas_tree.get_children():self._atlas_tree.delete(x)
        if not worlds:
            c.create_text(w/2,h/2,text='No recorded worlds yet',fill='#9fb5ca',font=('Consolas',14));return
        trail=data.get('body_trajectory',[])
        pts=np.asarray([q.get('center',[0,0,0]) for q in worlds]+trail,dtype=float);xs=pts[:,0];ys=pts[:,1]
        xmin,xmax=float(xs.min()),float(xs.max());ymin,ymax=float(ys.min()),float(ys.max());dx=max(1.0,xmax-xmin);dy=max(1.0,ymax-ymin)
        pad=55
        scale=min(max(1,w-2*pad)/dx,max(1,h-2*pad)/dy)
        def xy(v):
            return w/2+(float(v[0])-(xmin+xmax)/2)*scale,h/2-(float(v[1])-(ymin+ymax)/2)*scale
        if len(trail)>1:
            coords=[coord for point in trail for coord in xy(point)]
            c.create_line(*coords,fill='#55cbd3',width=2,tags=('actual-flight-trail',))
        else:
            c.create_text(10,h-15,text='No flight trace recorded in this session yet',anchor='w',fill='#9fb5ca')
        for q in worlds:
            bid=str(q.get('body_id'));alias=str(q.get('alias','')).strip();name=alias or bid;x,y=xy(q.get('center',[0,0,0]));land=int(q.get('landing_count',0) or 0)
            r=8 if land else 5;fill='#7be08d' if land else '#70a8d8'
            oval=c.create_oval(x-r,y-r,x+r,y+r,fill=fill,outline='#d7e2f0',width=1)
            primary=c.create_text(x,y-12,text=name,fill='#f0f5fb',anchor='s',font=('Consolas',9,'bold'))
            items=[oval,primary]
            if alias:items.append(c.create_text(x,y+12,text=bid,fill='#9fb5ca',anchor='n',font=('Consolas',7)))
            for item in items:
                c.tag_bind(item,'<Button-1>',lambda ev,b=bid:self._atlas_tree.selection_set(b))
                c.tag_bind(item,'<Button-3>',lambda ev,b=bid:self._edit_world_alias(b))
            cen=q.get('center',[0,0,0]);first=q.get('first_landed_s');last=q.get('last_landed_s') or q.get('last_encounter_s')
            self._atlas_tree.insert('', 'end', iid=bid, values=(name,bid,land,'' if first is None else f'{float(first):.1f}','' if last is None else f'{float(last):.1f}',f'{float(cen[0]):+.2f}, {float(cen[1]):+.2f}, {float(cen[2]):+.2f}'))

    def _thread_perf(self,s):
        ti=s.get('threads',{}) or {}; scores=ti.get('benchmark_scores') or {}
        def score(k):return scores.get(k,scores.get(str(k)))
        one=score(1); best=score(ti.get('best_threads'))
        gain=(float(one)/float(best)) if one and best and float(best)>0 else None
        return ti,one,best,gain
    def _update_perf_line(self,s,viewer_dt=0.):
        ti,one,best,gain=self._thread_perf(s); core=ti.get('last_epoch_wall_s'); step=s.get('wall_step_s'); save=s.get('last_save_wall_s',0.)
        bits=[]
        if core is not None:bits.append(f'neural slice {float(core):.3f}s')
        if step is not None:bits.append(f'closed-loop {float(step):.3f}s')
        bits.append(f"sim {float(s.get('sim_speed',0.)):.3f}×")
        if s.get('slice_count',1)>1:bits.append(f"slice {s.get('slice_index',1)}/{s['slice_count']}")
        bits.append(f"CPU {ti.get('threads',1)}/{ti.get('permitted',1)} {ti.get('mode','?')}"); gpu=s.get('gpu',{}) or {}; bits.append(f"GPU {'on' if gpu.get('enabled') else 'off'}")
        if one is not None and best is not None and gain is not None:bits.append(f'tuner 1T {float(one):.3f}s → best {float(best):.3f}s ({gain:.2f}×)')
        age=max(0.,time.perf_counter()-self._snapshot_received_wall) if self._snapshot_received_wall is not None else 0.
        render=getattr(self,'_last_render_wall',None)
        bits.append(f'new frames {self._viewer_measured_fps:.2f}/s (cap {self._viewer_target_fps:g})')
        if render is not None:bits.append(f'render {1000.*render:.0f} ms')
        bits.append(f'last state {age:.1f}s ago' if self.worker.running_evt.is_set() else 'paused')
        if viewer_dt>0:bits.append(f'visual predict +{viewer_dt:.3f} sim s')
        if save:bits.append(f'last save {float(save):.3f}s')
        self.perf_live_var.set('  |  '.join(bits))

    def _prepare_sky_static(self,w,h):
        size=(int(w),int(h));c=self.sky
        if self._sky_static_size==size and self._sky_status_item is not None:return
        self._sky_static_size=size
        if self._sky_status_item is None:
            self._sky_status_item=c.create_text(8,8,text='',fill='#cfe9ff',anchor='nw',font=('Consolas',10,'bold'),tags=('sky-status',))
        c.tag_raise('sky-status')

    @staticmethod
    def _sprite_bucket(diameter:int)->int:
        d=max(4,int(diameter))
        for q in (8,12,16,24,32,48,64,96,128,192,256,384,512):
            if d<=q:return q
        return 512

    def _planet_sprite(self,body_id:str,diameter:int,kind:str):
        bucket=self._sprite_bucket(diameter); key=(str(body_id),bucket,str(kind))
        z=self._planet_sprite_cache.get(key)
        if z is None:
            z=_planet_sprite_array(str(body_id),bucket,str(kind)); self._planet_sprite_cache[key]=z
        return z

    @staticmethod
    def _blit_masked(dst:np.ndarray,sprite:np.ndarray,mask:np.ndarray,cx:int,cy:int):
        h,w=dst.shape[:2]; sh,sw=sprite.shape[:2]; x0=int(cx-sw//2); y0=int(cy-sh//2)
        dx0=max(0,x0);dy0=max(0,y0);dx1=min(w,x0+sw);dy1=min(h,y0+sh)
        if dx0>=dx1 or dy0>=dy1:return
        sx0=dx0-x0;sy0=dy0-y0;sx1=sx0+(dx1-dx0);sy1=sy0+(dy1-dy0)
        m=mask[sy0:sy1,sx0:sx1]; tile=dst[dy0:dy1,dx0:dx1]; src=sprite[sy0:sy1,sx0:sx1]
        tile[m]=src[m]

    def _sky(self,s,viewer_dt=0.):
        c=self.sky;w=max(20,c.winfo_width());h=max(20,c.winfo_height());self._prepare_sky_static(w,h)
        # Human observer panorama only. While grounded, level the display to the
        # contacted sphere's local normal while preserving Yar's real heading. A
        # body-frame 360x180 equirectangular projection turns a tilted great-circle
        # horizon into a sinusoid; that was the cyclic "warping" seen in Patch21.
        img=np.empty((h,w,3),dtype=np.uint8);img[:]=np.array([0,1,6],dtype=np.uint8)
        for d in range(-180,181,45):img[:,max(0,min(w-1,int(round((d+180)/360*(w-1)))))]=np.array([17,26,38],dtype=np.uint8)
        for d in (-45,0,45):img[max(0,min(h-1,int(round((90-d)/180*(h-1))))),:]=np.array([17,26,38],dtype=np.uint8)

        ld=s.get('view_log',s.get('log',{})) or {}
        pos=np.asarray(ld.get('position',[0,0,0]),dtype=float)
        vel=np.asarray(ld.get('velocity',[0,0,0]),dtype=float)
        cam=np.asarray(s.get('eye_position',pos),dtype=float).copy()
        ori=np.asarray(s.get('orientation',[1,0,0,0]),dtype=float)
        delta_q=_viewer_delta_quaternion(s.get('angular_velocity',[0,0,0]),viewer_dt) if viewer_dt>0 else None
        ori_eff=q_normalize(q_mul(ori,delta_q)) if delta_q is not None else q_normalize(ori)

        render_bodies=list(s.get('render_bodies',[]))
        surface_normal=None; surface_rb=None
        if bool(ld.get('on_surface',False)):
            bid=str(ld.get('body_id'))
            for rb in render_bodies:
                if str(rb.get('id'))==bid and 'center' in rb and 'radius' in rb:
                    ctr=np.asarray(rb['center'],dtype=float);rad=max(1e-9,float(rb['radius']))
                    # IMPORTANT: extrapolate *along* the sphere, then level against the
                    # extrapolated normal. Linear tangent extrapolation was the residual
                    # Patch22 horizon-wave bug.
                    surface_normal=(pos-ctr)/max(float(np.linalg.norm(pos-ctr)),1e-12)
                    surface_rb=rb
                    break
        elif viewer_dt>0:
            cam=cam+vel*float(viewer_dt)
        basis=_observer_level_basis_world(ori_eff,surface_normal)

        # Distant worlds are cheap sprites, but draw them BEFORE true near-sphere
        # geometry. The nearby surface will therefore occlude them instead of letting
        # planets show through the ground as Patch21 did.
        near=[]
        for rb in sorted(render_bodies,key=lambda z:float(z.get('distance',0.)),reverse=True):
            ar=max(0.08,float(rb.get('angular_radius_deg',0.08)))
            if ar>=18.0 and 'center' in rb and 'radius' in rb:
                near.append(rb);continue
            if 'center' in rb:
                rel=np.asarray(rb['center'],dtype=float)-cam;dn=float(np.linalg.norm(rel))
                if dn<=1e-9:continue
                yaw,elev=_observer_project_world((rel/dn).reshape(1,3),basis);yaw=float(yaw[0]);elev=float(elev[0])
            else:
                yaw=float(rb['yaw']);elev=float(rb['elev'])
            xp0=int(round((yaw+180.)/360.*(w-1)));yp0=int(round((90.-elev)/180.*(h-1)))
            rpx=max(2,int(round(max(ar/360.*w,ar/180.*h))))
            # Orthographic approximation for small disks, oriented in world space.
            yr=math.radians(yaw);er=math.radians(elev)
            direction=np.array([math.cos(er)*math.cos(yr),math.cos(er)*math.sin(yr),math.sin(er)])
            right=np.array([-math.sin(yr),math.cos(yr),0.])
            up=np.cross(direction,right)
            sphere_basis=basis@np.column_stack((right,up,-direction))
            sprite,smask=_planet_sprite_array(rb['id'],2*rpx+1,rb.get('render_kind','planet'),
                albedo=float(rb.get('albedo',.5)),surface_basis=sphere_basis)
            self._blit_masked(img,sprite,smask,xp0,yp0)

        # True sphere geometry and depth for large nearby worlds.  Patch22 originally
        # ray-cast at the full Tk canvas resolution on every viewer frame.  A typical
        # 720x450 panel means 324,000 rays per sphere and, on the reference machine,
        # about 0.20 s of CPU per frame -- effectively a hard 5 fps ceiling and enough
        # memory-bandwidth/GIL pressure to make the otherwise independent worker look
        # frozen.  Render sphere geometry on an observer-only ~80k-pixel buffer, then
        # scale that result to the canvas.  This cannot affect Yar: only the human sky
        # raster and its occlusion depth use this buffer.
        depth=np.full((h,w),np.inf,dtype=np.float64)
        ray_cam=cam.copy()
        if near:
            max_surface_pixels=50000
            pix=w*h
            if pix>max_surface_pixels:
                q=math.sqrt(max_surface_pixels/float(pix))
                rw=max(90,int(round(w*q))); rh=max(56,int(round(h*q)))
            else:
                rw,rh=w,h
            rdepth=np.full((rh,rw),np.inf,dtype=np.float64)
            rimg=np.zeros((rh,rw,3),dtype=np.uint8)
            rhit=np.zeros((rh,rw),dtype=bool)
            for rb in near:
                ctr=np.asarray(rb['center'],dtype=float);rad=max(1e-9,float(rb['radius']))
                mask,dep,col=_observer_raycast_sphere(rw,rh,ray_cam,ori_eff,ctr,rad,str(rb['id']),rb.get('render_kind','planet'),None,basis,float(rb.get('albedo',.5)))
                take=mask&(dep<rdepth)
                if np.any(take):
                    rimg[take]=col[take];rdepth[take]=dep[take];rhit[take]=True

            if rw==w and rh==h:
                if np.any(rhit):img[rhit]=rimg[rhit];depth[rhit]=rdepth[rhit]
            elif np.any(rhit):
                # Dependency-free nearest resample. At the normal Observatory size
                # this is ~2x in each dimension, so it is visually smooth enough for
                # the cosmetic world texture while preserving a physically exact
                # ray/sphere horizon at the coarse sample locations.
                xi=np.minimum(rw-1,(np.arange(w,dtype=np.int64)*rw)//w)
                yi=np.minimum(rh-1,(np.arange(h,dtype=np.int64)*rh)//h)
                uh=rhit[yi[:,None],xi[None,:]]
                if np.any(uh):
                    uc=rimg[yi[:,None],xi[None,:]]
                    ud=rdepth[yi[:,None],xi[None,:]]
                    img[uh]=uc[uh];depth[uh]=ud[uh]

        # Stars and food/resources are point sources. Use the ray-cast surface depth
        # so a star/planet behind the local ground cannot bleed through, while a food
        # patch physically above the near surface can still appear in front of it.
        stars=s.get('stars',[]);n=len(stars)
        if n:
            yaw=np.empty(n,dtype=np.float64);elev=np.empty(n,dtype=np.float64);pdist=np.full(n,np.inf,dtype=np.float64)
            have_world=np.asarray([len(v)>=7 for v in stars],dtype=bool)
            if np.any(have_world):
                ids=np.flatnonzero(have_world);world=np.asarray([stars[i][6] for i in ids],dtype=np.float64)
                rel=world-cam[None,:];dn=np.linalg.norm(rel,axis=1);ok=dn>1e-9
                dirs=np.zeros_like(rel);dirs[ok]=rel[ok]/dn[ok,None]
                yy,ee=_observer_project_world(dirs,basis);yaw[ids]=yy;elev[ids]=ee;pdist[ids]=dn
            if np.any(~have_world):
                ids=np.flatnonzero(~have_world);yaw[ids]=[float(stars[i][0]) for i in ids];elev[ids]=[float(stars[i][1]) for i in ids]
                if delta_q is not None:
                    yr=np.radians(yaw[ids]);er=np.radians(elev[ids]);ce=np.cos(er)
                    vec=np.column_stack((ce*np.cos(yr),ce*np.sin(yr),np.sin(er)));vec=q_inverse_rotate(delta_q,vec)
                    yaw[ids]=np.degrees(np.arctan2(vec[:,1],vec[:,0]));elev[ids]=np.degrees(np.arctan2(vec[:,2],np.hypot(vec[:,0],vec[:,1])))
            app=np.fromiter((float(v[2]) for v in stars),dtype=np.float64,count=n);temp=np.fromiter((float(v[3]) for v in stars),dtype=np.float64,count=n)
            xp=np.rint((yaw+180.)/360.*(w-1)).astype(np.int32);yp=np.rint((90.-elev)/180.*(h-1)).astype(np.int32)
            kinds=np.asarray([v[5] for v in stars],dtype=object);resource=kinds=='resource';body=kinds=='body';star=~(resource|body)
            radius=np.ones(n,dtype=np.float64);radius[resource]=3.;radius[body]=np.clip(1.+np.log10(np.maximum(app[body],1e-5)+1.)*2.,1.5,5.);radius[star]=np.clip(1.+np.log10(np.maximum(app[star],1e-5)+1.)*1.5,1.,4.)
            colors=np.empty((n,3),dtype=np.uint8);colors[resource]=np.array([140,255,120],np.uint8);colors[body]=np.array([111,168,216],np.uint8)
            if np.any(star):
                t=np.clip(temp[star],1000.,40000.)/100.;rr=np.empty(len(t));gg=np.empty(len(t));bb=np.empty(len(t));lo=t<=66.
                rr[lo]=255.;gg[lo]=99.4708025861*np.log(t[lo])-161.1195681661;bb[lo]=0.;mid=lo&(t>19.);bb[mid]=138.5177312231*np.log(t[mid]-10.)-305.0447927307
                hi=~lo;rr[hi]=329.698727446*np.power(t[hi]-60.,-0.1332047592);gg[hi]=288.1221695283*np.power(t[hi]-60.,-0.0755148492);bb[hi]=255.;colors[star]=np.clip(np.column_stack((rr,gg,bb)),0,255).astype(np.uint8)
            r2=radius*radius
            for dy in range(-5,6):
                for dx in range(-5,6):
                    m=((dx*dx+dy*dy)<=r2)&(~body)
                    if not np.any(m):continue
                    ii=np.flatnonzero(m);xx=xp[ii]+dx;yy=yp[ii]+dy;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
                    if not np.any(inside):continue
                    ii=ii[inside];xx=xx[inside];yy=yy[inside]
                    # Eye-height ray depth and source-center distance differ by only a
                    # tiny amount; a small tolerance keeps surface resources visible.
                    front=(~np.isfinite(depth[yy,xx]))|(pdist[ii] < depth[yy,xx]+0.02)
                    if np.any(front):img[yy[front],xx[front]]=colors[ii[front]]

        cx=w//2;cy=h//2;img[max(0,cy):min(h,cy+1),max(0,cx-10):min(w,cx+11)]=np.array([102,255,170],np.uint8);img[max(0,cy-10):min(h,cy+11),max(0,cx):min(w,cx+1)]=np.array([102,255,170],np.uint8)
        self._photo_rgb(c,img,'sky')
        # Human-readable planet labels are observer-only overlays.  Anchor them to the
        # apparent south edge of each planet; where the geographic south pole projects
        # on-screen, prefer that exact point. Aliases replace the primary designation
        # while the immutable procedural ID moves underneath in smaller text.
        c.delete('world-label')
        for rb in render_bodies:
            bid=str(rb.get('id'));alias=str(rb.get('alias','')).strip();name=alias or bid
            ctr=np.asarray(rb.get('center',[0,0,0]),dtype=float);rad=max(1e-9,float(rb.get('radius',1.0)))
            south=ctr+np.array([0.0,0.0,-rad],dtype=float);rel=south-cam;dn=float(np.linalg.norm(rel))
            sx=sy=None
            if dn>1e-9:
                yy,ee=_observer_project_world((rel/dn).reshape(1,3),basis);sx=(float(yy[0])+180.)/360.*(w-1);sy=(90.-float(ee[0]))/180.*(h-1)
            if sx is None or sy is None or sx<4 or sx>w-4 or sy<15 or sy>h-18:
                relc=ctr-cam;dc=float(np.linalg.norm(relc))
                if dc<=1e-9:continue
                yy,ee=_observer_project_world((relc/dc).reshape(1,3),basis);cxp=(float(yy[0])+180.)/360.*(w-1);cyp=(90.-float(ee[0]))/180.*(h-1)
                ar=max(0.08,float(rb.get('angular_radius_deg',0.08)));rpx=max(ar/360.*w,ar/180.*h)
                sx=cxp;sy=cyp+rpx+3
            sx=max(8,min(w-8,float(sx)));sy=max(18,min(h-18,float(sy)))
            primary=c.create_text(sx,sy,text=name,fill='#f2f6fb',anchor='s',font=('Consolas',9,'bold'),tags=('world-label',))
            c.tag_bind(primary,'<Button-1>',lambda ev,z=bid:self._edit_world_alias(z))
            if alias:
                sub=c.create_text(sx,sy+2,text=bid,fill='#a7b8c8',anchor='n',font=('Consolas',7),tags=('world-label',))
                c.tag_bind(sub,'<Button-1>',lambda ev,z=bid:self._edit_world_alias(z))
        c.tag_raise('world-label');c.tag_raise('sky-status')
        mode=' observer extrapolation' if viewer_dt>1e-6 else ''
        frame=' level-surface' if surface_normal is not None else ' body-frame'
        c.itemconfigure(self._sky_status_item,text=f"W{s['generation']:03d}  eye={s.get('view_log',s['log'])['time_s']:.2f}s state={s['log']['time_s']:.2f}s  {s.get('sim_speed',0):.3f}× real-time  new frames {self._viewer_measured_fps:.2f}/s{mode}{frame} perf-ray")


    def _photo_rgb(self,c,rgb,key):
        h,w=rgb.shape[:2];ppm=f'P6\n{w} {h}\n255\n'.encode('ascii')+np.ascontiguousarray(rgb).tobytes()
        photo=self.tk.PhotoImage(data=ppm,format='PPM')
        item=self._eye_image_items.get(key)
        if item is None:
            item=c.create_image(0,0,anchor='nw',image=photo,tags=(f'{key}-raster',));self._eye_image_items[key]=item
        else:c.itemconfigure(item,image=photo);c.coords(item,0,0)
        self._eye_photos[key]=photo
        try:c.tag_lower(item)
        except Exception:pass

    def _raw_eye_raster(self,c,ids,response,yaw,elev,key):
        w=max(20,c.winfo_width());h=max(20,c.winfo_height())
        if not len(ids):
            self._photo_rgb(c,np.zeros((h,w,3),dtype=np.uint8),key);return
        yy=np.asarray(yaw[ids],float);ee=np.asarray(elev[ids],float)
        xp=((yy-np.min(yy))/max(1e-9,float(np.ptp(yy)))*(w-16)+8).astype(np.int32)
        yp=((1-(ee-np.min(ee))/max(1e-9,float(np.ptp(ee))))*(h-16)+8).astype(np.int32)
        img=_eye_raster(w,h,xp,yp,np.asarray(response[ids],float),blur_passes=1)
        self._photo_rgb(c,img,key)

    def _prepare_vision_overlay(self,w,h,column_yaw_deg=None):
        # In the packed-eye presentation, x is receptor-column order rather than a
        # window-stretched angular axis.  Put degree guides at the nearest real sensor
        # column so the observer labels remain honest while resize cannot open gaps.
        yaw_key=() if column_yaw_deg is None else tuple(np.round(np.asarray(column_yaw_deg,float),4).tolist())
        size=(int(w),int(h),yaw_key);c=self.vision
        if self._vision_overlay_size==size:return
        self._vision_overlay_size=size;c.delete('vision-overlay')
        pitch=max(1,int(getattr(self,'_vision_column_pitch_px',3)))
        margin=max(0,int(getattr(self,'_vision_margin_px',6)))
        cols=np.asarray(column_yaw_deg if column_yaw_deg is not None else [],dtype=float)
        def guide_x(deg):
            if cols.size:
                i=int(np.argmin(np.abs(cols-float(deg))))
                return float(margin+i*pitch+pitch//2)
            return float((deg+135.)/270.*w)
        x0=guide_x(0.0)
        c.create_line(x0,0,x0,h,fill='#263445',tags=('vision-overlay',));c.create_text(x0,7,text='FORWARD 0°',fill='#6e879e',anchor='n',font=('Consolas',7),tags=('vision-overlay',))
        for deg in (-135,-90,-45,45,90,135):
            x=guide_x(deg);c.create_line(x,0,x,h,fill='#111a26',tags=('vision-overlay',));c.create_text(x,h-2,text=f'{deg:+d}°',fill='#536577',anchor='s',font=('Consolas',7),tags=('vision-overlay',))
        c.create_text(7,7,text='',fill='#b9cce0',anchor='nw',font=('Consolas',8),tags=('vision-overlay','vision-cal'))
        c.tag_raise('vision-overlay')

    def _eyes(self,s):
        nl=int(s['eye_n_left']);r=np.asarray(s['facet_response'],float);y=np.asarray(s['eye_yaw_rad'],float);e=np.asarray(s['eye_elevation_rad'],float)
        self._raw_eye_raster(self.eye_l,np.arange(nl,dtype=np.int32),r,y,e,'eye-l')
        self._raw_eye_raster(self.eye_r,np.arange(nl,len(r),dtype=np.int32),r,y,e,'eye-r')
        c=self.vision;w=max(20,c.winfo_width());h=max(20,c.winfo_height())
        conf=np.asarray(s.get('eye_facet_confidence',np.ones(len(r))),float);yawd=np.degrees(y);elevd=np.degrees(e)
        edge=np.maximum(0.,np.abs(yawd)-135.);plot_yaw=np.clip(yawd,-135.,135.)

        # Pack the actual horizontal receptor columns at a fixed pixel pitch instead
        # of stretching their angular x coordinates across the current widget width.
        # Window growth therefore enlarges only the fly-model pane; it cannot create
        # elastic black gaps inside the eye representation.
        rounded=np.round(plot_yaw,6)
        columns,inverse=np.unique(rounded,return_inverse=True)
        pitch=max(1,int(getattr(self,'_vision_column_pitch_px',3)))
        margin=max(0,int(getattr(self,'_vision_margin_px',6)))
        packed_w=max(20,2*margin+max(1,len(columns))*pitch)
        if int(c.cget('width'))!=packed_w:
            c.configure(width=packed_w)
            w=packed_w
            self._vision_overlay_size=None
        xp=(margin+inverse*pitch+pitch//2).astype(np.int32)
        yp=((90.-np.clip(elevd,-90.,90.))/180.*(h-1)).astype(np.int32)
        certainty=np.clip(conf,0.,1.)*(1.-.55*np.minimum(1.,edge/20.))
        img=_eye_raster(w,h,xp,yp,r,certainty,blur_passes=2);self._photo_rgb(c,img,'vision')
        self._prepare_vision_overlay(w,h,columns)
        c.tag_raise('vision-overlay')
        cal=s.get('eye_calibration',{}) or {};mode=cal.get('mapping_mode','?');off=cal.get('official_workbook_loaded',False)
        rt=s.get('eye_runtime',{}) or {}; c.itemconfigure('vision-cal',text=f"{mode} • official columns={'yes' if off else 'fallback'} • overlap≈{cal.get('frontal_binocular_overlap_deg',0):.0f}° • FWHM≈{rt.get('effective_fwhm_mean_deg',0):.1f}° • retinal×{rt.get('retinal_subsamples',1)} • dither={rt.get('dither_sensor_count',0)}")


    @staticmethod
    def _attitude_euler_deg(q):
        """Observer-only quaternion -> roll/pitch/yaw in degrees."""
        q=np.asarray(q,dtype=float).reshape(-1)
        if len(q)!=4:return (0.0,0.0,0.0)
        w,x,y,z=[float(v) for v in q]
        sinr=2.0*(w*x+y*z); cosr=1.0-2.0*(x*x+y*y)
        roll=math.atan2(sinr,cosr)
        sinp=2.0*(w*y-z*x)
        pitch=math.copysign(math.pi/2.0,sinp) if abs(sinp)>=1.0 else math.asin(sinp)
        siny=2.0*(w*z+x*y); cosy=1.0-2.0*(y*y+z*z)
        yaw=math.atan2(siny,cosy)
        return tuple(math.degrees(v) for v in (roll,pitch,yaw))

    def _fly_model(self,s):
        """Render Yar's current body/motor state without inventing behavior.

        This is deliberately an observer rendering.  Body attitude and velocity are
        taken from the physical state; limb/head/proboscis deflections come only from
        the current motor-effector readout.  Wingbeat phase is *not* available, so the
        wings do not flap cosmetically: their brightness/width represents actual power
        output instead.
        """
        c=self.fly_model
        try:
            w=max(80,int(c.winfo_width()));h=max(60,int(c.winfo_height()))
        except Exception:return
        c.delete('all')
        cx,cy=0.50*w,0.53*h
        scale=max(11.0,min(w/13.0,h/5.4))
        eff=s.get('motor_effectors',{}) or {}
        q=np.asarray(s.get('orientation',[1.,0.,0.,0.]),dtype=float)
        roll,pitch,yaw=self._attitude_euler_deg(q)
        log=s.get('log',{}) or {}
        vel=np.asarray(log.get('velocity',[0.,0.,0.]),dtype=float)
        speed=float(np.linalg.norm(vel))

        # Fixed observer camera: world X projects right, world Z projects up, with
        # world Y providing a shallow depth skew.  Actual body quaternion rotates the
        # model into this frame; no gaze/attention estimate is involved.
        def world_point(body_xyz):
            v=q_rotate(q,np.asarray(body_xyz,dtype=float))
            return np.asarray(v,dtype=float)
        def proj(body_xyz):
            xw,yw,zw=world_point(body_xyz)
            return (cx+scale*(xw+0.34*yw), cy-scale*(zw+0.20*yw))
        def line(points,**kw):
            xy=[]
            for p0 in points:xy.extend(proj(p0))
            return c.create_line(*xy,**kw)
        def poly(points,**kw):
            xy=[]
            for p0 in points:xy.extend(proj(p0))
            return c.create_polygon(*xy,**kw)
        def activity_color(v,dim='#263445',hot='#e8f3ff'):
            # Tk has no alpha; interpolate RGB for observer brightness only.
            a=max(0.0,min(1.0,abs(float(v))))
            def rgb(x):return tuple(int(x[i:i+2],16) for i in (1,3,5))
            d=rgb(dim);hh=rgb(hot);z=tuple(int(d[i]+a*(hh[i]-d[i])) for i in range(3))
            return '#%02x%02x%02x'%z

        wl=float(eff.get('wing_left',0.0));wr=float(eff.get('wing_right',0.0))
        fl=float(eff.get('foreleg_left',0.0));fr=float(eff.get('foreleg_right',0.0))
        ml=float(eff.get('midleg_left',0.0));mr=float(eff.get('midleg_right',0.0))
        hl=float(eff.get('hindleg_left',0.0));hr=float(eff.get('hindleg_right',0.0))
        hy=float(eff.get('head_yaw',0.0));hp=float(eff.get('head_pitch',0.0))
        antl=float(eff.get('antenna_left',0.0));antr=float(eff.get('antenna_right',0.0))
        abdomen=float(eff.get('abdomen_bend',0.0));prob=float(eff.get('proboscis_extension',0.0))
        pump=float(eff.get('pharyngeal_pump',0.0));fill=float(eff.get('cibarium_fill',0.0))
        hal_l=float(eff.get('haltere_left',0.0));hal_r=float(eff.get('haltere_right',0.0))

        # Wings: actual phase is unavailable, so pose stays neutral and only real
        # power activity changes their visual intensity/thickness.
        for side,powv in ((-1,wl),(+1,wr)):
            sy=float(side)
            wingpts=[(0.25,0.22*sy,0.08),(-0.15,1.45*sy,0.18),(-1.05,1.68*sy,0.10),(-0.55,0.48*sy,0.02)]
            poly(wingpts,fill='#0b111b',outline=activity_color(powv,'#263445','#b9e9ff'),width=max(1,int(1+2.5*powv)))

        # Legs: signed motor output changes only the displayed limb deflection.  It is
        # a motor-effector pose, not a claim of reconstructed joint-angle kinematics.
        leg_specs=[
            ('F',0.42,0.40,fl,fr),('M',-0.05,0.54,ml,mr),('H',-0.52,0.44,hl,hr)
        ]
        for _name,x0,y0,lv,rv in leg_specs:
            for side,v in ((-1,lv),(+1,rv)):
                sy=float(side);sweep=0.48*float(v)
                hip=(x0,0.26*sy,-0.06)
                knee=(x0+0.28+sweep,0.72*sy,-0.20)
                foot=(x0+0.58+0.80*sweep,1.15*sy,-0.32)
                line([hip,knee,foot],fill=activity_color(v,'#4a3b31','#ffd6a0'),width=max(1,int(1+2*abs(v))),smooth=True)

        # Abdomen/thorax/head axis.
        body_outline='#b67442'
        line([(-1.28,-0.04,0.0),(-0.55,0.0,0.0),(0.18,0.0,0.04),(0.82,0.0,0.08)],fill=body_outline,width=max(3,int(scale*0.20)),smooth=True)
        # Abdomen bend is an actual effector signal; render a short bent terminal.
        line([(-0.88,0.0,0.0),(-1.45,0.38*abdomen,-0.03)],fill='#8c4c2c',width=max(3,int(scale*0.16)))
        hx=0.92;hyoff=0.20*hy;hz=0.12+0.16*hp
        head=proj((hx,hyoff,hz));rhead=max(4,int(scale*0.24))
        c.create_oval(head[0]-rhead,head[1]-rhead,head[0]+rhead,head[1]+rhead,fill='#7c3f2a',outline='#d47755',width=1)
        # Compound eyes on the external model are observer decoration tied to the head,
        # not Yar's visual response/attention.
        for side in (-1,+1):
            ep=proj((1.02,hyoff+0.17*side,hz+0.03));re=max(2,int(scale*0.09))
            c.create_oval(ep[0]-re,ep[1]-re,ep[0]+re,ep[1]+re,fill='#8c1838',outline='#d14b6a')

        # Antennae and proboscis from real effectors.
        for side,av in ((-1,antl),(+1,antr)):
            sy=float(side);base=(1.06,hyoff+0.10*sy,hz+0.10);tip=(1.35,hyoff+(0.22+0.15*av)*sy,hz+0.24+0.08*av)
            line([base,tip],fill=activity_color(av,'#584937','#ffe2ae'),width=1)
        p0=(1.08,hyoff,hz-0.08);p1=(1.08+0.72*max(0.0,prob),hyoff,hz-0.20-0.10*max(0.0,prob))
        line([p0,p1],fill=activity_color(prob,'#61452f','#fff1b0'),width=max(1,int(1+3*max(0.0,prob))))

        # Halteres: activity brightens the stalk/knob without inventing oscillation.
        for side,hv in ((-1,hal_l),(+1,hal_r)):
            sy=float(side);a=(-0.15,0.32*sy,-0.02);b=(-0.42,0.60*sy,-0.05)
            col=activity_color(hv,'#36404a','#d8f5ff');line([a,b],fill=col,width=1)
            bx,by=proj(b);rr=max(2,int(scale*0.055));c.create_oval(bx-rr,by-rr,bx+rr,by+rr,fill=col,outline=col)

        # Physical heading and velocity are separate observer vectors.  They are drawn
        # in screen coordinates from the real world vectors; neither is fed back.
        forward=q_rotate(q,np.array([1.,0.,0.],dtype=float))
        def vec2(v):
            v=np.asarray(v,dtype=float);return np.array([v[0]+0.34*v[1],-(v[2]+0.20*v[1])],dtype=float)
        def arrow(v,col,label,mag=30.0):
            vv=vec2(v);n=float(np.linalg.norm(vv))
            if n<1e-9:return
            vv=vv/n*mag
            x0,y0=50.0,h-18.0;x1,y1=x0+vv[0],y0+vv[1]
            c.create_line(x0,y0,x1,y1,fill=col,width=2,arrow='last')
            c.create_text(x1+3,y1,text=label,fill=col,anchor='w',font=('Consolas',7,'bold'))
        arrow(forward,'#4da3ff','body',28.0)
        if speed>1e-6:arrow(vel,'#66f6ff','vel',34.0)

        # Compact, truthful telemetry overlays.
        c.create_text(6,5,text='LIVE BODY • observer-only motor pose',fill='#a9bed5',anchor='nw',font=('Consolas',8,'bold'))
        c.create_text(w-6,5,text=f'R/P/Y {roll:+.0f}° {pitch:+.0f}° {yaw:+.0f}°  v={speed:.2f}',fill='#8397aa',anchor='ne',font=('Consolas',7))
        c.create_text(w-6,h-5,text=f'wing {wl:.2f}/{wr:.2f}  walk {float(log.get("walking_command",0.0)):.2f}  prob {prob:.2f}  pump {pump:.2f}  fill {fill:.2f}',fill='#8397aa',anchor='se',font=('Consolas',7))


    def _summary(self,s):
        l=s['log'];h=s['homeostasis'];m=s.get('memory',{}) or {};pos=l.get('position',[0]*3);vel=l.get('velocity',[0]*3)
        last=s.get('last_growth_event')
        if last:
            grow_line=f"W{int(last.get('working_generation',0)):03d}: {last.get('parent_body')} {last.get('parent_type')} → {last.get('daughter_body')} (+{last.get('new_edges',0)} edges)"
        elif s.get('lineage'):
            r=s['lineage'][-1];grow_line=f"W{int(r.get('working_generation',0)):03d}: {r.get('parent_body')} {r.get('parent_type')} → {r.get('daughter_body')} (+{r.get('new_edges',0)} edges)"
        else:grow_line='none yet'
        txt=(f"{s.get('specimen','Specimen')} — FULL LIFE LINEAGE\nworking graph  {s['graph']}\nworking gen    W{s['generation']:03d} (working, not canonical G)\n"
             f"age            {l['time_s']:10.2f} s\nneurons        {s['neurons']:,}\nedges          {s['edges']:,}\n\n"
             f"NEURAL / LEARNING HIGHLIGHTS\nnetwork        {l.get('network_hz',0):9.3f} Hz\nnovelty        {l.get('novelty_hz',0):9.3f} Hz\n"
             f"slow/state wave {100.0*l.get('keepalive_wave_phase',0):8.2f}% ER5+ExR1  {int(l.get('keepalive_wave_active_neurons',0)):6,d} cells  peak ±{l.get('keepalive_wave_peak_mv',0):.3f} mV\n"
             f"motor mean     {l.get('motor_mean_hz',0):9.3f} Hz\n"
             f"learned edges  {l.get('modified_edges',m.get('modified_edges',0)):9,d}\nfamiliar cells {int(m.get('familiar_neurons',0)):9,d}\nmean |Δw|      {float(m.get('mean_abs_delta_modified',0.0)):.6f} modified\nmax |Δw|       {float(m.get('max_abs_delta',0.0)):.6f}\n"
             f"DA/OA/5HT      {l.get('dopamine_mean',0):.5f} / {l.get('octopamine_mean',0):.5f} / {l.get('serotonin_mean',0):.5f}\n"
             f"growth pressure{s['max_growth_pressure']:8d}/{s['growth_threshold']} epochs\ngrowth cooldown{s.get('growth_cooldown_remaining',0):7d}/{s.get('growth_cooldown_total',0)} epochs\nlatest growth  {grow_line}\n\n"
             f"LIFE STATE\nsatiety        {l.get('satiety',h.get('satiety',0)):.4f}\nnutr. hunger   {l.get('nutritional_hunger',h.get('nutritional_hunger',l.get('hunger',0))):.4f}\nexpressed      {l.get('hunger',h.get('hunger',0)):.4f}  suppress {l.get('hunger_suppression',h.get('hunger_suppression',0)):.3f}\nrelief/pang    {l.get('approach_relief',h.get('approach_relief',0)):.3f} / {l.get('hunger_pang',h.get('hunger_pang',1)):.3f}\nenergy reserve {l.get('energy',h.get('energy',0)):.4f}  motor power {l.get('motor_energy_scale',1.0):.3f}\n"
             f"thermal        {l.get('thermal_c',0):7.3f} °C  error {l.get('thermal_error',0):.4f}\nhomeo error    {l.get('homeostatic_error',0):.4f}\nhost reward    DISABLED\n\n"
             f"flight         {l.get('flight_command',0):.3f}  takeoff {l.get('takeoff_command',0):.3f}  walk {l.get('walking_command',0):.3f}  feed {l.get('feeding_command',0):.3f}\n"
             f"surface        {l.get('on_surface',False)}  contact={l.get('body_id')}  altitude {self.engine._altitude_display(l)}  atm={100.0*float(l.get('atmosphere_fraction',0.0)):.1f}%\nresource       {l.get('nearest_resource_id')}  d={l.get('nearest_resource_distance',float('inf')):.3f}\n"
             f"odor/foot/mouth {l.get('odor',0):.3f} / {l.get('resource_contact',False)} / {l.get('mouth_contact',False)}   eaten={l.get('nutrient_consumed',0):.4f}\n\n"
             f"position       {pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}\nvelocity       {vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}\nSTATUS: {s['status']}\n")
        self.tele.delete('1.0','end');self.tele.insert('1.0',txt)

    def _performance(self,s):
        ti,one,best,gain=self._thread_perf(s);gpu=s.get('gpu',{}) or {};core=ti.get('last_epoch_wall_s');step=s.get('wall_step_s');save=s.get('last_save_wall_s',0.);eff=s.get('effective_sim_speed',s.get('sim_speed',0.))
        candidates=' → '.join(str(x) for x in ti.get('candidates',[])) or '?'
        coarse=' → '.join(str(x) for x in ti.get('coarse_probes',[])) or '?'
        bracket=ti.get('search_bracket') or ['?','?'];regression=ti.get('first_regression_threads');anchor=ti.get('refinement_anchor_threads')
        tuner='not available'
        if one is not None and best is not None and gain is not None:tuner=f"1T={float(one):.4f}s  best={float(best):.4f}s  speedup≈{gain:.2f}×"
        gprobe='not run'
        if gpu.get('gpu_probe_s') is not None:gprobe=f"CPU est {float(gpu.get('cpu_probe_s') or 0):.4f}s → GPU est {float(gpu.get('gpu_probe_s')):.4f}s / full forgetting pass"
        timing_dt=float(s.get('window_world_dt',s.get('slice_world_dt',self.worker.world_dt)))
        last_gpu_pass = 'n/a' if gpu.get('last_forget_s') is None else f"{float(gpu.get('last_forget_s')):.4f}s"
        txt=(f"HOST / ACCELERATION\nAutoThreads    {ti.get('threads',1)}/{ti.get('permitted',1)} workers  mode={ti.get('mode','?')}\nbackend        {ti.get('backend','?')}\n"
             f"coarse trail   {coarse}\nsearch bracket {bracket[0]}..{bracket[1]} workers\nprobe trail    {candidates}\nfirst slow     {regression if regression is not None else 'none before runtime cap'}\nrefine anchor  {anchor if anchor is not None else '?'}\nselected       {ti.get('best_threads','?')} workers\ntuner score    {tuner}\nstrategy       {ti.get('tuning_strategy','?')}\n\n"
             f"GPU HYBRID\nrequested      {gpu.get('requested',False)}\nenabled        {gpu.get('enabled',False)}\ndevice         {gpu.get('device_name') or 'none'}\nneural GPU     {gpu.get('neural_enabled',False)}\nneural status  {gpu.get('neural_reason','not requested')}\nexact probe    {gpu.get('exact_probe')}\nprobe          {gprobe}\nreason         {gpu.get('reason','?')}\nscope          {gpu.get('scope','?')}\nlast GPU pass  {last_gpu_pass}\n\n"
             f"UNIVERSE STATE\ncached sectors {int(s.get('universe_cached_sectors',0))}/{int(getattr(self.engine.exp.universe,'BODY_CACHE_MAX_SECTORS',0) or 0)}\nresource deltas {int(s.get('universe_resource_overrides',0))} persisted modified-only\ndynamic objects {int(s.get('universe_dynamic_resources',0))}\n\n"
             f"LIVE TIMING\nneural core    {'n/a' if core is None else f'{float(core):.4f} s / neural slice'}\nclosed-loop    {'n/a' if step is None else f'{float(step):.4f} s / {timing_dt:.3f} sim s'}\nsim rate       {float(s.get('sim_speed',0.)):.4f}× real-time compute-only\neffective      {float(eff):.4f}× incl. checkpoint I/O\nlast save      {float(save):.4f} s    autosave every {self.worker.autosave_sim_s:g} sim s\nviewer         target {self._viewer_target_fps:g} fps / new frames {self._viewer_measured_fps:.2f}/s\nscheduler      slice {s.get('slice_index',1)}/{s.get('slice_count',1)}; learning {'pending' if s.get('learning_pending') else 'complete'}\nviewer rule    sky/eyes/map share one sampled pose; no future-position prediction\n")
        self.perf_text.delete('1.0','end');self.perf_text.insert('1.0',txt)

    def _history(self):
        c=self.hist;c.delete('all');w=max(20,c.winfo_width());h=max(20,c.winfo_height());series=[('net','#5ed6ff'),('motor','#79ff8f'),('novelty','#ffcc66'),('hunger','#ff7aa8')]
        vals=[v for k in self.history for v in self.history[k]];ym=max(1.,max(vals) if vals else 1.)
        for name,col in series:
            a=self.history[name]
            if len(a)<2:continue
            pts=[]
            for i,v in enumerate(a):pts += [6+i/max(1,len(a)-1)*(w-12),h-10-(v/ym)*(h-22)]
            c.create_line(*pts,fill=col,width=2)
        x=8
        for name,col in series:c.create_text(x,h-3,text=name if name!='hunger' else 'hunger×100',fill=col,anchor='sw',font=('Consolas',8));x+=95
    def _motor(self,s):
        for x in self.motor_tree.get_children():self.motor_tree.delete(x)
        resolved={'wm':'wings/flight','fl':'forelegs','ml':'midlegs','hl':'hindlegs','nm':'neck/head','hm':'halteres','ad':'abdomen','pm':'feeding','am':'antennae','rm':'UNRESOLVED—preserved','xm':'MNxm03→haltere; rest preserved'}
        for sub in sorted(s['motor_subclass']):
            d=s['motor_subclass'][sub];self.motor_tree.insert('', 'end', values=(sub,f"{d.get('L',0):.2f}",f"{d.get('R',0):.2f}",resolved.get(sub,'UNRESOLVED—preserved')))
        e=s['motor_effectors'];av=s['angular_velocity'];f=s['force_body'];t=s['torque_body'];un=s['motor_unresolved']
        active_un=[kv for kv in un.items() if kv[1].get('rate_hz',0)>0]
        top=sorted(active_un,key=lambda kv:kv[1].get('rate_hz',0),reverse=True)[:12]
        txt=(f"force XYZ      {f[0]:+.5f} {f[1]:+.5f} {f[2]:+.5f}\ntorque RPY     {t[0]:+.5f} {t[1]:+.5f} {t[2]:+.5f}\nangular vel    {av[0]:+.5f} {av[1]:+.5f} {av[2]:+.5f} rad/s\n\n"
             f"wings L/R      {e['wing_left']:.3f} / {e['wing_right']:.3f}\ntakeoff L/R    {e['takeoff_left']:.3f} / {e['takeoff_right']:.3f}\nhead yaw/pitch {e['head_yaw']:+.3f} / {e['head_pitch']:+.3f}\nantennae L/R   {e['antenna_left']:.3f} / {e['antenna_right']:.3f}\n"
             f"abdomen        {e['abdomen_bend']:+.3f}\nproboscis/pump {e['proboscis_extension']:+.3f} / {e['pharyngeal_pump']:.3f}  cibarium={e.get('cibarium_fill',0):.3f}\n"
             f"fore L/R       {e['foreleg_left']:+.3f} / {e['foreleg_right']:+.3f}\nmid L/R        {e['midleg_left']:+.3f} / {e['midleg_right']:+.3f}\nhind L/R       {e['hindleg_left']:+.3f} / {e['hindleg_right']:+.3f}\n\n"
             f"UNRESOLVED MOTOR CHANNELS (preserved, not discarded): {len(un)} total / {len(active_un)} active\n")
        for body,d in top:txt+=f"{body:>12} {d['subclass']:>3} {d['rate_hz']:7.2f} Hz  {d['type']} {d['side']}\n"
        self.motor_text.delete('1.0','end');self.motor_text.insert('1.0',txt)
    def _drop_care_package(self):
        # Queue the intervention onto the simulation worker so world state is never
        # mutated concurrently with a neural/mechanical epoch.  It also works paused.
        if self.worker.care_package_evt.is_set():
            return
        self.worker.care_package_evt.set()

    def _life(self,s,viewer_dt=0.,draw_world=True):
        l=dict(s['log']);l.update({k:v for k,v in s.get('view_log',{}).items() if k in ('position','velocity','on_surface','body_id')});txt=(f"VIEW eye t={s.get('view_log',s['log'])['time_s']:.2f}s | state t={s['log']['time_s']:.2f}s\nHOMEOSTASIS — bounded need; cue suppression is NOT nutritional reward\nSatiety {l.get('satiety',0):.4f}   Nutritional hunger {l.get('nutritional_hunger',l.get('hunger',0)):.4f}   Energy reserve {l.get('energy',0):.4f}\nMotor power available {l.get('motor_energy_scale',1.0):.3f}\n"
            f"Expressed hunger {l.get('hunger',0):.4f}   suppression {l.get('hunger_suppression',0):.3f}   anticipatory suppression {l.get('approach_relief',0):.3f}   pang {l.get('hunger_pang',1):.3f}\n"
            f"Thermal {l.get('thermal_c',0):.3f} °C   thermal error {l.get('thermal_error',0):.4f}\nTotal error {l.get('homeostatic_error',0):.4f}   host reinforcement DISABLED\n"
            f"Antennae L/R {l.get('antenna_left_c',0):.3f}/{l.get('antenna_right_c',0):.3f} °C   RH {100*l.get('relative_humidity',0):.1f}%  dry/moist={l.get('dry_sense',0):.2f}/{l.get('moist_sense',0):.2f}\n\n"
            f"Position {l.get('position',[0,0,0])[0]:+.4f}, {l.get('position',[0,0,0])[1]:+.4f}, {l.get('position',[0,0,0])[2]:+.4f}\n"
            f"Velocity {l.get('velocity',[0,0,0])[0]:+.4f}, {l.get('velocity',[0,0,0])[1]:+.4f}, {l.get('velocity',[0,0,0])[2]:+.4f}  speed={float(np.linalg.norm(np.asarray(l.get('velocity',[0,0,0]),float))):.4f}\n"
            f"Forward speed {l.get('forward_speed',0):+.4f}   sideslip {l.get('slip_angle_deg',0):+.2f} deg\n"
            f"Gravity -> {l.get('gravity_body_id')}   |g|={l.get('gravity_accel',0):.3f} local units/s^2\n"
            f"Surface {l.get('on_surface',False)}  contact body={l.get('body_id')}  altitude={self.engine._altitude_display(l)}  atmosphere={100.0*float(l.get('atmosphere_fraction',0.0)):.1f}%\n"
            f"Nearest nutrient {l.get('nearest_resource_id')}  d={l.get('nearest_resource_distance',float('inf')):.3f}\nOdor={l.get('odor',0):.3f} foot_contact={l.get('resource_contact',False)} mouth_contact={l.get('mouth_contact',False)} mouth_access={l.get('proboscis_contact',0):.3f} consumed step={l.get('nutrient_consumed',0):.5f} total={l.get('consumed_total',0):.4f}\nDivine fruit drops={l.get('divine_fruit_drops',0)} starvation timer={l.get('starvation_rescue_s',0):.1f}s\n"
            f"Flight={l.get('flight_command',0):.3f} takeoff={l.get('takeoff_command',0):.3f} walk={l.get('walking_command',0):.3f} feed={l.get('feeding_command',0):.3f}  proboscis={l.get('proboscis_extension',0):.3f} pump={l.get('pharyngeal_pump',0):.3f} fill={l.get('cibarium_fill',0):.3f}\n")
        self.life_text.delete('1.0','end');self.life_text.insert('1.0',txt)
        if draw_world:self._worldmap(s,viewer_dt=viewer_dt)
    def _worldmap(self,s,viewer_dt=0.):
        c=self.world;c.delete('all');w=max(20,c.winfo_width());h=max(20,c.winfo_height())
        auth_p=np.asarray(s.get('view_log',s['log']).get('position',[0,0,0]),float)
        p=auth_p.copy()
        viewer_dt=0.  # The Life map uses the same captured pose as the eye/sky.
        span=12.

        # Life Depth 1: a shallow tilted orthographic observer camera.  Spheres remain
        # circles under orthographic projection, while world Z now contributes visibly
        # to screen position and to draw depth.  No part of this projection is supplied
        # to Yar.
        tilt=math.radians(28.0);ct=math.cos(tilt);st=math.sin(tilt)

        if self._world_camera_center is None:
            self._world_camera_center=auth_p.copy()
        center=np.asarray(self._world_camera_center,float).copy()
        dead=0.55*span
        # Track all three coordinates with a dead-band.  Z used to be silently ignored,
        # making a fly pass above/below a world while the Life pane still painted it on
        # top.
        for ax in (0,1,2):
            d=float(auth_p[ax]-center[ax])
            if abs(d)>dead:
                center[ax]+=d-math.copysign(dead,d)
        self._world_camera_center=center

        def project(q):
            q=np.asarray(q,float);d=q-center
            sx=float(d[0])
            sy=float(d[1])*ct+float(d[2])*st
            depth=-float(d[1])*st+float(d[2])*ct  # larger = nearer observer
            return w/2+sx/span*w/2,h/2-sy/span*h/2,depth
        def xy(q):
            x,y,_=project(q);return x,y

        # Life Trail Depth 1: the retained history already contains full XYZ points.
        # Do not flatten it into one background polyline.  We clip each short segment
        # against the same circular sphere silhouettes used by the Life pane, then put
        # each visible piece back into the depth-sorted painter list below.
        draw=[]
        occluders=[]
        body_labels=[]
        for b in s['bodies']:
            ctr=np.asarray(b['center'],float);x,y,dcenter=project(ctr)
            # Orthographic projection preserves a sphere's circular silhouette.
            r=max(3,float(b['radius'])/span*w/2)
            front_depth=dcenter+float(b['radius'])
            temp=float(b['temp_c']);col='#ff8a65' if temp>33 else ('#73d0ff' if temp<21 else '#78d58b')
            diam=max(5,min(512,int(round(2*r+1))))
            key=(str(b['id']),diam,str(b.get('render_kind','planet')),'world-alpha-v1')
            photo=self._planet_tk_cache.get(key)
            if photo is None:
                spr,msk=_planet_sprite_array(str(b['id']),diam,str(b.get('render_kind','planet')))
                photo=self.tk.PhotoImage(data=_rgba_png_bytes(spr,msk),format='png')
                self._planet_tk_cache[key]=photo
            bid=str(b.get('id'));alias=str(b.get('alias','')).strip();name=alias or bid
            draw.append((front_depth,0,'body',(x,y,photo,bid,name,alias,r,temp,col)))
            occluders.append({
                'x':float(x),'y':float(y),'depth':float(dcenter),
                'front_depth':float(front_depth),'r_px':float(r),
                'radius_world':max(1e-9,float(b['radius'])),'body_id':bid,
            })
            # Surface resources need sphere-aware occlusion.  Sorting the icon by its
            # raw point depth makes the planet's front surface paint over nearly every
            # resource on the visible hemisphere.  Keep far-side food genuinely hidden,
            # but lift near-side surface food just above its parent sphere in painter
            # order so the icon remains visible at the correct projected position.
            view_forward=np.array([0.0,-st,ct],dtype=float)
            for rr in b['resources']:
                if rr['remaining']<=0:continue
                rpos=np.asarray(rr['position'],float)
                rx,ry,rd=project(rpos)
                rel=rpos-ctr
                near_side=float(np.dot(rel,view_forward))>=-1e-9
                rdraw=max(float(rd),front_depth+1e-6) if near_side else float(rd)
                draw.append((rdraw,1,'resource',(rx,ry,rr.get('icon','🍎'))))

        def _quadratic_roots(a,b,c0):
            eps=1e-12
            if abs(a)<eps:
                if abs(b)<eps:return []
                return [-c0/b]
            disc=b*b-4.0*a*c0
            if disc<0.0:return []
            sd=math.sqrt(max(0.0,disc))
            return [(-b-sd)/(2.0*a),(-b+sd)/(2.0*a)]

        def _trail_visible_pieces(q0,q1):
            # Return (x0,y0,x1,y1,depth_key) pieces.  Circle-entry roots split a
            # segment that merely passes behind a world; ellipsoid roots additionally
            # split genuine front/back surface crossings (e.g. an impact/landing).
            x0,y0,d0=project(q0);x1,y1,d1=project(q1)
            cuts=[0.0,1.0]
            for o in occluders:
                rr=max(1e-9,float(o['r_px']))
                ux=(x0-float(o['x']))/rr;uy=(y0-float(o['y']))/rr
                vx=(x1-x0)/rr;vy=(y1-y0)/rr
                a=vx*vx+vy*vy;b=2.0*(ux*vx+uy*vy);cc=ux*ux+uy*uy-1.0
                for t in _quadratic_roots(a,b,cc):
                    if 1e-9<t<1.0-1e-9:cuts.append(float(t))
                rz=max(1e-9,float(o['radius_world']))
                uz=(d0-float(o['depth']))/rz;vz=(d1-d0)/rz
                ae=a+vz*vz;be=b+2.0*uz*vz;ce=cc+uz*uz
                for t in _quadratic_roots(ae,be,ce):
                    if 1e-9<t<1.0-1e-9:cuts.append(float(t))
            cuts=sorted(set(round(t,12) for t in cuts))
            pieces=[]
            for ta,tb in zip(cuts[:-1],cuts[1:]):
                if tb-ta<1e-10:continue
                tm=0.5*(ta+tb)
                xm=x0+(x1-x0)*tm;ym=y0+(y1-y0)*tm;dm=d0+(d1-d0)*tm
                hidden=False;overlap_front=[]
                for o in occluders:
                    rr=max(1e-9,float(o['r_px']))
                    ux=(xm-float(o['x']))/rr;uy=(ym-float(o['y']))/rr
                    rho2=ux*ux+uy*uy
                    if rho2<=1.0+1e-10:
                        # Front surface depth at this point in the circular silhouette.
                        front=float(o['depth'])+float(o['radius_world'])*math.sqrt(max(0.0,1.0-min(1.0,rho2)))
                        if dm<front-1e-9:
                            hidden=True;break
                        overlap_front.append(float(o['front_depth']))
                if hidden:continue
                xa=x0+(x1-x0)*ta;ya=y0+(y1-y0)*ta;da=d0+(d1-d0)*ta
                xb=x0+(x1-x0)*tb;yb=y0+(y1-y0)*tb;db=d0+(d1-d0)*tb
                depth_key=0.5*(da+db)
                # A piece that is physically in front of a planet silhouette must be
                # painted after that opaque planet sprite.  Outside silhouettes, normal
                # camera depth is sufficient because there is no overlap to occlude.
                if overlap_front:depth_key=max(depth_key,max(overlap_front)+1e-6)
                pieces.append((xa,ya,xb,yb,depth_key))
            return pieces

        if len(self.path)>1:
            for q0,q1 in zip(self.path[:-1],self.path[1:]):
                for tx0,ty0,tx1,ty1,td in _trail_visible_pieces(q0,q1):
                    draw.append((td,1,'trail',(tx0,ty0,tx1,ty1)))

        # Yar participates in the same depth order as worlds.  If he is on the far side
        # of a sphere in this projection, the circular planet sprite can genuinely hide
        # him instead of the marker always being painted last.  When that happens, draw
        # a tiny top-layer x-ray ghost of the fly so the observer can still locate him
        # without lying about the occlusion.
        x,y,fly_depth=project(p)
        # Keep Yar in the same depth calculation as everything else, but do not let
        # painter order erase him.  We render him once, last, and use depth at his
        # projected coordinate to decide whether that final marker is normal or x-ray.
        fly_hidden=False; fly_occluder=None
        for o in occluders:
            rr=max(1e-9,float(o['r_px']))
            ux=(x-float(o['x']))/rr; uy=(y-float(o['y']))/rr
            rho2=ux*ux+uy*uy
            if rho2<=1.0+1e-10:
                front=float(o['depth'])+float(o['radius_world'])*math.sqrt(max(0.0,1.0-min(1.0,rho2)))
                if fly_depth<front-1e-9:
                    fly_hidden=True; fly_occluder=o; break

        def _draw_fly_marker(fx,fy,*,body_fill='#fff36b',wing_fill='#d7e8ff',heading_fill='#4da6ff',course_fill='#54f2f2',xray=False):
            fw=q_rotate(np.asarray(s.get('orientation',[1,0,0,0]),float),np.array([1.,0.,0.]))
            # Project body heading and velocity through the same tilted camera, not
            # the old XY-only map.
            fsx=float(fw[0]);fsy=float(fw[1])*ct+float(fw[2])*st
            hn=math.hypot(fsx,fsy);hx,hy=(1.,0.) if hn<1e-9 else (fsx/hn,fsy/hn)
            head_dash=(3,2) if xray else None
            c.create_line(fx,fy,fx+12*hx,fy-12*hy,fill=heading_fill,width=1,dash=head_dash)
            vv=np.asarray(s.get('view_log',s['log']).get('velocity',[0,0,0]),float)
            vsx=float(vv[0]);vsy=float(vv[1])*ct+float(vv[2])*st
            vn=math.hypot(vsx,vsy)
            if vn>1e-9:
                vx,vy=vsx/vn,vsy/vn
                c.create_line(fx,fy,fx+12*vx,fy-12*vy,fill=course_fill,width=1,dash=head_dash)
            px,py=-hy,hx
            if xray:
                # Reverse-xray: dark cut line under a neon ghost so the marker reads as
                # an occluded body rather than a normal visible fly.
                c.create_oval(fx-7,fy-7,fx+7,fy+7,outline='#d62bff',width=1,dash=(2,2))
                c.create_line(fx-5*hx,fy+5*hy,fx+5*hx,fy-5*hy,fill='#090c14',width=5)
                c.create_line(fx-1*hx+4*px,fy+1*hy-4*py,fx+2*hx,fy-2*hy,fill='#0d1220',width=3)
                c.create_line(fx-1*hx-4*px,fy+1*hy+4*py,fx+2*hx,fy-2*hy,fill='#0d1220',width=3)
            c.create_line(fx-5*hx,fy+5*hy,fx+5*hx,fy-5*hy,fill=body_fill,width=3)
            c.create_line(fx-1*hx+4*px,fy+1*hy-4*py,fx+2*hx,fy-2*hy,fill=wing_fill,width=1)
            c.create_line(fx-1*hx-4*px,fy+1*hy+4*py,fx+2*hx,fy-2*hy,fill=wing_fill,width=1)
            return hx,hy

        # Far-to-near painter's order.  Alpha-backed world sprites have no opaque square
        # corners, so only the actual circular world can cover an object behind it.
        draw.sort(key=lambda z:(float(z[0]),int(z[1])))
        for _depth,_order,kind,payload in draw:
            if kind=='body':
                x0,y0,photo,bid,name,alias,r,temp,col=payload
                c.create_image(x0,y0,image=photo)
                label=c.create_text(x0,y0-r-4,text=name,fill=col,anchor='s',font=('Consolas',8,'bold'))
                c.tag_bind(label,'<Button-1>',lambda ev,z=bid:self._edit_world_alias(z))
                if alias:
                    sub=c.create_text(x0,y0-r+8,text=bid,fill='#8fa7bb',anchor='s',font=('Consolas',6))
                    c.tag_bind(sub,'<Button-1>',lambda ev,z=bid:self._edit_world_alias(z))
                c.create_text(x0,y0+r+3,text=f"{temp:.0f}C",fill=col,anchor='n',font=('Consolas',7))
            elif kind=='resource':
                rx,ry,icon=payload
                c.create_text(rx,ry,text=icon,fill='#ffffff',font=('Segoe UI Emoji',11),anchor='center')
            elif kind=='trail':
                tx0,ty0,tx1,ty1=payload
                c.create_line(tx0,ty0,tx1,ty1,fill='#405b72',width=2)
            else:
                # No fly item is expected in the painter list: Yar is intentionally
                # drawn once, last, after depth has classified his visibility.
                pass

        # Always draw Yar last so the operator never loses him.  His color tells the
        # truth about depth: normal colors when he is the frontmost object at this
        # screen coordinate; reverse-xray colors when a planet surface is nearer.
        if fly_hidden:
            _draw_fly_marker(x,y,body_fill='#ff46d9',wing_fill='#9cf7ff',heading_fill='#f58fff',course_fill='#aefbff',xray=True)
            c.create_text(x+10,y+10,text='XR',fill='#f4b0ff',anchor='nw',font=('Consolas',7,'bold'))
        else:
            _draw_fly_marker(x,y)

        note=' • observer extrapolation' if viewer_dt>1e-6 else ''
        xray_text=' • fly always visible; reverse-xray color = occluded'
        c.create_text(7,7,text='3D observer map • circular worlds + depth-occluded trail • blue=body heading • cyan=velocity/course • fruit/💩=food'+xray_text+note,fill='#b9cce0',anchor='nw',font=('Consolas',8))
    def _neuro(self,s):
        for x in self.neuro_tree.get_children():self.neuro_tree.delete(x)
        for r in s['lineage']:
            self.neuro_tree.insert('', 'end', values=(f"W{int(r['working_generation']):03d}",f"{r.get('time_s',0):.1f}",r.get('parent_body'),r.get('parent_type'),r.get('daughter_body'),r.get('new_edges'),
                '' if r.get('parent_live_hz') is None else f"{r['parent_live_hz']:.1f}",'' if r.get('daughter_live_hz') is None else f"{r['daughter_live_hz']:.1f}"))
        cand=s.get('growable_candidate');txt=(f"build: {s.get('build_id',BUILD_ID)}\n"
            f"growth pressure max: {s['max_growth_pressure']}/{s['growth_threshold']} epochs\n"
            f"growth cooldown: {s.get('growth_cooldown_remaining',0)}/{s.get('growth_cooldown_total',0)} epochs\n")
        txt+=('ready candidate: none\n' if cand is None else f"ready candidate: body {cand['body_id']} {cand['type']} rate={cand['rate_hz']:.1f}Hz saturated={cand['saturated_fraction']:.1%}\n")
        txt+='New neurons are weak provisional copies of local structural topology. Working W generations are not automatically canonical G generations.\n'
        self.neuro_text.delete('1.0','end');self.neuro_text.insert('1.0',txt)
    def _close(self):
        self.worker.running_evt.clear();self.worker.stop_evt.set()
        try:self.engine.save()
        except Exception as e:
            if not self.messagebox.askyesno('Save failed',f'Final save failed:\n{e}\n\nClose anyway?'):self.worker.stop_evt.clear();return
        self.engine.release_process_lock()
        self.root.destroy()
    def run(self):self.root.mainloop()


def run_headless(a):
    e=LifeLineageEngine(); e.configure_threads(a.threads); n=0;print("No Man's Fly — Specimen-003 headless life mode");print(f"resume t={e.exp.time_s:.2f}s W{e.generation:03d} {e.current_graph.name}")
    print(f"thread policy: {e.exp.core.thread_info()}",flush=True)
    neural_ms = 1000.0*a.world_dt if a.neural_ms <= 0 else float(a.neural_ms)
    if a.neural_ms > 0 and abs(neural_ms-1000.0*a.world_dt) > 1e-9:
        print(f"WARNING: debug clock override active: world {a.world_dt}s vs neural {neural_ms}ms",flush=True)
    else:
        print(f"single simulated clock: {a.world_dt:.3f}s world == {neural_ms:.1f}ms neural",flush=True)
    try:
        while a.steps<=0 or n<a.steps:
            st=time.perf_counter();s=e.step(world_dt=a.world_dt,neural_ms=neural_ms,learn=not a.no_learn,auto_growth=not a.no_growth);wall=time.perf_counter()-st;n+=1;l=s['log']
            print(f"{n:06d} t={l['time_s']:9.2f}s W{s['generation']:03d} net={l.get('network_hz',0):6.2f} motor={l.get('motor_mean_hz',0):6.2f} hunger={l.get('hunger',0):.3f} temp={l.get('thermal_c',0):5.1f} err={l.get('homeostatic_error',0):.3f} val={l.get('valence_for_next_epoch',0):+.4f} food={l.get('nearest_resource_distance',float('inf')):5.2f} eat={l.get('nutrient_consumed',0):.4f} wall={wall:.2f}s",flush=True)
            if a.autosave_sim_seconds>0 and e.exp.time_s-e.last_save_sim_s>=a.autosave_sim_seconds:e.save();print(' checkpoint saved',flush=True)
    except KeyboardInterrupt:print('\nStopping...')
    finally:
        try:e.save();print(f'Final checkpoint saved at t={e.exp.time_s:.2f}s W{e.generation:03d}')
        finally:e.release_process_lock()


def self_test(graph:Path):
    e=Specimen003Experiment(graph); c=e.census();x=e.step(world_dt=.05,neural_ms=50.,learn=False)
    assert c['motor']['motor_neurons']>0 and len(e.eye_map.entries)>0
    print(json.dumps({'ok':True,'graph':graph.name,'neurons':e.core.n,'edges':e.core.edge_count,'census':c,'first_step':x.__dict__},indent=2,default=str))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--headless',action='store_true');ap.add_argument('--steps',type=int,default=0);ap.add_argument('--world-dt',type=float,default=.10);ap.add_argument('--neural-ms',type=float,default=0.,help='debug override; 0 locks neural time to world time');ap.add_argument('--autosave-sim-seconds',type=float,default=30.);ap.add_argument('--threads',default='auto',help='auto or an integer host thread count');ap.add_argument('--no-learn',action='store_true');ap.add_argument('--no-growth',action='store_true');ap.add_argument('--reset-live',action='store_true');ap.add_argument('--reset-only',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--graph',type=Path,default=BASE_GRAPH);a=ap.parse_args()
    if a.self_test:self_test(a.graph);return 0
    try:
        if a.reset_only:
            e=LifeLineageEngine(reset_live=True); e.save(); e.release_process_lock(); print('Specimen-003 live lineage reset to fresh immutable seed.'); return 0
        if a.headless:run_headless(a);return 0
        ObservatoryApp(LifeLineageEngine(reset_live=a.reset_live)).run(); return 0
    except SpecimenAlreadyRunningError as exc:
        msg=str(exc)
        print(f'ERROR: {msg}',file=sys.stderr,flush=True)
        if not a.headless and os.name=='nt':
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(0,msg,"No Man's Fly — specimen already running",0x10)
            except Exception:
                pass
        return 2
if __name__=='__main__':raise SystemExit(main())


