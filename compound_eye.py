#!/usr/bin/env python3
"""MaleCNS compound-eye frontend with anatomically anchored spatial calibration.

Patch 11 replaces the old random/synthetic receptor-to-facet assignment whenever
possible.  The neural boundary remains unchanged: optical drive enters retained
MaleCNS photoreceptor bodies only; no downstream visual feature or steering signal
is injected.

Spatial calibration tiers:
  1. Official MaleCNS optic-column workbook (L1/R7/R8 body -> ME_[LR]_col_h1_h2).
  2. Retained R1-R6 bodies are assigned to the strongest *actual* outgoing L1
     cartridge in the graph, then inherit that cartridge's official column.
  3. Optical axes embed that real hex-column lattice inside the measured Drosophila
     field envelope (small frontal binocular overlap, broad lateral/posterior field,
     posterior blind region, dorsal-to-ventral coverage).
  4. If the workbook cannot be obtained, a clearly labelled deterministic fallback
     is used.  The fallback still assigns R1-R6 through their real L1 connectivity,
     but it is NOT claimed to be exact retinotopy.

The measured envelope is not a claim of specimen-specific per-ommatidium micro-CT
axes.  Those can later replace only the column->axis embedding without changing the
neural simulator or checkpoint format.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from world_visual import surface_channels

OPSIN_NM = {
    "R1-R6": 478.0,
    "R7p": 345.0,
    "R7y": 375.0,
    "R7d": 360.0,
    "R7_unclear": 360.0,
    "R8p": 437.0,
    "R8y": 508.0,
    "R8d": 345.0,
    "R8_unclear": 475.0,
    "R7R8_unclear": 420.0,
}
PHOTORECEPTOR_TYPES = tuple(OPSIN_NM)

# Pinned public MaleCNS supplement.  First run caches this tiny workbook next to
# the graph under data/ when network access is available.
OFFICIAL_COLUMN_URL = (
    "https://raw.githubusercontent.com/flyconnectome/2025malecns/"
    "67767d2233657983993ff6c2be48e836a935863c/"
    "supplemental_data/optic-column-type-assignments-v1.0.xlsx"
)


@dataclass(frozen=True)
class EyeOpticsConfig:
    n_air: float = 1.0003
    n_cornea: float = 1.40
    # Patch VISION_BIO_DITHER1: preserve the measured dark-adapted width, but
    # allow light-driven photomechanical narrowing toward the ~4 degree regime
    # observed in vivo instead of forcing every ommatidium to remain broad.
    external_fwhm_deg: float = 8.2  # compatibility alias / dark-adapted width
    bright_fwhm_deg: float = 4.0
    adaptation_tau_s: float = 0.080
    microsaccade_enabled: bool = True
    microsaccade_max_deg: float = 5.0
    microsaccade_tau_s: float = 0.045
    microsaccade_contrast_scale: float = 0.18
    retinal_sample_hz: float = 50.0
    retinal_max_subsamples: int = 8
    temporal_burst_weight: float = 0.55
    # Experimental interstitial spectral samplers. These are explicitly synthetic
    # pre-connectome sensors, never graph neurons. Each samples between two real
    # ommatidia and weakly feeds both neighboring documented receptor channels.
    interstitial_dither_enabled: bool = True
    interstitial_dither_gain: float = 0.22
    interstitial_offset_fraction: float = 0.38
    max_drive_hz: float = 240.0
    drive_bins: int = 48
    exposure_scale: float = 0.5
    # Measured-field envelope used to embed the real MaleCNS hex lattice.
    # R eye spans slightly across the forward midline (-8 deg) to posterior +155.
    frontal_cross_deg: float = 8.0
    posterior_azimuth_deg: float = 155.0
    dorsal_elevation_deg: float = 90.0
    ventral_elevation_deg: float = -70.0
    auto_fetch_official_columns: bool = True
    fetch_timeout_s: float = 6.0


@dataclass
class EyeFrame:
    facet_response: np.ndarray
    rh1: np.ndarray
    r7p: np.ndarray
    r7y: np.ndarray
    r8p: np.ndarray
    r8y: np.ndarray
    brightest_id: str | None
    brightest_strength: float
    visible_star_count: int
    left_mean: float
    right_mean: float
    peak_response: float
    peak_side: str
    peak_yaw_deg: float
    peak_elevation_deg: float
    ambient_rh6: float = 0.0
    effective_fwhm_mean_deg: float = 8.2
    microsaccade_rms_deg: float = 0.0
    retinal_subsamples: int = 1
    dither_sensor_count: int = 0
    spectral_model: str = "legacy-five-band"
    scene_position: np.ndarray | None = None
    scene_eye_position: np.ndarray | None = None
    scene_orientation: np.ndarray | None = None
    scene_velocity: np.ndarray | None = None
    scene_time_s: float = 0.0
    scene_on_surface: bool = False
    scene_body_id: str | None = None
    scene_bodies: tuple | None = None
    scene_sources: tuple | None = None


@dataclass(frozen=True)
class ReceptorMapEntry:
    neuron_index: int
    body_id: int
    receptor_type: str
    side: str
    facet_index_global: int
    assignment: str = ""


@dataclass(frozen=True)
class OpticColumn:
    name: str
    side: str
    h1: int
    h2: int
    l1: int | None
    r7: int | None
    r8: int | None
    column_type: str = ""


def _unit_from_yaw_elevation(yaw: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    ce=np.cos(elevation)
    return np.column_stack((ce*np.cos(yaw),ce*np.sin(yaw),np.sin(elevation))).astype(np.float64)


def _yaw_elevation(v: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    v=np.asarray(v,dtype=np.float64)
    return np.arctan2(v[:,1],v[:,0]), np.arctan2(v[:,2],np.sqrt(v[:,0]**2+v[:,1]**2))


def _hex_disk_points(n:int)->np.ndarray:
    if n<=0:return np.empty((0,2),dtype=np.float64)
    radius=1
    while 1+3*radius*(radius+1)<n: radius+=1
    pts=[]
    for q in range(-radius,radius+1):
        for r in range(-radius,radius+1):
            s=-q-r
            if max(abs(q),abs(r),abs(s))<=radius:
                x=math.sqrt(3.)*(q+.5*r); y=1.5*r
                pts.append((x,y,q,r))
    pts.sort(key=lambda x:(x[0]*x[0]+x[1]*x[1],x[2],x[3]))
    a=np.asarray([[p[0],p[1]] for p in pts[:n]],dtype=np.float64)
    a[:,0]/=max(1e-9,float(np.max(np.abs(a[:,0])))); a[:,1]/=max(1e-9,float(np.max(np.abs(a[:,1]))))
    return a


def _xlsx_cell_text(cell, shared, ns):
    t=cell.get("t")
    if t=="inlineStr":
        node=cell.find("s:is",ns)
        return "" if node is None else "".join(node.itertext())
    v=cell.find("s:v",ns)
    if v is None:return ""
    raw=v.text or ""
    if t=="s":
        try:return shared[int(raw)]
        except Exception:return ""
    return raw


def _xlsx_rows(path:Path):
    """Yield worksheet rows as {column-letter: text} using only stdlib."""
    ns={"s":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared=[]
        if "xl/sharedStrings.xml" in z.namelist():
            root=ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("s:si",ns):shared.append("".join(si.itertext()))
        sheets=sorted(x for x in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml",x))
        for sheet in sheets:
            root=ET.fromstring(z.read(sheet))
            for row in root.findall("s:sheetData/s:row",ns):
                out={}
                for cell in row.findall("s:c",ns):
                    ref=cell.get("r",""); col=re.sub(r"\d","",ref)
                    out[col]=_xlsx_cell_text(cell,shared,ns)
                if out:yield out


def load_optic_columns(path:str|Path)->list[OpticColumn]:
    """Parse the official workbook (and simple equivalent test workbooks)."""
    path=Path(path); cols=[]
    current_header=None
    for row in _xlsx_rows(path):
        vals={k:str(v).strip() for k,v in row.items()}
        # Locate a header row regardless of worksheet/column ordering.
        if any(v.lower()=="column" for v in vals.values()):
            current_header={k:v.strip().lower() for k,v in vals.items()}; continue
        if current_header:
            byname={name:vals.get(letter,"") for letter,name in current_header.items()}
            cname=byname.get("column","")
        else:
            cname=vals.get("A",""); byname={}
        m=re.fullmatch(r"ME_([LR])_col_(\d+)_(\d+)",cname)
        if not m:continue
        side,h1,h2=m.groups()
        def body(name, fallback_letter):
            raw=byname.get(name.lower(),vals.get(fallback_letter,""))
            try:x=int(float(raw))
            except Exception:return None
            return x if x>0 else None
        ctype=(byname.get("column_type") or byname.get("column type") or byname.get("type") or "").strip()
        cols.append(OpticColumn(cname,side,int(h1),int(h2),body("l1","B"),body("r7","C"),body("r8","E"),ctype))
    # Deterministic ordering: left facets first, then right, each by lattice coords.
    cols.sort(key=lambda c:(0 if c.side=="L" else 1,c.h1,c.h2,c.name))
    if not cols:raise ValueError(f"no ME_[LR]_col_* rows found in {path}")
    return cols


def _candidate_workbooks(graph_path:Path):
    env=os.environ.get("NMF_OPTIC_COLUMNS")
    if env:yield Path(env)
    root=graph_path.parent
    package_root=Path(__file__).resolve().parent
    for p in (package_root/"data"/"optic-column-type-assignments-v1.0.xlsx",
              package_root/"optic-column-type-assignments-v1.0.xlsx",
              root/"optic-column-type-assignments-v1.0.xlsx",root/"optic-columns.xlsx",
              graph_path.parent.parent/"optic-column-type-assignments-v1.0.xlsx"):
        yield p


def _try_fetch_workbook(graph_path:Path, timeout:float)->Path|None:
    # Prefer graph's data directory so a downloaded calibration travels with install,
    # but never make lack of Internet a launch failure.
    target=Path(__file__).resolve().parent/"data"/"optic-column-type-assignments-v1.0.xlsx"
    try:
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix(target.suffix+".part")
        with urllib.request.urlopen(OFFICIAL_COLUMN_URL,timeout=max(1.,float(timeout))) as r, open(tmp,"wb") as f:
            f.write(r.read())
        # Validate before publishing cache.
        load_optic_columns(tmp)
        tmp.replace(target); return target
    except Exception:
        try:
            if tmp.exists():tmp.unlink()
        except Exception:pass
        return None


class SyntheticHexEyeMap:
    """Compatibility name; Patch11 is column-calibrated when official data exists."""

    OLD_MAPPING_MODES={"synthetic-hex-v1-awaiting-official-column-map","male-cns-L1-connectivity+measured-envelope-fallback-v2"}

    def __init__(self, graph_path:str|Path, config:EyeOpticsConfig|None=None):
        self.graph_path=Path(graph_path); self.config=config or EyeOpticsConfig()
        z=np.load(self.graph_path,allow_pickle=True)
        self.bodies=np.asarray(z["bodies"],dtype=np.int64)
        self.types=np.asarray(z["types"],dtype=str)
        self.instance=np.asarray(z["instance"],dtype=str)
        self.indptr=np.asarray(z["indptr"],dtype=np.int64)
        self.indices=np.asarray(z["indices"],dtype=np.int32)
        self.synapse_count=np.asarray(z["synapse_count"],dtype=np.uint32) if "synapse_count" in z else np.ones(len(self.indices),np.uint32)
        self._body_to_index={int(b):i for i,b in enumerate(self.bodies)}

        workbook=None; columns=None
        for p in _candidate_workbooks(self.graph_path):
            if p.exists():
                try:columns=load_optic_columns(p); workbook=p; break
                except Exception:pass
        if columns is None and self.config.auto_fetch_official_columns:
            p=_try_fetch_workbook(self.graph_path,self.config.fetch_timeout_s)
            if p is not None:
                try:columns=load_optic_columns(p); workbook=p
                except Exception:columns=None
        self.workbook_path=workbook
        self.official_columns=columns or []
        if self.official_columns:
            self._build_official()
            self.mapping_mode="male-cns-official-columns+measured-eye-envelope-v2"
        else:
            self._build_fallback()
            self.mapping_mode="male-cns-L1-connectivity+measured-envelope-fallback-v2"

        self.yaw_rad,self.elevation_rad=_yaw_elevation(self.axes)
        self.neuron_index=np.asarray([e.neuron_index for e in self.entries],dtype=np.int32)
        self.body_id=np.asarray([e.body_id for e in self.entries],dtype=np.int64)
        self.receptor_type=np.asarray([e.receptor_type for e in self.entries],dtype=str)
        self.neuron_side=np.asarray([e.side for e in self.entries],dtype=str)
        self.neuron_facet=np.asarray([e.facet_index_global for e in self.entries],dtype=np.int32)
        self.assignment=np.asarray([e.assignment for e in self.entries],dtype=str)
        self._release_construction_graph()

    def _release_construction_graph(self):
        """Drop graph arrays used only while resolving the eye calibration.

        FullMaleCNSCore already owns the authoritative sparse graph. Retaining another
        25M-edge copy here wastes hundreds of MB and makes structural growth hold
        several full graphs at once. Runtime eye rendering uses only the resolved
        facet/receptor arrays.
        """
        for name in ("bodies","types","instance","indptr","indices","synapse_count","_body_to_index"):
            if hasattr(self,name):
                try: delattr(self,name)
                except Exception: setattr(self,name,None)
        self._construction_graph_released=True

    @staticmethod
    def _side_from_instance(inst:str)->str:
        s=str(inst).upper()
        if s.endswith("_L") or "_L_" in s:return "L"
        if s.endswith("_R") or "_R_" in s:return "R"
        return ""

    def _ids(self,receptor_type:str,side:str|None=None)->np.ndarray:
        ids=np.flatnonzero(self.types==receptor_type)
        if side:
            ids=np.asarray([i for i in ids if self._side_from_instance(self.instance[i])==side],dtype=np.int32)
        return ids[np.argsort(self.bodies[ids])]

    @staticmethod
    def _stable_permutation(n:int,label:str)->np.ndarray:
        seed=int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],"little")
        return np.random.default_rng(seed).permutation(n)

    def _column_axes(self, columns:list[OpticColumn])->tuple[np.ndarray,np.ndarray]:
        """Embed official h1/h2 hex coordinates into a measured fly-eye envelope."""
        c=self.config; axes=[]; confidence=[]
        for side in ("L","R"):
            cc=[x for x in columns if x.side==side]
            h1=np.asarray([x.h1 for x in cc],float); h2=np.asarray([x.h2 for x in cc],float)
            if not len(cc):continue
            # h1 is the published front->back lattice coordinate.  Remove its linear
            # component from h2 to recover the transverse/dorsal-ventral coordinate.
            if len(cc)>2 and np.ptp(h1)>0:
                A=np.column_stack((h1,np.ones(len(h1))))
                beta=np.linalg.lstsq(A,h2,rcond=None)[0]
                vert=h2-(A@beta)
            else:vert=h2-np.mean(h2)
            u=(h1-np.min(h1))/max(1e-9,float(np.ptp(h1)))
            # Robust vertical scaling avoids one edge cell flattening the whole eye.
            lo,hi=np.percentile(vert,[1,99]) if len(vert)>10 else (float(np.min(vert)),float(np.max(vert)))
            vv=np.clip((vert-(lo+hi)/2.)/max(1e-9,(hi-lo)/2.),-1.,1.)
            # DRA is dorsal.  Use it to choose transverse sign when labels exist.
            dra=np.asarray([("dra" in x.column_type.lower()) for x in cc],bool)
            if np.any(dra) and float(np.mean(vv[dra]))<0: vv=-vv
            # Right eye crosses ~8 degrees into left visual hemisphere anteriorly;
            # left eye mirrors this.  This creates a small honest frontal overlap.
            mag=-c.frontal_cross_deg+(c.posterior_azimuth_deg+c.frontal_cross_deg)*u
            yaw=np.radians(mag if side=="R" else -mag)
            elev=np.radians(c.ventral_elevation_deg+(c.dorsal_elevation_deg-c.ventral_elevation_deg)*(vv+1.)/2.)
            axes.append(_unit_from_yaw_elevation(yaw,elev)); confidence.extend([1.0]*len(cc))
        return np.vstack(axes),np.asarray(confidence,np.float32)

    def _strongest_l1_target(self, source_idx:int, l1_body_to_facet:dict[int,int])->tuple[int|None,int]:
        s,e=int(self.indptr[source_idx]),int(self.indptr[source_idx+1])
        if e<=s:return None,0
        tgt=self.indices[s:e]; sc=self.synapse_count[s:e]
        best_f=None;best_w=-1
        for ti,w in zip(tgt,sc):
            fi=l1_body_to_facet.get(int(self.bodies[int(ti)]))
            if fi is not None and int(w)>best_w:best_f,best_w=fi,int(w)
        return best_f,max(0,best_w)

    def _build_official(self):
        cols=self.official_columns
        left=[x for x in cols if x.side=="L"]; right=[x for x in cols if x.side=="R"]
        self.n_left=len(left); self.n_right=len(right); ordered=left+right
        self.axes,self.facet_confidence=self._column_axes(ordered)
        self.side=np.asarray([x.side for x in ordered],dtype=str)
        self.column_name=np.asarray([x.name for x in ordered],dtype=str)
        self.column_h1=np.asarray([x.h1 for x in ordered],dtype=np.int16)
        self.column_h2=np.asarray([x.h2 for x in ordered],dtype=np.int16)
        self.column_type=np.asarray([x.column_type for x in ordered],dtype=str)
        l1_to_facet={int(x.l1):i for i,x in enumerate(ordered) if x.l1}
        body_to_facet={}
        for i,x in enumerate(ordered):
            for b in (x.r7,x.r8):
                if b:body_to_facet[int(b)]=i
        entries=[]; used=set(); self.r1r6_via_l1=0; self.official_inner_receptors=0; self.fallback_receptors=0
        # Exact R7/R8 identities from the official workbook.
        for body,fi in body_to_facet.items():
            ni=self._body_to_index.get(body)
            if ni is None or self.types[ni] not in PHOTORECEPTOR_TYPES:continue
            rt=str(self.types[ni]); side=str(self.side[fi])
            entries.append(ReceptorMapEntry(ni,body,rt,side,fi,"official-column"));used.add(ni);self.official_inner_receptors+=1
        # R1-R6 through actual retained graph connectivity to official L1 cartridges.
        for ni in self._ids("R1-R6"):
            fi,w=self._strongest_l1_target(int(ni),l1_to_facet)
            if fi is None:continue
            side=str(self.side[fi]);entries.append(ReceptorMapEntry(int(ni),int(self.bodies[ni]),"R1-R6",side,int(fi),f"strongest-L1:{w}"));used.add(int(ni));self.r1r6_via_l1+=1
        # Preserve every remaining retained photoreceptor.  Fallback assignment is
        # explicit and counted; it never masquerades as an official column match.
        for rt in PHOTORECEPTOR_TYPES:
            for ni in self._ids(rt):
                if int(ni) in used:continue
                side=self._side_from_instance(self.instance[ni]) or ("L" if (int(self.bodies[ni])&1)==0 else "R")
                ids=np.flatnonzero(self.side==side)
                if not len(ids):continue
                perm=self._stable_permutation(len(ids),f"patch11-fallback-{rt}-{side}")
                fi=int(ids[perm[int(self.bodies[ni])%len(perm)]])
                entries.append(ReceptorMapEntry(int(ni),int(self.bodies[ni]),rt,side,fi,"fallback-unplaced"));self.fallback_receptors+=1
        entries.sort(key=lambda e:e.neuron_index); self.entries=entries

    def _build_fallback(self):
        # L1 counts define the number of retained visual cartridges, but without the
        # official workbook their spatial order is only a measured-envelope scaffold.
        l1=np.flatnonzero(self.types=="L1")
        left=np.asarray([i for i in l1 if self._side_from_instance(self.instance[i])=="L"],dtype=np.int32)
        right=np.asarray([i for i in l1 if self._side_from_instance(self.instance[i])=="R"],dtype=np.int32)
        if not len(left) or not len(right):raise ValueError("MaleCNS graph does not contain bilateral L1 populations")
        self.n_left=len(left);self.n_right=len(right); self.side=np.asarray(["L"]*len(left)+["R"]*len(right),dtype=str)
        axes=[]
        for side,n in (("L",len(left)),("R",len(right))):
            p=_hex_disk_points(n); u=(p[:,0]+1.)/2.; vv=p[:,1]
            mag=-self.config.frontal_cross_deg+(self.config.posterior_azimuth_deg+self.config.frontal_cross_deg)*u
            yaw=np.radians(mag if side=="R" else -mag)
            elev=np.radians(self.config.ventral_elevation_deg+(self.config.dorsal_elevation_deg-self.config.ventral_elevation_deg)*(vv+1.)/2.)
            axes.append(_unit_from_yaw_elevation(yaw,elev))
        self.axes=np.vstack(axes);self.facet_confidence=np.full(len(self.axes),.55,np.float32)
        self.column_name=np.asarray([f"fallback_{i}" for i in range(len(self.axes))]);self.column_h1=np.zeros(len(self.axes),np.int16);self.column_h2=np.zeros(len(self.axes),np.int16);self.column_type=np.asarray([""]*len(self.axes))
        # Deterministically map L1 bodies to fallback facets, then recover R1-R6 via
        # their real graph edges.  This preserves connectivity evidence even offline.
        l1_to_facet={int(self.bodies[ni]):i for i,ni in enumerate(left)}
        l1_to_facet.update({int(self.bodies[ni]):self.n_left+i for i,ni in enumerate(right)})
        entries=[];used=set();self.r1r6_via_l1=0;self.official_inner_receptors=0;self.fallback_receptors=0
        for ni in self._ids("R1-R6"):
            fi,w=self._strongest_l1_target(int(ni),l1_to_facet)
            if fi is not None:
                side=str(self.side[fi]);entries.append(ReceptorMapEntry(int(ni),int(self.bodies[ni]),"R1-R6",side,int(fi),f"strongest-L1-fallback-axis:{w}"));used.add(int(ni));self.r1r6_via_l1+=1
        for rt in PHOTORECEPTOR_TYPES:
            for ni in self._ids(rt):
                if int(ni) in used:continue
                side=self._side_from_instance(self.instance[ni]) or ("L" if (int(self.bodies[ni])&1)==0 else "R")
                ids=np.flatnonzero(self.side==side); perm=self._stable_permutation(len(ids),f"offline-{rt}-{side}")
                fi=int(ids[perm[int(self.bodies[ni])%len(perm)]])
                entries.append(ReceptorMapEntry(int(ni),int(self.bodies[ni]),rt,side,fi,"fallback-unplaced"));self.fallback_receptors+=1
        entries.sort(key=lambda e:e.neuron_index);self.entries=entries

    @classmethod
    def load_resolved_npz(cls,path:str|Path,config:EyeOpticsConfig|None=None,*,
                          graph_path:str|Path|None=None,calibration:dict|None=None):
        """Load an already-resolved eye calibration without reopening the full graph.

        The checkpoint map contains every runtime facet/receptor assignment.  Graph
        identity is validated by Specimen003Experiment against its already-loaded core,
        avoiding a redundant 25M-edge NPZ read during resume and structural growth.
        """
        path=Path(path);calibration=dict(calibration or {})
        m=np.load(path,allow_pickle=False)
        fmt=str(m["format"]) if "format" in m else ""
        if fmt!="space-fly-compound-eye-map-v2":
            raise ValueError(f"unsupported saved eye-map format: {fmt!r}")
        self=cls.__new__(cls)
        self.graph_path=Path(graph_path) if graph_path is not None else Path('')
        self.config=config or EyeOpticsConfig()
        self.mapping_mode=str(m["mapping_mode"])
        self.axes=np.asarray(m["axes"],dtype=np.float64)
        self.side=np.asarray(m["side"],dtype=str)
        self.yaw_rad=np.asarray(m["yaw_rad"],dtype=np.float64)
        self.elevation_rad=np.asarray(m["elevation_rad"],dtype=np.float64)
        self.facet_confidence=np.asarray(m["facet_confidence"],dtype=np.float32)
        self.column_name=np.asarray(m["column_name"],dtype=str)
        self.column_h1=np.asarray(m["column_h1"],dtype=np.int16)
        self.column_h2=np.asarray(m["column_h2"],dtype=np.int16)
        self.column_type=np.asarray(m["column_type"],dtype=str)
        self.neuron_index=np.asarray(m["neuron_index"],dtype=np.int32)
        self.body_id=np.asarray(m["body_id"],dtype=np.int64)
        self.receptor_type=np.asarray(m["receptor_type"],dtype=str)
        self.neuron_side=np.asarray(m["neuron_side"],dtype=str)
        self.neuron_facet=np.asarray(m["neuron_facet"],dtype=np.int32)
        self.assignment=np.asarray(m["assignment"],dtype=str)
        nf=len(self.axes)
        if not (len(self.side)==len(self.yaw_rad)==len(self.elevation_rad)==len(self.facet_confidence)==nf):
            raise ValueError("saved eye map facet arrays disagree")
        if not (len(self.neuron_index)==len(self.body_id)==len(self.receptor_type)==len(self.neuron_side)==len(self.neuron_facet)==len(self.assignment)):
            raise ValueError("saved eye map receptor arrays disagree")
        if np.any(self.neuron_index<0) or np.any(self.neuron_facet<0) or np.any(self.neuron_facet>=nf):
            raise ValueError("saved eye map contains invalid neuron/facet index")
        self.n_left=int(np.count_nonzero(self.side=="L"));self.n_right=int(np.count_nonzero(self.side=="R"))
        self.entries=[ReceptorMapEntry(int(i),int(b),str(rt),str(sd),int(fi),str(a))
                      for i,b,rt,sd,fi,a in zip(self.neuron_index,self.body_id,self.receptor_type,self.neuron_side,self.neuron_facet,self.assignment)]
        self.r1r6_via_l1=int(calibration.get("r1r6_via_real_l1",sum(str(x).startswith("strongest-L1") for x in self.assignment)))
        self.official_inner_receptors=int(calibration.get("official_inner_receptors",sum(str(x)=="official-column" for x in self.assignment)))
        self.fallback_receptors=int(calibration.get("fallback_receptors",sum(str(x).startswith("fallback") for x in self.assignment)))
        official_count=int(calibration.get("official_columns",nf if "official-columns" in self.mapping_mode else 0))
        self.official_columns=[None]*official_count;self.workbook_path=None
        self._construction_graph_released=True
        return self

    @classmethod
    def load_npz(cls,graph_path:str|Path,path:str|Path,config:EyeOpticsConfig|None=None,*,calibration:dict|None=None):
        """Restore and validate a persisted eye map against a graph.

        Only neuron metadata is read for validation; the sparse 25M-edge arrays are
        not needed to validate an already-resolved calibration.
        """
        graph_path=Path(graph_path)
        self=cls.load_resolved_npz(path,config=config,graph_path=graph_path,calibration=calibration)
        gz=np.load(graph_path,allow_pickle=True)
        bodies=np.asarray(gz["bodies"],dtype=np.int64);types=np.asarray(gz["types"],dtype=str)
        if np.any(self.neuron_index>=len(bodies)):
            raise ValueError("saved eye map contains neuron index outside graph")
        if not np.array_equal(bodies[self.neuron_index],self.body_id):
            raise ValueError("saved eye map photoreceptor bodies do not match selected graph")
        if not np.array_equal(types[self.neuron_index].astype(str),self.receptor_type.astype(str)):
            raise ValueError("saved eye map receptor types do not match selected graph")
        return self

    def save_npz(self,path:str|Path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,format=np.asarray("space-fly-compound-eye-map-v2"),mapping_mode=np.asarray(self.mapping_mode),
            axes=self.axes.astype(np.float64),side=self.side,yaw_rad=self.yaw_rad.astype(np.float64),elevation_rad=self.elevation_rad.astype(np.float64),
            facet_confidence=self.facet_confidence.astype(np.float32),column_name=self.column_name,column_h1=self.column_h1,column_h2=self.column_h2,column_type=self.column_type,
            neuron_index=self.neuron_index,body_id=self.body_id,receptor_type=self.receptor_type,neuron_side=self.neuron_side,neuron_facet=self.neuron_facet,assignment=self.assignment)


class CompoundEyeRenderer:
    """Anatomy-grounded retinal frontend with physiology-aware optical sampling.

    VISION_BIO_DITHER1 keeps the documented MaleCNS photoreceptor targets unchanged.
    The biological path adds light-dependent receptive-field narrowing, stimulus-driven
    photomechanical microshifts and overlapping opsin sensitivity. The optional dither
    path is deliberately separate: virtual pre-connectome samplers sit between real
    ommatidial axes and weakly distribute spectral evidence into the two neighboring
    documented channels. No virtual sensor is inserted into the CNS graph.
    """
    SPECTRAL_MODEL="opsin-overlap+photomechanics-v1"
    _ANCHOR_WL=np.asarray([345.,375.,437.,478.,508.],dtype=np.float64)
    _WL=np.asarray([330.,345.,355.,375.,405.,437.,478.,508.,550.,600.],dtype=np.float64)

    def __init__(self, eye_map:SyntheticHexEyeMap, config:EyeOpticsConfig|None=None):
        self.map=eye_map; self.config=config or eye_map.config
        nf=len(self.map.axes)
        self.prev_facet_response:np.ndarray|None=None
        self._adaptation=np.zeros(nf,dtype=np.float64)
        self._microshift_deg=np.zeros(nf,dtype=np.float64)
        self._last_observe_time_s:float|None=None
        self._tangent=self._build_photomechanical_tangents()
        self._opsin=self._build_opsin_matrix()
        self._build_dither_lattice()

    @staticmethod
    def _gaussian(x,mu,sigma):
        return np.exp(-0.5*np.square((np.asarray(x,dtype=np.float64)-float(mu))/max(float(sigma),1e-9)))

    def _build_opsin_matrix(self):
        """Broad overlapping sensitivities on a compact wavelength grid.

        Peak locations follow the Drosophila Rh1/Rh3/Rh4/Rh5/Rh6 literature. Curve
        widths and secondary-lobe amplitudes remain explicit engineering approximations;
        the goal is to stop treating each receptor as a single monochromatic sample.
        Columns: Rh1, R7p/Rh3, R7y/Rh4, R8p/Rh5, R8y/Rh6.
        """
        w=self._WL
        rh1=self._gaussian(w,490.,46.) + .34*self._gaussian(w,350.,24.) + .12*self._gaussian(w,375.,20.)
        rh3=self._gaussian(w,330.,24.)
        rh4=self._gaussian(w,355.,28.)
        rh5=self._gaussian(w,435.,34.) + .12*self._gaussian(w,350.,32.)
        # For a directional ommatidial ray keep the underlying ~508-nm pigment peak
        # dominant; a weak long-wave tail acknowledges red-eye screening leakage
        # without turning the point sampler into a wide-field ERG preparation.
        rh6=self._gaussian(w,508.,48.) + .10*self._gaussian(w,590.,58.)
        M=np.column_stack((rh1,rh3,rh4,rh5,rh6))
        M/=np.maximum(np.sum(M,axis=0,keepdims=True),1e-12)
        return M

    def _build_photomechanical_tangents(self):
        axes=np.asarray(self.map.axes,dtype=np.float64)
        z=np.array([0.,0.,1.],dtype=np.float64)
        t=np.cross(np.broadcast_to(z,axes.shape),axes)
        n=np.linalg.norm(t,axis=1)
        bad=n<1e-8
        if np.any(bad):
            y=np.array([0.,1.,0.],dtype=np.float64)
            t[bad]=np.cross(np.broadcast_to(y,(int(np.count_nonzero(bad)),3)),axes[bad])
            n=np.linalg.norm(t,axis=1)
        t/=np.maximum(n[:,None],1e-12)
        # Mirror the scaffold between eyes so the synthetic geometric choice cannot
        # create a global handedness preference. Exact per-ommatidial contraction
        # directions remain unknown in this runtime.
        sign=np.where(np.asarray(self.map.side)=="L",-1.,1.)
        return t*sign[:,None]

    def _build_dither_lattice(self):
        axes=np.asarray(self.map.axes,dtype=np.float64); side=np.asarray(self.map.side)
        a=[];b=[];tau=[];kind=[]
        f=float(np.clip(self.config.interstitial_offset_fraction,.20,.80))
        for sd in ("L","R"):
            ids=np.flatnonzero(side==sd)
            if not len(ids): continue
            sub=axes[ids]
            dots=sub@sub.T
            np.fill_diagonal(dots,-2.)
            nn=np.argmax(dots,axis=1)
            for local_i,local_j in enumerate(nn):
                i=int(ids[local_i]);j=int(ids[int(local_j)])
                # Alternating 38/62 placement gives the virtual sensor a genuine
                # geometric asymmetry while keeping the whole-eye lattice balanced.
                t=f if ((i*2654435761)&1)==0 else 1.-f
                a.append(i);b.append(j);tau.append(t)
                # Cycle UV-short, UV-long, blue, green; Rh1 remains entirely real.
                kind.append(1+(int(hashlib.sha256(str(self.map.column_name[i]).encode()).digest()[0])%4))
        self.dither_a=np.asarray(a,dtype=np.int32);self.dither_b=np.asarray(b,dtype=np.int32)
        self.dither_tau=np.asarray(tau,dtype=np.float64);self.dither_kind=np.asarray(kind,dtype=np.int8)
        if len(self.dither_a):
            aa=axes[self.dither_a];bb=axes[self.dither_b];t=self.dither_tau[:,None]
            v=(1.-t)*aa+t*bb;v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
            self.dither_axes=v
            self.dither_weight_a=(1.-self.dither_tau).astype(np.float64)
            self.dither_weight_b=self.dither_tau.astype(np.float64)
        else:
            self.dither_axes=np.empty((0,3),dtype=np.float64);self.dither_weight_a=np.empty(0);self.dither_weight_b=np.empty(0)

    @staticmethod
    def _blackbody_relative(temperature,wavelengths_nm):
        T=np.asarray(temperature,dtype=np.float64)[:,None];lam=np.asarray(wavelengths_nm,dtype=np.float64)[None,:]*1e-9;c2=1.438776877e-2
        x=np.clip(c2/(lam*T),1e-6,700.);B=1./(np.power(lam,5)*np.expm1(x));B/=np.maximum(np.max(B,axis=1,keepdims=True),1e-300);return B

    @staticmethod
    def _inverse_rotate(orientation,vectors):
        q=np.asarray(orientation,dtype=np.float64);qc=np.array([q[0],-q[1],-q[2],-q[3]],dtype=np.float64);vv=np.asarray(vectors,dtype=np.float64).reshape((-1,3));qv=qc[1:]
        t=2.*np.cross(np.broadcast_to(qv,vv.shape),vv);return vv+qc[0]*t+np.cross(np.broadcast_to(qv,vv.shape),t)

    @staticmethod
    def _rotate(orientation,vectors):
        q=np.asarray(orientation,dtype=np.float64);vv=np.asarray(vectors,dtype=np.float64).reshape((-1,3));qv=q[1:]
        t=2.*np.cross(np.broadcast_to(qv,vv.shape),vv);return vv+q[0]*t+np.cross(np.broadcast_to(qv,vv.shape),t)

    @staticmethod
    def _ray_sphere_near(origin,directions,center,radius):
        o=np.asarray(origin,dtype=np.float64);d=np.asarray(directions,dtype=np.float64).reshape((-1,3));c=np.asarray(center,dtype=np.float64)
        r=max(1e-12,float(radius));oc=o-c;b=d@oc;cc=float(oc@oc-r*r);disc=b*b-cc
        root=np.sqrt(np.clip(disc,0.,None));t0=-b-root;t1=-b+root;eps=max(1e-9,r*1e-8)
        t=np.where(t0>eps,t0,np.where(t1>eps,t1,np.inf));t[disc<0.]=np.inf
        return t

    def _surface_spectrum(self):
        old=np.asarray([0.22,0.32,0.55,0.82,1.0],dtype=np.float64)
        a=np.interp(self._WL,self._ANCHOR_WL,old,left=.5*old[0],right=.35*old[-1])
        return a/max(float(np.max(a)),1e-12)

    def _source_spectra(self,ss):
        if not ss:return np.empty((0,len(self._WL)),dtype=np.float64)
        temp=np.asarray([x.temperature for x in ss],dtype=np.float64)
        sp=self._blackbody_relative(temp,self._WL)
        for j,obj in enumerate(ss):
            sw=getattr(obj,"spectral_weights",None)
            if sw is None:continue
            a=np.asarray(sw,dtype=np.float64).reshape(-1)
            if len(a)==5 and np.max(a)>0:
                dense=np.interp(self._WL,self._ANCHOR_WL,a,left=.5*a[0],right=.35*a[-1])
                sp[j]=dense/max(float(np.max(dense)),1e-12)
        return sp

    def _effective_fwhm_deg(self):
        dark=max(float(self.config.external_fwhm_deg),float(self.config.bright_fwhm_deg))
        bright=min(float(self.config.external_fwhm_deg),float(self.config.bright_fwhm_deg))
        # sqrt gives meaningful narrowing before the response is nearly saturated.
        return dark-(dark-bright)*np.sqrt(np.clip(self._adaptation,0.,1.))

    def _shifted_real_axes(self):
        if not self.config.microsaccade_enabled:return np.asarray(self.map.axes,dtype=np.float64)
        a=np.asarray(self.map.axes,dtype=np.float64);t=self._tangent;ang=np.radians(np.clip(self._microshift_deg,-self.config.microsaccade_max_deg,self.config.microsaccade_max_deg))
        v=a*np.cos(ang)[:,None]+t*np.sin(ang)[:,None]
        return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)

    def _prepare_scene(self,universe,craft):
        p=np.asarray(craft.position,dtype=np.float64);bodies=list(universe.local_bodies(p)) if hasattr(universe,"local_bodies") else []
        eye_p=p.copy();contact=getattr(craft,"contact",None)
        if contact is not None and bool(getattr(contact,"on_surface",False)):
            bid=str(getattr(contact,"body_id",''))
            for b in bodies:
                if str(getattr(b,"body_id",''))==bid:
                    n=eye_p-np.asarray(b.center,dtype=np.float64);nn=float(np.linalg.norm(n))
                    if nn>1e-12:eye_p=eye_p+n/nn*max(1e-5,float(b.radius)*1e-4)
                    break
        points=(universe.local_visual_sources(p) if hasattr(universe,"local_visual_sources") else universe.local_stars(p))
        ss=[];dirs_world=np.empty((0,3));dist=np.empty(0)
        if points:
            pos=np.vstack([x.position for x in points]).astype(np.float64);rel=pos-eye_p[None,:];dd=np.linalg.norm(rel,axis=1);good=dd>1e-8
            if np.any(good):
                ss=[x for i,x in enumerate(points) if good[i]];dist=dd[good];rel=rel[good];dirs_world=rel/dist[:,None]
                blocked=np.zeros(len(ss),dtype=bool)
                parent=np.asarray([str(getattr(x,'surface_body_id','') or '') for x in ss],dtype=object)
                for b in bodies:
                    tt=self._ray_sphere_near(eye_p,dirs_world,b.center,b.radius)
                    same=(parent==str(b.body_id));margin=max(2e-5,float(b.radius)*2e-4)
                    blocked|=(~same)&np.isfinite(tt)&(tt<dist-margin)
                if np.any(blocked):
                    keep=~blocked;dirs_world=dirs_world[keep];dist=dist[keep];ss=[x for i,x in enumerate(ss) if keep[i]]
        return p,eye_p,bodies,ss,dirs_world,dist,self._source_spectra(ss)@self._opsin

    def _render_axes_raw(self,axes_body,fwhm_deg,craft,time_s,scene):
        """Return pre-exposure Rh1/Rh3/Rh4/Rh5/Rh6 channel energy.

        Spectral integration is collapsed per source before the large spatial
        acceptance matrix. This is algebraically equivalent to rendering every
        wavelength into every facet, but avoids a costly facet×source×wavelength
        expansion on the 1,772-facet eye.
        """
        _,eye_p,bodies,ss,dirs_world,dist,scene_ops=scene
        axes_body=np.asarray(axes_body,dtype=np.float64);nax=len(axes_body)
        fwhm=np.broadcast_to(np.asarray(fwhm_deg,dtype=np.float64).reshape(-1) if np.ndim(fwhm_deg) else np.asarray([float(fwhm_deg)]), (nax,))
        axes_world=self._rotate(craft.orientation,axes_body);axes_world/=np.maximum(np.linalg.norm(axes_world,axis=1,keepdims=True),1e-12)
        surface_raw=np.zeros((nax,5),dtype=np.float64);surface_depth=np.full(nax,np.inf);surface_id=np.full(nax,'',dtype=object)
        surf_ops=self._surface_spectrum()@self._opsin
        body_visible=0
        for b in bodies:
            tt=self._ray_sphere_near(eye_p,axes_world,b.center,b.radius);hit=np.isfinite(tt)&(tt<surface_depth)
            if not np.any(hit):continue
            hp=eye_p[None,:]+axes_world[hit]*tt[hit,None];normal=(hp-np.asarray(b.center,dtype=np.float64)[None,:])/max(float(b.radius),1e-12);facing=np.clip(np.sum(normal*(-axes_world[hit]),axis=1),0.,1.)
            surface_raw[hit]=surface_channels(str(b.body_id),normal,float(b.albedo),self._opsin);surface_depth[hit]=tt[hit];surface_id[hit]=str(b.body_id);body_visible+=1
        sigma_ext=np.maximum(np.radians(fwhm)/2.355,1e-6)
        for b in bodies:
            rel=np.asarray(b.center,dtype=np.float64)-eye_p;d=float(np.linalg.norm(rel));r=float(b.radius)
            if d<=r+1e-9:continue
            cdir=rel/d;cb=self._inverse_rotate(craft.orientation,cdir.reshape(1,3))[0];dots=np.clip(axes_body@cb,-1.,1.);theta=np.arccos(dots);alpha=math.asin(min(.999999,r/d));outside=np.maximum(0.,theta-alpha);spill=np.exp(-.5*np.square(outside/sigma_ext));area=np.minimum(1.0,np.square(alpha/np.maximum(sigma_ext,1e-9)))
            m=(~np.isfinite(surface_depth))&(spill>1e-4)
            if np.any(m):
                radiance=surface_channels(str(b.body_id),(-cdir).reshape(1,3),float(b.albedo),self._opsin)[0];surface_raw[m]+=(area[m]*spill[m])[:,None]*radiance[None,:]
        star_strength=np.empty(0);sp_ops=np.empty((0,5))
        if len(ss):
            dirs_body=self._inverse_rotate(craft.orientation,dirs_world);dirs_body/=np.maximum(np.linalg.norm(dirs_body,axis=1,keepdims=True),1e-12)
            lum=np.asarray([x.luminosity for x in ss],dtype=np.float64);pulse=1.+.035*np.sin(.17*time_s+np.asarray([x.pulse_phase for x in ss]));star_strength=lum*pulse/(.20+(dist/12.)**2)
            dot=np.clip(axes_body@dirs_body.T,-1.,1.);front=dot>0.;theta_ext=np.arccos(np.clip(dot,0.,1.));sin_internal=np.clip((self.config.n_air/self.config.n_cornea)*np.sin(theta_ext),0.,1.);theta_internal=np.arcsin(sin_internal)
            ext_half=np.radians(fwhm*.5);internal_half=np.arcsin(np.clip((self.config.n_air/self.config.n_cornea)*np.sin(ext_half),0.,1.));sigma_internal=np.maximum(internal_half/math.sqrt(2.*math.log(2.)),1e-9)
            accept=np.exp(-.5*np.square(theta_internal/sigma_internal[:,None]));n1=float(self.config.n_air);n2=float(self.config.n_cornea);ci=np.clip(np.cos(theta_ext),1e-9,1.);ct=np.clip(np.cos(theta_internal),1e-9,1.);rs=np.square((n1*ci-n2*ct)/np.maximum(n1*ci+n2*ct,1e-12));rp=np.square((n1*ct-n2*ci)/np.maximum(n1*ct+n2*ci,1e-12));accept*=np.clip(1.-.5*(rs+rp),0.,1.)*np.clip(dot,0.,1.);accept*=front
            sp_ops=scene_ops;point_raw=(accept*star_strength[None,:])@sp_ops
        else:point_raw=np.zeros_like(surface_raw)
        raw=point_raw+surface_raw
        meta={"surface_raw":surface_raw,"surface_id":surface_id,"body_visible":body_visible,"star_strength":star_strength,"sp_ops":sp_ops,"ss":ss}
        return raw,meta

    def _channelize(self,raw):
        out=1.-np.exp(-np.asarray(raw,dtype=np.float64)/max(float(self.config.exposure_scale),1e-9))
        return np.clip(out,0.,1.).astype(np.float32)

    def _apply_dither(self,channels,craft,time_s,scene,fwhm_real):
        if not self.config.interstitial_dither_enabled or not len(self.dither_axes):return channels
        df=(1.-self.dither_tau)*fwhm_real[self.dither_a]+self.dither_tau*fwhm_real[self.dither_b]
        draw,_=self._render_axes_raw(self.dither_axes,df,craft,time_s,scene);dch=self._channelize(draw);gain=float(max(0.,self.config.interstitial_dither_gain));out=np.asarray(channels,dtype=np.float32).copy()
        for k in range(1,5):
            m=self.dither_kind==k
            if not np.any(m):continue
            val=dch[m,k].astype(np.float64)*gain
            np.add.at(out[:,k],self.dither_a[m],(val*self.dither_weight_a[m]).astype(np.float32))
            np.add.at(out[:,k],self.dither_b[m],(val*self.dither_weight_b[m]).astype(np.float32))
        return np.clip(out,0.,1.)

    def _observe_once(self,universe,craft,time_s:float,dt_s:float,scene=None)->EyeFrame:
        scene=self._prepare_scene(universe,craft) if scene is None else scene;fwhm=self._effective_fwhm_deg();axes=self._shifted_real_axes();raw,meta=self._render_axes_raw(axes,fwhm,craft,time_s,scene);ch=self._channelize(raw)
        rh1,r7p,r7y,r8p,r8y=(ch[:,i].astype(np.float32) for i in range(5));facet=np.maximum.reduce([rh1,r7p,r7y,r8p,r8y]).astype(np.float32)
        # Update adaptation and photomechanical state for the *next* retinal sample.
        dt=max(1e-4,float(dt_s));aa=1.-math.exp(-dt/max(float(self.config.adaptation_tau_s),1e-4));self._adaptation=(1.-aa)*self._adaptation+aa*facet.astype(np.float64)
        if self.prev_facet_response is None:delta=np.zeros_like(facet,dtype=np.float64)
        else:delta=facet.astype(np.float64)-self.prev_facet_response.astype(np.float64)
        if self.config.microsaccade_enabled:
            target=float(self.config.microsaccade_max_deg)*np.tanh(delta/max(float(self.config.microsaccade_contrast_scale),1e-6));ma=1.-math.exp(-dt/max(float(self.config.microsaccade_tau_s),1e-4));self._microshift_deg=(1.-ma)*self._microshift_deg+ma*target
            self._microshift_deg=np.clip(self._microshift_deg,-self.config.microsaccade_max_deg,self.config.microsaccade_max_deg)
        self.prev_facet_response=facet.copy();self._last_observe_time_s=float(time_s)
        left=float(np.mean(facet[:self.map.n_left])) if self.map.n_left else 0.;right=float(np.mean(facet[self.map.n_left:])) if self.map.n_right else 0.;pi=int(np.argmax(facet)) if len(facet) else 0
        star_strength=meta["star_strength"];sp_ops=meta["sp_ops"];surface_raw=meta["surface_raw"];surface_id=meta["surface_id"];ss=meta["ss"]
        ambient_point=float(np.sum(star_strength*sp_ops[:,4])) if len(star_strength) else 0.;ambient_surface=float(4.0*np.mean(surface_raw[:,4])) if len(surface_raw) else 0.;ambient_rh6=float(np.clip(1.-math.exp(-(ambient_point+ambient_surface)/max(4.*self.config.exposure_scale,1e-9)),0.,1.))
        brightest_id=None;brightest_strength=0.
        if len(star_strength):bi=int(np.argmax(star_strength));brightest_id=str(ss[bi].star_id);brightest_strength=float(star_strength[bi])
        if np.any(surface_raw):si=int(np.argmax(np.max(surface_raw,axis=1)));sv=float(np.max(surface_raw[si]));sid=str(surface_id[si]);
        else:si=-1;sv=0.;sid=''
        if sv>brightest_strength and sid:brightest_id='body:'+sid;brightest_strength=sv
        return EyeFrame(facet,rh1,r7p,r7y,r8p,r8y,brightest_id,brightest_strength,int(len(ss))+int(meta["body_visible"]),left,right,float(facet[pi]),str(self.map.side[pi]),float(math.degrees(self.map.yaw_rad[pi])),float(math.degrees(self.map.elevation_rad[pi])),ambient_rh6,float(np.mean(fwhm)),float(np.sqrt(np.mean(np.square(self._microshift_deg)))),1,0,self.SPECTRAL_MODEL)

    def observe_sequence(self,universe,craft,time_s:float,duration_s:float)->EyeFrame:
        duration=max(0.,float(duration_s));n=max(1,min(int(self.config.retinal_max_subsamples),int(math.ceil(duration*max(float(self.config.retinal_sample_hz),1.0)))))
        dt=(duration/n) if duration>0 else 1./max(float(self.config.retinal_sample_hz),1.0);frames=[]
        scene=self._prepare_scene(universe,craft)
        for k in range(n):frames.append(self._observe_once(universe,craft,float(time_s)+k*dt,dt,scene))
        w=float(np.clip(self.config.temporal_burst_weight,0.,1.));stack=lambda name:np.stack([getattr(f,name) for f in frames],axis=0);merged=[]
        for name in ("rh1","r7p","r7y","r8p","r8y"):
            a=stack(name);merged.append(np.clip((1.-w)*np.mean(a,axis=0)+w*np.max(a,axis=0),0.,1.).astype(np.float32))
        # The speculative interstitial lattice is sampled once per neural epoch,
        # after the documented photomechanical sub-sampling. This keeps it a weak
        # supplementary channel and avoids doubling the cost of every retinal sample.
        ch=np.column_stack(merged).astype(np.float32);dither_count=0
        if self.config.interstitial_dither_enabled and len(self.dither_axes):
            ch=self._apply_dither(ch,craft,float(time_s)+max(0.,duration-dt),scene,self._effective_fwhm_deg());dither_count=int(len(self.dither_axes))
        rh1,r7p,r7y,r8p,r8y=(ch[:,i].astype(np.float32) for i in range(5));facet=np.maximum.reduce([rh1,r7p,r7y,r8p,r8y]).astype(np.float32);last=frames[-1];left=float(np.mean(facet[:self.map.n_left])) if self.map.n_left else 0.;right=float(np.mean(facet[self.map.n_left:])) if self.map.n_right else 0.;pi=int(np.argmax(facet)) if len(facet) else 0
        frame=EyeFrame(facet,rh1,r7p,r7y,r8p,r8y,last.brightest_id,max(f.brightest_strength for f in frames),max(f.visible_star_count for f in frames),left,right,float(facet[pi]),str(self.map.side[pi]),float(math.degrees(self.map.yaw_rad[pi])),float(math.degrees(self.map.elevation_rad[pi])),max(f.ambient_rh6 for f in frames),float(np.mean([f.effective_fwhm_mean_deg for f in frames])),float(np.mean([f.microsaccade_rms_deg for f in frames])),n,dither_count,self.SPECTRAL_MODEL)

        frame.scene_position=np.asarray(craft.position,dtype=float).copy()
        frame.scene_eye_position=np.asarray(scene[1],dtype=float).copy()
        frame.scene_orientation=np.asarray(craft.orientation,dtype=float).copy()
        frame.scene_velocity=np.asarray(getattr(craft,'velocity',[0,0,0]),dtype=float).copy()
        frame.scene_bodies=tuple(scene[2]);frame.scene_sources=tuple(scene[3])
        frame.scene_time_s=float(time_s)
        contact=getattr(craft,'contact',None)
        frame.scene_on_surface=bool(getattr(contact,'on_surface',False))
        frame.scene_body_id=getattr(contact,'body_id',None)
        return frame

    def observe(self,universe,craft,time_s:float)->EyeFrame:
        # Compatibility path for tests/tools that call the renderer directly.
        if self._last_observe_time_s is None:duration=1./max(float(self.config.retinal_sample_hz),1.0)
        else:duration=max(1e-4,float(time_s)-float(self._last_observe_time_s))
        return self.observe_sequence(universe,craft,time_s,duration)

    def preview(self,universe,craft,time_s:float)->EyeFrame:
        """Observer-only preview that cannot advance retinal adaptation state."""
        prev=None if self.prev_facet_response is None else self.prev_facet_response.copy();adapt=self._adaptation.copy();shift=self._microshift_deg.copy();last=self._last_observe_time_s
        try:return self.observe_sequence(universe,craft,time_s,1./max(float(self.config.retinal_sample_hz),1.0))
        finally:
            self.prev_facet_response=prev;self._adaptation[:]=adapt;self._microshift_deg[:]=shift;self._last_observe_time_s=last

    def neuron_rates(self,frame:EyeFrame):
        values=np.zeros(len(self.map.entries),dtype=np.float32);f=self.map.neuron_facet;rt=self.map.receptor_type
        for typ in np.unique(rt):
            m=rt==typ
            if typ=="R1-R6":src=frame.rh1
            elif typ=="R7p":src=frame.r7p
            elif typ=="R7y":src=frame.r7y
            elif typ in ("R7d","R7_unclear"):src=.5*(frame.r7p+frame.r7y)
            elif typ=="R8p":src=frame.r8p
            elif typ=="R8y":src=frame.r8y
            elif typ=="R8d":src=frame.r7p
            elif typ=="R8_unclear":src=.5*(frame.r8p+frame.r8y)
            elif typ=="R7R8_unclear":src=.25*(frame.r7p+frame.r7y+frame.r8p+frame.r8y)
            else:src=frame.facet_response
            values[m]=src[f[m]]
        return self.map.neuron_index,self.config.max_drive_hz*np.clip(values,0.,1.)

    def drives(self,frame:EyeFrame):
        ids,rates=self.neuron_rates(frame);bins=max(2,int(self.config.drive_bins));maxhz=max(1e-9,float(self.config.max_drive_hz));q=np.clip(np.rint((bins-1)*rates/maxhz),0,bins-1).astype(np.int16);out=[]
        for b in range(1,bins):
            m=q==b
            if np.any(m):out.append((ids[m].astype(np.int32,copy=False),float(maxhz*b/(bins-1))))
        return out

    def snapshot_temporal(self):
        return dict(adaptation=self._adaptation.tolist(),microshift_deg=self._microshift_deg.tolist(),
                    previous_facet=None if self.prev_facet_response is None else self.prev_facet_response.tolist(),
                    last_observe_time_s=self._last_observe_time_s)

    def restore_temporal(self,payload):
        if payload is None:
            self.reset_temporal_state();return
        adaptation=np.asarray(payload['adaptation'],dtype=np.float64)
        microshift=np.asarray(payload['microshift_deg'],dtype=np.float64)
        previous=payload.get('previous_facet')
        previous=None if previous is None else np.asarray(previous,dtype=np.float32)
        for value in (adaptation,microshift,previous):
            if value is not None and (value.shape!=self._adaptation.shape or not np.all(np.isfinite(value))):
                raise ValueError('Retinal temporal checkpoint does not match eye map')
        stamp=payload.get('last_observe_time_s')
        if stamp is not None and not math.isfinite(float(stamp)):
            raise ValueError('Invalid retinal checkpoint time')
        self._adaptation=adaptation.copy();self._microshift_deg=microshift.copy()
        self.prev_facet_response=None if previous is None else previous.copy()
        self._last_observe_time_s=None if stamp is None else float(stamp)

    def reset_temporal_state(self):
        self.prev_facet_response=None;self._adaptation.fill(0.);self._microshift_deg.fill(0.);self._last_observe_time_s=None


class HofbauerBuchnerEyeletBridge:
    """Weak non-spatial Rh6-like ambient-light input to retained H-B eyelets.

    The MaleCNS graph has seven HBeyelet sensory cells.  This bridge carries only
    508-nm-weighted physical irradiance from the same scene seen by the compound
    eyes.  It injects no direction, clock phase, reward, novelty, or action signal.
    """
    MAX_DRIVE_HZ=55.0
    def __init__(self,core): self.ids=core.ids(neuron_type="HBeyelet")
    def drives(self,frame:EyeFrame):
        hz=self.MAX_DRIVE_HZ*float(np.clip(getattr(frame,"ambient_rh6",0.0),0.0,1.0))
        return [(self.ids,hz)] if len(self.ids) and hz>0.0 else []
    def census(self): return {"HBeyelet":int(len(self.ids)),"mode":"ambient Rh6-like physical light; non-spatial"}

def summarize_eye_map(eye_map:SyntheticHexEyeMap)->dict:
    rt=eye_map.receptor_type; yaw=np.degrees(eye_map.yaw_rad)
    return {
        "mapping_mode":eye_map.mapping_mode,
        "official_workbook_loaded":bool(eye_map.official_columns),
        "official_workbook":None if eye_map.workbook_path is None else str(eye_map.workbook_path),
        "official_columns":int(len(eye_map.official_columns)),
        "facets_left":int(eye_map.n_left),"facets_right":int(eye_map.n_right),
        "mapped_photoreceptor_bodies":int(len(eye_map.entries)),
        "official_inner_receptors":int(eye_map.official_inner_receptors),
        "r1r6_via_real_l1":int(eye_map.r1r6_via_l1),"fallback_receptors":int(eye_map.fallback_receptors),
        "frontal_binocular_overlap_deg":float(2*eye_map.config.frontal_cross_deg),
        "posterior_blind_gap_deg":float(360-2*eye_map.config.posterior_azimuth_deg),
        "azimuth_range_deg":[float(np.min(yaw)),float(np.max(yaw))],
        "receptors":{str(t):int(np.sum(rt==t)) for t in sorted(set(rt.tolist()))},
        "axis_fidelity":"official MaleCNS column lattice embedded in measured field envelope; not specimen-specific per-ommatidium micro-CT axes" if eye_map.official_columns else "offline fallback measured-envelope scaffold; not exact retinotopy",
    }


