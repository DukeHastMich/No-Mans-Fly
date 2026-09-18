#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
LIVE=ROOT/'Runs'/'Specimen-003-Live'
def resolve(text:str)->Path:
    s=str(text).replace('\\',os.sep).replace('/',os.sep)
    p=Path(s)
    return p if p.is_absolute() else ROOT/p
try:
    mp=LIVE/'live_lineage.json'; sp=LIVE/'specimen003_state.json'
    if not mp.exists() or not sp.exists(): raise RuntimeError('live manifest/state missing')
    m=json.loads(mp.read_text(encoding='utf-8'))
    g=resolve(m.get('current_graph',''))
    if not g.exists(): raise RuntimeError(f'current frozen graph missing: {g}')
    st=json.loads(sp.read_text(encoding='utf-8'))
    with np.load(g,allow_pickle=True) as z:
        if 'bodies' not in z: raise RuntimeError('frozen graph has no bodies array')
        n=int(len(z['bodies']))
    fast=LIVE/'specimen003_fast_state.npz'
    if not fast.exists(): raise RuntimeError('fast-state checkpoint missing')
    print(f"VALID EXISTING LINEAGE: W{int(m.get('working_generation',0)):03d}  t={float(st.get('time_s',0.0)):.3f}s  neurons={n:,}")
    print(f"Current graph: {g}")
except Exception as e:
    print(f"NO VALID EXISTING LINEAGE: {e}",file=sys.stderr)
    raise SystemExit(2)
