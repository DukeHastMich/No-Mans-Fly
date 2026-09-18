#!/usr/bin/env python3
"""One-time runtime validation for a freshly built Specimen-003 baseline.

A small marker caches successful validation by graph SHA + runtime-code SHA.  Normal
launches therefore do not rerun a neural epoch unless the graph or critical runtime
code changes.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
GRAPH=ROOT/'data'/'male-cns-v1.0-full-graph-v2.npz'
MANIFEST=GRAPH.with_suffix(GRAPH.suffix+'.manifest.json')
MARKER=ROOT/'data'/'.specimen003-runtime-validation.json'
CRITICAL=[
    'full_plastic_connectome.py','full_connectome_evolution.py','specimen003_experiment.py',
    'living_universe.py','fly_body.py','homeostasis.py','life_sensory.py','proprioception.py',
    'compound_eye.py','space_fly_universe.py',
]

def code_sha():
    h=hashlib.sha256()
    for name in CRITICAL:
        p=ROOT/name; h.update(name.encode()); h.update(b'\0'); h.update(p.read_bytes()); h.update(b'\0')
    return h.hexdigest()

def main():
    if not GRAPH.exists() or not MANIFEST.exists():
        raise SystemExit('Full baseline/manifest missing; run prepare_full_baseline.py first.')
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    graph_sha=str(m.get('graph_sha256','')); csha=code_sha()
    if MARKER.exists():
        try:
            old=json.loads(MARKER.read_text(encoding='utf-8'))
            if old.get('graph_sha256')==graph_sha and old.get('runtime_code_sha256')==csha and old.get('ok') is True:
                print('Specimen-003 runtime validation marker matches current graph/code: OK')
                return 0
        except Exception:
            pass

    from specimen003_experiment import Specimen003Experiment
    print('First-run runtime validation: loading full graph and running one closed-loop epoch...',flush=True)
    e=Specimen003Experiment(GRAPH,neural_seed=3003)
    census=e.census()
    if int(census['neurons'])!=166_700:
        raise RuntimeError(f"runtime loaded {census['neurons']:,} neurons; expected 166,700 from the pinned MaleCNS v1.0 release")
    if int(census['edges'])!=int(m.get('directed_edges',-1)):
        raise RuntimeError('runtime edge count does not match graph manifest')
    if int(census['eye']['mapped_photoreceptor_bodies'])<=0:
        raise RuntimeError('compound-eye mapping is empty')
    if int(census['motor']['motor_neurons'])<=0:
        raise RuntimeError('motor census is empty')
    frame=e.step(world_dt=.02,neural_ms=20.0,learn=False)
    result={
        'format':'no-mans-fly-specimen003-runtime-validation-v1','ok':True,
        'graph_sha256':graph_sha,'runtime_code_sha256':csha,
        'neurons':int(census['neurons']),'edges':int(census['edges']),
        'motor_neurons':int(census['motor']['motor_neurons']),
        'mapped_photoreceptor_bodies':int(census['eye']['mapped_photoreceptor_bodies']),
        'homeostatic_populations':census['homeostatic_populations'],
        'food_sensory':census['food_sensory'],'proprioceptive_populations':census['proprioceptive_populations'],
        'test_epoch':{'world_dt_s':.02,'neural_ms':20.0,'network_hz':frame.network_hz,
                      'motor_mean_hz':frame.motor_mean_hz,'homeostatic_error':frame.homeostatic_error},
    }
    MARKER.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2)); return 0

if __name__=='__main__': raise SystemExit(main())
