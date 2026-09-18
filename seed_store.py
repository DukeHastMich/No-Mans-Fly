"""Immutable default graph and independent, numbered organism snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'Runs'
DEFAULT_SEED = RUNS / 'Specimen-Seed'
GRAPH_NAME = 'male-cns-v1.0-full-graph-v2.npz'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_seed(folder, *, verify=True):
    folder = Path(folder)
    info = json.loads((folder / 'seed.json').read_text(encoding='utf-8'))
    if info.get('format') != 'no-mans-fly-seed-v1':
        raise ValueError(f'Unsupported seed: {folder}')
    name = info['graph']
    if not isinstance(name, str) or Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError('Seed graph must be a file inside the seed folder')
    graph = folder / name
    if not graph.is_file():
        raise FileNotFoundError(graph)
    if verify and digest(graph) != info['graph_sha256']:
        raise ValueError(f'Seed graph changed: {folder}. It will not be overwritten.')
    return info, graph


def ensure_default_seed(source, *, runs=None):
    """Publish once; even an incomplete existing default is never overwritten."""
    runs = Path(runs) if runs is not None else RUNS
    target = runs / 'Specimen-Seed'
    if target.exists():
        info, graph = read_seed(target)
        if info.get('kind') != 'default' or info.get('neurons') != 166700:
            raise ValueError('Specimen-Seed must contain the original 166700-neuron baseline')
        return graph
    source = Path(source)
    import numpy as np
    with np.load(source, allow_pickle=False) as data:
        if len(data['bodies']) != 166700:
            raise ValueError('Default seed requires the original 166700-neuron connectome')
    # The baseline builder supplies its provenance/hash manifest.
    manifest = source.with_suffix(source.suffix + '.manifest.json')
    provenance = json.loads(manifest.read_text(encoding='utf-8'))
    source_sha = digest(source)
    if (provenance.get('format') != 'male-cns-full-graph-v2-manifest'
            or provenance.get('graph_sha256') != source_sha):
        raise ValueError('Baseline provenance/hash validation failed')
    runs.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.default-seed-', dir=runs))
    try:
        shutil.copy2(source, temp / GRAPH_NAME)
        if digest(temp / GRAPH_NAME) != source_sha:
            raise ValueError('Baseline changed during seed creation')
        shutil.copy2(manifest, temp / (GRAPH_NAME + '.manifest.json'))
        info = dict(format='no-mans-fly-seed-v1', kind='default', neurons=166700,
                    graph=GRAPH_NAME, graph_sha256=source_sha, working_generation=0)
        (temp / 'seed.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        try:
            temp.rename(target)
        except OSError:
            if not target.exists():
                raise
            return read_seed(target)[1]
        return target / GRAPH_NAME
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def save_numbered_seed(runs, graph, save_checkpoint, validate_checkpoint, *, specimen, generation):
    """Copy a complete snapshot and graph; never reference a mutable live folder."""
    runs = Path(runs)
    runs.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        target = runs / f'Specimen-Seed-{number:03d}'
        claim = runs / f'.{target.name}.reserving'
        if target.exists():
            number += 1
            continue
        try:
            claim.mkdir()
            break
        except FileExistsError:
            number += 1
    temp = None
    try:
        temp = Path(tempfile.mkdtemp(prefix='.custom-seed-', dir=runs))
        save_checkpoint(temp)
        if not validate_checkpoint(temp):
            raise ValueError('Seed checkpoint is incomplete; no seed was published')
        graph = Path(graph)
        shutil.copy2(graph, temp / graph.name)
        info = dict(format='no-mans-fly-seed-v1', kind='custom', graph=graph.name,
                    graph_sha256=digest(temp / graph.name), source_specimen=specimen,
                    working_generation=int(generation))
        (temp / 'seed.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        # A seed is visible only after the complete snapshot is ready.
        if target.exists():
            raise FileExistsError(target)
        temp.rename(target)
        return target
    finally:
        if temp is not None and temp.exists():
            shutil.rmtree(temp)
        claim.rmdir()


def available_seeds(runs=None):
    runs = Path(runs) if runs is not None else RUNS
    seeds = [runs / 'Specimen-Seed']
    if runs.exists():
        for p in sorted(runs.glob('Specimen-Seed-[0-9]*')):
            try:
                info, _ = read_seed(p, verify=False)
                if info.get('kind') == 'custom':
                    seeds.append(p)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return seeds
