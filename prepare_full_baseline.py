#!/usr/bin/env python3
"""One-time Specimen-003 full MaleCNS baseline preparer.

Downloads the three pinned MaleCNS v1.0 flat-connectome sources when necessary,
verifies exact byte length + SHA256, and builds the no-pruning v2 runtime graph.
The source files are deliberately kept after success so the build remains auditable;
use --delete-sources only when you explicitly want the ~1.1 GiB source cache removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

from build_full_malecns_graph import SOURCE_HASHES, SOURCE_BYTES, build, sha256_file

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = ROOT / "source" / "male-cns-v1.0"
DEFAULT_GRAPH = ROOT / "data" / "male-cns-v1.0-full-graph-v2.npz"
BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"


def human_bytes(n: int) -> str:
    x=float(n)
    for u in ("B","KiB","MiB","GiB","TiB"):
        if x < 1024 or u == "TiB": return f"{x:.1f} {u}"
        x/=1024


def hash_ok(path: Path, name: str, verbose: bool=True) -> bool:
    if not path.exists() or path.stat().st_size != SOURCE_BYTES[name]:
        return False
    if verbose: print(f"  verifying SHA256: {name}", flush=True)
    return sha256_file(path) == SOURCE_HASHES[name]


def download_one(name: str, dest: Path) -> None:
    expected=SOURCE_BYTES[name]
    url=BASE_URL+name
    temp=dest.with_suffix(dest.suffix+".part")
    dest.parent.mkdir(parents=True,exist_ok=True)
    # Partial files are restarted rather than HTTP-resumed so the final SHA remains
    # simple and unambiguous on every platform/proxy.
    if temp.exists(): temp.unlink()
    print(f"\nDownloading {name}\n  {url}\n  {human_bytes(expected)}", flush=True)
    req=urllib.request.Request(url,headers={"User-Agent":"No-Mans-Fly-Specimen003/1.0"})
    started=time.time(); done=0; last=-1
    with urllib.request.urlopen(req,timeout=60) as src, temp.open("wb") as out:
        while True:
            chunk=src.read(4*1024*1024)
            if not chunk: break
            out.write(chunk); done+=len(chunk)
            pct=int(done*100/max(1,expected))
            if pct!=last and (pct%2==0 or done==expected):
                rate=done/max(time.time()-started,1e-6)
                print(f"  {pct:3d}%  {human_bytes(done)} / {human_bytes(expected)}  {human_bytes(int(rate))}/s",flush=True)
                last=pct
    if temp.stat().st_size != expected:
        raise RuntimeError(f"download size mismatch for {name}: {temp.stat().st_size:,} != {expected:,}")
    digest=sha256_file(temp)
    if digest != SOURCE_HASHES[name]:
        raise RuntimeError(f"download SHA256 mismatch for {name}: {digest}")
    os.replace(temp,dest)
    print("  verified",digest,flush=True)


def ensure_sources(source_dir: Path, *, offline: bool=False) -> None:
    source_dir.mkdir(parents=True,exist_ok=True)
    for name in SOURCE_HASHES:
        p=source_dir/name
        if hash_ok(p,name,verbose=p.exists()):
            print(f"source OK: {name}",flush=True); continue
        if p.exists():
            bad=p.with_suffix(p.suffix+".bad")
            if bad.exists(): bad.unlink()
            p.rename(bad)
            print(f"Existing source failed verification; moved to {bad.name}",flush=True)
        if offline:
            raise FileNotFoundError(f"verified source missing in offline mode: {p}")
        download_one(name,p)


def graph_is_ready(path: Path) -> bool:
    manifest=path.with_suffix(path.suffix+".manifest.json")
    if not path.exists() or not manifest.exists(): return False
    try:
        m=json.loads(manifest.read_text(encoding="utf-8"))
        if m.get("format")!="male-cns-full-graph-v2-manifest": return False
        if int(m.get("neurons",0))!=166_700: return False
        if m.get("graph_sha256") != sha256_file(path): return False
        return True
    except Exception:
        return False


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-dir",type=Path,default=DEFAULT_SOURCE_DIR)
    ap.add_argument("--graph",type=Path,default=DEFAULT_GRAPH)
    ap.add_argument("--offline",action="store_true",help="never download; require verified local source files")
    ap.add_argument("--rebuild",action="store_true")
    ap.add_argument("--delete-sources",action="store_true",help="explicitly remove source cache after a verified build")
    a=ap.parse_args()

    a.graph.parent.mkdir(parents=True,exist_ok=True)
    if not a.rebuild and graph_is_ready(a.graph):
        print(f"Full MaleCNS baseline already verified: {a.graph}")
        return 0

    print("No Man's Fly — Specimen-003 full MaleCNS first-run preparation")
    print("This is a one-time ~1.1 GiB source acquisition/build. Runtime launches reuse the built graph.\n")
    ensure_sources(a.source_dir,offline=a.offline)

    print("\nAll official source hashes verified. Building full no-pruning graph...",flush=True)
    manifest=build(a.source_dir,a.graph)
    if int(manifest["neurons"]) != 166_700:
        raise RuntimeError("builder returned a noncanonical pinned-release neuron census")
    print(f"\nREADY: {a.graph}\n  neurons: {manifest['neurons']:,}\n  edges:   {manifest['directed_edges']:,}\n  SHA256:  {manifest['graph_sha256']}")

    if a.delete_sources:
        # Consequential only because caller explicitly requested the flag.
        shutil.rmtree(a.source_dir)
        print("Source cache deleted by explicit --delete-sources request.")
    else:
        print(f"Source cache retained for audit/rebuild: {a.source_dir}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
