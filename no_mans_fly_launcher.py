#!/usr/bin/env python3
"""No Man's Fly graphical bootstrap / hardware tuning splash.

This file intentionally uses only the Python standard library until Begin is
pressed.  Runtime/build dependencies, lineage validation, AutoThreads tuning,
and the optional exact GPU hybrid probe all happen while the splash is visible.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from seed_store import available_seeds

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "Runs"
BASE_GRAPH = ROOT / "data" / "male-cns-v1.0-full-graph-v2.npz"
SEED_DIR = RUNS / "Specimen-Seed"
ASSET = ROOT / "assets" / "no_mans_fly_splash.png"

FLAVOR = [
    "Resplining ticulators ...",
    "Calibrating compound-eye observers ...",
    "Counting tiny opinions ...",
    "Aligning imaginary flight corridors ...",
    "Reticulating synapses ...",
    "Asking Yar to hold still ...",
    "Polishing the apple atmosphere ...",
]


def _run_label(path: Path) -> str:
    mp = path / "live_lineage.json"
    try:
        d = json.loads(mp.read_text(encoding="utf-8"))
        return str(d.get("specimen") or path.name.removesuffix("-Live"))
    except Exception:
        return path.name.removesuffix("-Live")


def scan_lineages():
    out = []
    if not RUNS.exists():
        return out
    for p in RUNS.iterdir():
        if not p.is_dir() or not p.name.endswith("-Live"):
            continue
        sp = p / "specimen003_state.json"
        mp = p / "live_lineage.json"
        if not sp.exists() or not mp.exists():
            continue
        try:
            state = json.loads(sp.read_text(encoding="utf-8"))
            man = json.loads(mp.read_text(encoding="utf-8"))
            label = str(man.get("specimen") or _run_label(p))
            gen = int(man.get("working_generation", 0))
            age = float(state.get("time_s", man.get("sim_time_s", 0.0)))
            display = f"{label} — W{gen:03d} — {age:.1f} sim s"
            out.append({"display": display, "path": p, "label": label,
                        "generation": gen, "age": age, "mtime": sp.stat().st_mtime})
        except Exception:
            display = f"{_run_label(p)} — checkpoint needs validation"
            out.append({"display": display, "path": p, "label": _run_label(p),
                        "generation": 0, "age": 0.0, "mtime": sp.stat().st_mtime})
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


def next_specimen_label():
    """Return the lowest unused positive ID, preserving even partial lineages."""
    used = set()
    if RUNS.exists():
        for p in RUNS.iterdir():
            m = re.fullmatch(r"Specimen-(\d+)-Live(?:\.creating)?", p.name, re.IGNORECASE)
            if m:
                used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return f"Specimen-{n:03d}"


def reserve_new_lineage():
    """Claim a live folder; seed numbering is independent of fly numbering."""
    RUNS.mkdir(parents=True, exist_ok=True)
    while True:
        label = next_specimen_label()
        live = RUNS / f"{label}-Live"
        try:
            live.mkdir()
        except FileExistsError:
            continue
        return label, live


def resolve_graph(text: str) -> Path:
    p = Path(str(text).replace("\\", os.sep).replace("/", os.sep))
    return p if p.is_absolute() else ROOT / p


def validate_lineage_files(live: Path, *, deep_hash: bool, progress):
    progress(21, f"Checking {live.name} checkpoint files ...")
    mp = live / "live_lineage.json"
    sp = live / "specimen003_state.json"
    required = [mp, sp, live / "specimen003_memory.npz", live / "specimen003_fast_state.npz"]
    missing = [x.name for x in required if not x.exists() or x.stat().st_size <= 0]
    if missing:
        raise RuntimeError("checkpoint is incomplete: missing " + ", ".join(missing))
    man = json.loads(mp.read_text(encoding="utf-8"))
    state = json.loads(sp.read_text(encoding="utf-8"))
    graph = resolve_graph(man.get("current_graph", ""))
    if not graph.exists() or graph.stat().st_size <= 0:
        raise RuntimeError(f"frozen working graph is missing: {graph}")
    if deep_hash:
        expected = None
        gp = live / "growth_events.jsonl"
        if gp.exists():
            rows = []
            for line in gp.read_text(encoding="utf-8").splitlines():
                try: rows.append(json.loads(line))
                except Exception: pass
            for row in reversed(rows):
                if str(row.get("graph")) == graph.name and row.get("graph_sha256"):
                    expected = str(row["graph_sha256"]).lower(); break
        if expected:
            progress(25, f"Verifying frozen W{int(man.get('working_generation',0)):03d} graph hash ...")
            h = hashlib.sha256()
            with graph.open("rb") as f:
                while True:
                    b = f.read(8 * 1024 * 1024)
                    if not b: break
                    h.update(b)
            got = h.hexdigest().lower()
            if got != expected:
                raise RuntimeError(f"working graph SHA-256 mismatch: expected {expected}, got {got}")
            progress(28, "Frozen graph hash verified.")
        else:
            progress(28, "No stored graph hash for this generation; structural file checks passed.")
    return man, state, graph


def module_group_available(names):
    try:
        for n in names: importlib.import_module(n)
        return True
    except Exception:
        return False


def run_stream(cmd, progress, base_percent, label):
    progress(base_percent, label)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    last = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line:
            last = line
            progress(base_percent, line[:180])
    rc = proc.wait()
    if rc:
        raise RuntimeError(f"{label} failed with code {rc}" + (f": {last}" if last else ""))


class Splash:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root = tk.Tk()
        self.root.title("No Man's Fly")
        self.root.geometry("1280x720")
        self.root.resizable(False, False)
        self.root.configure(bg="#03070d")
        self.q = queue.Queue()
        self.result = None
        self.error = None
        self.lineages = scan_lineages()
        self.by_display = {x["display"]: x for x in self.lineages}
        self.mode = tk.StringVar(value="continue" if self.lineages else "new")
        self.use_gpu = tk.BooleanVar(value=True)
        self.tune_cpu = tk.BooleanVar(value=True)
        self.verify = tk.BooleanVar(value=True)
        self._running = False
        self._flavor_index = 0
        self._build()
        self.root.after(40, self._poll)

    def _build(self):
        tk, ttk = self.tk, self.ttk
        self.canvas = tk.Canvas(self.root, width=1280, height=720, bd=0, highlightthickness=0, bg="#02060b")
        self.canvas.pack(fill="both", expand=True)
        if ASSET.exists():
            self.bg = tk.PhotoImage(file=str(ASSET))
            self.canvas.create_image(0, 0, image=self.bg, anchor="nw")
        else:
            self.canvas.create_text(640, 150, text="NO MAN'S FLY", fill="white", font=("Segoe UI Light", 48))

        style = ttk.Style()
        try: style.theme_use("clam")
        except Exception: pass
        style.configure("Splash.TFrame", background="#07101b")
        style.configure("Splash.TLabel", background="#07101b", foreground="#d9edf7")
        style.configure("Splash.TCheckbutton", background="#07101b", foreground="#d9edf7")
        style.configure("Splash.TRadiobutton", background="#07101b", foreground="#d9edf7")

        panel = ttk.Frame(self.root, style="Splash.TFrame", padding=10)
        self.canvas.create_window(35, 465, width=1210, height=230, anchor="nw", window=panel)

        left = ttk.Frame(panel, style="Splash.TFrame"); left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(panel, style="Splash.TFrame"); right.pack(side="right", fill="y", padx=(16,0))

        ttk.Label(left, text="LAUNCH PROFILE", style="Splash.TLabel", font=("Consolas", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,4))
        self.cont_radio = ttk.Radiobutton(left, text="Continue", variable=self.mode, value="continue", style="Splash.TRadiobutton", command=self._mode_changed)
        self.cont_radio.grid(row=1, column=0, sticky="w")
        vals = [x["display"] for x in self.lineages]
        self.lineage_var = tk.StringVar(value=vals[0] if vals else "No saved lineages found")
        self.combo = ttk.Combobox(left, textvariable=self.lineage_var, values=vals, width=47, state="readonly" if vals else "disabled")
        self.combo.grid(row=1, column=1, columnspan=2, sticky="we", padx=(6,0))
        if not vals: self.cont_radio.state(["disabled"])

        self.new_radio = ttk.Radiobutton(left, text=f"New Fly  ({next_specimen_label()})", variable=self.mode, value="new", style="Splash.TRadiobutton", command=self._mode_changed)
        self.new_radio.grid(row=2,column=0,sticky="w",pady=(2,4))
        self.seed_var = tk.StringVar(value="Specimen-Seed")
        self.seed_combo = ttk.Combobox(left, textvariable=self.seed_var,
                                     values=[p.name for p in available_seeds(RUNS)],
                                     width=30, state="readonly")
        self.seed_combo.grid(row=2,column=1,columnspan=2,sticky="we",padx=(6,0))
        ttk.Checkbutton(left, text="Tune CPU threads to this hardware", variable=self.tune_cpu, style="Splash.TCheckbutton").grid(row=3,column=0,columnspan=2,sticky="w")
        ttk.Checkbutton(left, text="Use GPU if available", variable=self.use_gpu, style="Splash.TCheckbutton").grid(row=4,column=0,columnspan=2,sticky="w")
        ttk.Checkbutton(left, text="Verify frozen checkpoint before launch", variable=self.verify, style="Splash.TCheckbutton").grid(row=5,column=0,columnspan=2,sticky="w")

        self.status = tk.StringVar(value="Choose Continue or New Fly, then Begin.")
        ttk.Label(left, textvariable=self.status, style="Splash.TLabel", font=("Consolas", 9)).grid(row=6,column=0,columnspan=3,sticky="we",pady=(7,0))
        self.progress = ttk.Progressbar(left, orient="horizontal", mode="determinate", maximum=100, value=0)
        self.progress.grid(row=7,column=0,columnspan=3,sticky="we",pady=(4,0))
        left.columnconfigure(1, weight=1)

        self.begin = ttk.Button(right, text="BEGIN", command=self._begin, width=18)
        self.begin.pack(pady=(20,8), ipady=6)
        self.quit_btn = ttk.Button(right, text="Exit", command=self.root.destroy, width=18)
        self.quit_btn.pack()
        self.detail = tk.StringVar(value="Hardware tests run before the Observatory opens.")
        ttk.Label(right, textvariable=self.detail, style="Splash.TLabel", wraplength=260, justify="left", font=("Consolas", 8)).pack(pady=(12,0))
        self._mode_changed()

    def _refresh_lineages(self):
        selected = self.by_display.get(self.lineage_var.get())
        self.lineages = scan_lineages()
        self.by_display = {x["display"]: x for x in self.lineages}
        vals = list(self.by_display)
        choice = next((x["display"] for x in self.lineages
                       if selected and x["path"] == selected["path"]),
                      vals[0] if vals else "No saved lineages found")
        self.lineage_var.set(choice)
        self.combo.configure(values=vals, state="readonly" if vals else "disabled")
        self.cont_radio.state(["!disabled"] if vals else ["disabled"])
        self.new_radio.configure(text=f"New Fly  ({next_specimen_label()})")
        if not vals:
            self.mode.set("new")
        seeds = [p.name for p in available_seeds(RUNS)]
        self.seed_combo.configure(values=seeds)
        if self.seed_var.get() not in seeds:
            self.seed_var.set("Specimen-Seed")
        self._mode_changed()

    def _mode_changed(self):
        self.seed_combo.configure(state="readonly" if self.mode.get() == "new" else "disabled")
        if self.mode.get() == "new":
            self.detail.set("Choose the original seed or a saved custom seed. Creates a separate fly; seeds stay unchanged.")
        elif self.lineages:
            self.detail.set("Continues the selected frozen W-generation exactly where it was saved.")

    def post(self, percent, text):
        self.q.put(("progress", float(percent), str(text)))

    def flavor(self, percent):
        text = FLAVOR[self._flavor_index % len(FLAVOR)]; self._flavor_index += 1
        self.post(percent, text)

    def _set_controls(self, disabled=True):
        state = "disabled" if disabled else "normal"
        self.begin.configure(state=state)
        self.quit_btn.configure(state=state if disabled else "normal")
        self.new_radio.state(["disabled"] if disabled else ["!disabled"])
        if disabled:
            self.seed_combo.configure(state="disabled")
            self.cont_radio.state(["disabled"])
            self.combo.configure(state="disabled")
        else:
            self._refresh_lineages()

    def _begin(self):
        if self._running: return
        self._refresh_lineages()
        if self.mode.get() == "continue" and not self.lineages:
            self.status.set("No saved lineage is available to continue."); return
        self._running = True; self._set_controls(True)
        self.progress["value"] = 2
        self.status.set("Starting launch checks ...")
        opts = {"mode": self.mode.get(), "use_gpu": bool(self.use_gpu.get()),
                "tune_cpu": bool(self.tune_cpu.get()), "verify": bool(self.verify.get()),
                "lineage_display": self.lineage_var.get(), "seed_name": self.seed_var.get()}
        threading.Thread(target=self._bootstrap, args=(opts,), daemon=True).start()

    def _tuner_progress(self, base, span):
        seen = {"n":0}
        def cb(msg, **payload):
            seen["n"] += 1
            pct = min(base+span-1, base + min(span-1, seen["n"]*1.6))
            self.post(pct, msg)
        return cb

    def _bootstrap(self, opts):
        reserved = None
        engine = None
        try:
            self.post(5, "Checking Python runtime dependencies ...")
            if not module_group_available(("numpy","scipy","numba")):
                run_stream([sys.executable,"-m","pip","install","-r",str(ROOT/"requirements-runtime.txt")], self.post, 7, "Installing runtime dependencies ...")
            self.post(12, "Runtime dependencies ready.")

            if (opts["mode"] == "new" and opts.get("seed_name", "Specimen-Seed") == "Specimen-Seed"
                    and not SEED_DIR.exists() and not BASE_GRAPH.exists()):
                self.post(14, "Fresh fly needs the MaleCNS baseline; preparing first-run data ...")
                if not module_group_available(("pandas","pyarrow")):
                    run_stream([sys.executable,"-m","pip","install","-r",str(ROOT/"requirements-build.txt")], self.post, 15, "Installing graph-build dependencies ...")
                run_stream([sys.executable, str(ROOT/"prepare_full_baseline.py")], self.post, 17, "Preparing MaleCNS baseline ...")
                if not BASE_GRAPH.exists(): raise RuntimeError("baseline preparation finished but the runtime graph is still missing")

            if opts["mode"] == "continue":
                ent = self.by_display.get(opts["lineage_display"])
                if ent is None: raise RuntimeError("selected lineage disappeared")
                live_dir = ent["path"]; label = ent["label"]
                manifest = json.loads((live_dir / "live_lineage.json").read_text(encoding="utf-8"))
                seed_dir = (resolve_graph(manifest["seed_checkpoint"]) if manifest.get("seed_checkpoint")
                            else RUNS / f"{label}-Seed")
                if opts["verify"]:
                    validate_lineage_files(live_dir, deep_hash=True, progress=self.post)
                else:
                    validate_lineage_files(live_dir, deep_hash=False, progress=self.post)
            else:
                seed_name = opts.get("seed_name", "Specimen-Seed")
                choices = {p.name: p for p in available_seeds(RUNS)}
                if seed_name not in choices:
                    raise RuntimeError("Selected seed is no longer available")
                seed_dir = choices[seed_name]
                label, live_dir = reserve_new_lineage()
                reserved = (live_dir,)
                self.post(28, f"Creating a separate fresh lineage: {label}")

            self.flavor(31)
            self.post(34, "Loading connectome and saved neural state ...")
            importlib.invalidate_caches()
            desktop = importlib.import_module("specimen003_desktop")
            engine = desktop.LifeLineageEngine(base_graph=desktop.BASE_GRAPH, live_dir=live_dir,
                                               seed_dir=seed_dir, specimen_label=label,
                                               reset_live=False, defer_tuning=True)
            self.post(40, f"Loaded {engine.specimen_label}: W{engine.generation:03d} at t={engine.exp.time_s:.1f}s")

            if opts["tune_cpu"]:
                self.post(42, "Tuning CPU threads to hardware ...")
                engine.configure_threads("auto", force=True, progress=self._tuner_progress(42, 25))
            else:
                self.post(62, "CPU tuning skipped; using one exact worker.")
                engine.configure_threads("1")

            self.flavor(68)
            gpu = engine.configure_gpu(opts["use_gpu"], progress=self._tuner_progress(70, 10))
            if gpu.get("enabled") and opts["tune_cpu"]:
                # Removing the CPU forgetting pass changes the worker-count curve, so
                # tune again against the actual hybrid cost rather than assuming the
                # CPU-only optimum remains optimal.
                self.post(81, "GPU hybrid accepted; narrowing CPU count around the hybrid workload ...")
                engine.configure_threads("auto", force=True, progress=self._tuner_progress(81, 12))
            elif opts["use_gpu"]:
                self.post(82, "GPU hybrid not selected; CPU exact path remains authoritative.")

            self.flavor(94)
            self.post(96, "Applying final launch profile ...")
            # The simulation worker reapplies thread masks / CUDA context in its own
            # thread.  This call only verifies the selected profile is internally sane.
            engine.exp.core.thread_info(); engine.exp.core.gpu_info()
            self.post(100, f"Ready — {engine.specimen_label} W{engine.generation:03d}. Opening Observatory ...")
            time.sleep(.25)
            self.q.put(("done", engine, opts, desktop))
        except Exception as e:
            if engine is not None:
                engine.release_process_lock()
            if reserved:
                for folder in reserved:
                    try:
                        folder.rmdir()
                    except OSError:
                        pass  # Preserve any data written before the failure.
            import traceback
            self.q.put(("error", f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=8)}"))

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait(); kind = item[0]
                if kind == "progress":
                    _, pct, text = item; self.progress["value"] = pct; self.status.set(text)
                elif kind == "done":
                    _, engine, opts, desktop = item; self.result=(engine,opts,desktop); self.root.quit(); return
                elif kind == "error":
                    self.error = item[1]; self.progress["value"] = 0; self.status.set("STARTUP FAILED — details below")
                    self.detail.set(self.error[-900:]); self._running=False; self._set_controls(False); self.begin.configure(text="TRY AGAIN")
        except queue.Empty:
            pass
        self.root.after(40, self._poll)

    def run(self):
        self.root.mainloop()
        result = self.result
        try: self.root.destroy()
        except Exception: pass
        return result


def main():
    splash = Splash()
    result = splash.run()
    if result is None: return 1
    engine, opts, desktop = result
    desktop.ObservatoryApp(engine).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
