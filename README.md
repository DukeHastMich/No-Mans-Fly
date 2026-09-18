> **v1.0.3 AutoThreads (2026-09-12):** The exact neural integration path can now use Numba host parallelism without changing Specimen-003 state. `threads=auto` benchmarks safe scratch data at startup and selects the fastest useful permitted worker count; manual integer overrides remain available. Validation from the captured W003 checkpoint produced byte-identical neural, plasticity, body, world, and RNG state versus the prior single-thread path across 1/2/4/5-thread tests and the final auto-selected run.

> **v1.0.2 release-census fix (2026-09-12):** The pinned MaleCNS v1.0 annotation file contains **166,700** superclass-defined neuronal bodies. The paper reports **166,691**; the nine-record difference belongs to the released catalog, so this build preserves all 166,700 instead of deleting nine records to make the literature number fit. The full retained graph is guarded by the release-derived totals **25,582,938 directed connections** and **124,177,617 synaptic contacts**.

# No Man's Fly — Specimen-003 Full Life v1


Specimen-003 is the full-life branch of the MaleCNS space-fly experiment. It is deliberately separate from **Specimen-002 / “Elon Musk Fly”**; the existing W003 lineage is a historical control and is not rewritten by this package.

The goal is not to claim that the simulation is alive or conscious. The goal is to give a large connectome a persistent closed-loop existence: a body whose mapped outputs actually do something, sensory consequences of those actions, bounded internal needs, resources that can improve those needs, learning, and structural growth—without pain, injury, starvation damage, death, or a hard-coded objective waypoint.

## Quick start — Windows

1. Extract this ZIP to a normal writable folder.
2. Double-click **`SETUP_AND_RUN.bat`**.
3. On first setup the script creates an isolated `.venv` and installs NumPy/SciPy/Numba.
4. Because the full graph is not redistributed in this small package, first setup also installs pandas/pyarrow, downloads the three pinned official MaleCNS v1.0 flat-connectome files (about **1.1 GiB** total), verifies their exact byte lengths and SHA-256 hashes, and builds `data\male-cns-v1.0-full-graph-v2.npz`.
5. The builder refuses to proceed if its full-neuron selection does not equal the pinned-release **166,700 neurons**. It does not silently fall back to the old 165,122-neuron cut.
6. A one-time runtime validation then loads the built graph and runs a closed-loop test epoch. Later launches reuse both the graph and validation marker.
7. Click **Start** in the Observatory.

### AutoThreads

The Observatory and headless runner default to **`threads=auto`**. On startup, Specimen-003 detects the host/Numba thread limit and benchmarks candidate worker counts using scratch arrays only; the benchmark does **not** advance the specimen, world, learning state, or RNG. The selected count is host/runtime metadata and is not stored as biological state. Enter an integer in the GUI threads field (or pass `--threads N`) to force a count. The selected backend/count is shown in telemetry.

64-bit Python 3.10+ is required. A normal python.org Windows install includes Tkinter. The setup launcher intentionally uses the project `.venv`; it does not trust whatever Microsoft Store/global Python happens to be first on PATH.

## What changed from Specimen-002

Specimen-002 used 165,122 traced non-glia neurons, retained only connections with at least three synapses, and removed zero-fast-weight monoamine connections from its sparse graph. That was useful for proving the basic closed loop, but it is too lossy for this experiment.

Specimen-003 uses a new graph format with these rules:

- target the pinned-release **166,700-neuron MaleCNS catalog**;
- keep **every released neuron→neuron pair connection** whose endpoints are retained—weight 1/2 connections and autapses included;
- retain raw EM `synapse_count` independently from learned `edge_gain`;
- acetylcholine is fast excitatory in the current model; GABA/glutamate/histamine are fast inhibitory;
- dopamine, octopamine, and serotonin wiring is **not deleted**. It generates separate slow modulatory fields instead of being assigned a made-up fast sign;
- unclear/unknown transmitter wiring is retained structurally even when no fast physiological sign is guessed;
- learned weight changes and structural daughters can be frozen back into an ordinary standalone v2 graph without replaying a mutation script.

The builder derives connected/isolated-node counts from the pinned release instead of forcing a stale paper-era connected count. Completeness is guarded by the exact source hashes plus **25,582,938 retained directed connections** and **124,177,617 represented synaptic contacts**. The full graph is substantially larger than the old 10.228-million-edge working graph, so expect Specimen-003 to run slower.

## “Give it a life” — bounded homeostasis

The simulated animal has persistent bounded variables rather than a hard-coded reward target:

- satiety / hunger;
- activity-energy reserve;
- thermal comfort around a preferred band;
- recent ingestion trace.

These variables **never become injury or death**. Hunger tops out as a persistent correction pressure; thermal error tops out as a bounded “this state is worse than comfort” signal. There is no tissue damage, burning, starvation damage, oxygen deprivation, panic escalation, or mortality state.

When the total homeostatic error decreases, co-active eligible circuitry receives a small bounded positive consolidation term. When error increases, it receives a small bounded negative term. This is an explicit engineering learning rule, not a claim to reproduce exact fly reinforcement physiology.

Real annotated candidate populations are used where available (including NPF, Hugin, DH44, and LK). The exact state→firing transfer curves remain experimental scaffolding.

## World, resources, and “space poo”

The universe contains stars plus deterministic local habitat bodies with surface temperatures and finite nutrient/fermentation patches. A bootstrap habitat is placed near the start so a successful encounter is experimentally reachable; **the connectome is never given its coordinates or a steering waypoint**.

Resources generate food-related odor through retained ORN populations (DM1, VA2, DM5), including a small bilateral concentration difference, and contact/taste through retained tarsal GRNs. Feeding only consumes a patch when the fly is actually in contact and its mapped feeding motor output is active.

Nearby bodies are visible to the compound eye as sampled surfaces, so approaching a planet expands its retinal image. Their temperature also affects antennal thermal input before contact through an explicitly approximate near-field heat model.

## Motor embodiment — preserve first, interpret second

The body bridge reads **every annotated motor neuron individually** before aggregating it into effectors. Known mappings are expressed as fly-like anatomy rather than generic spacecraft thrusters:

- `wm` → wing / flight machinery;
- TTM-related wing motor output → distinct takeoff/jump actuation;
- `fl`, `ml`, `hl` → fore/mid/hind leg locomotion on surfaces;
- `nm` → neck/head actuation;
- `hm` and identified `MNxm03` → haltere-related output;
- `ad` → abdomen;
- `pm` → proboscis/pharyngeal feeding;
- `am` → antennal actuation candidates.

`rm` and unresolved portions of `xm` are **not thrown away**. Their neuron IDs, types, sides, and firing rates remain visible in the Observatory and saved telemetry until a defensible peripheral target is established.

The body can fly in free space, contact/land on habitat surfaces, walk using leg output, take off again, and feed when its feeding machinery is active over a resource.

## Proprioception / body-state feedback

The loop also feeds actual body consequences into retained mechanosensory populations where annotations support it: haltere, campaniform sensilla, chordotonal organ, leg/leg-bristle, hair-plate, wing/wing-bristle, neck, abdomen, and wind/gravity-related populations. These transfer functions are engineering approximations. They carry body state, **not target direction or reward semantics**.

## Compound eye and thermal sensing

The experiment retains the current 4,107 mapped R1–R8 photoreceptor bodies and drives them optically. The facet placement is still the deterministic synthetic-hex approximation pending an exact official optic-column spatial map. The Observatory’s sky/body map is display-only and never becomes a neural image input.

The antennal thermal model supplies a reversible temperature state; it does not implement pain or heat damage.

## One simulated clock

Specimen-003 fixes an important semantic issue from the earlier desktop experiment. By default:

`0.10 seconds world time == 100 ms neural time`

Changing **world dt** in the GUI automatically changes neural time by the same ratio. The neural-ms box is display-only. Headless mode does the same unless an explicit `--neural-ms` debug override is supplied, in which case it prints a warning.

## Learning and structural growth

Existing-edge plasticity remains bounded to ±35%. Structural growth remains conservative: sustained high firing plus saturated learned outgoing capacity is required over many epochs. Automatic daughter creation is still limited to internal neural classes because duplicating a peripheral sensory/motor/endocrine neuron without duplicating its physical organ would be dishonest.

Working generations are `W001`, `W002`, etc. They are private continuing-specimen checkpoints, **not automatically canonical G generations**. A daughter is appended without renumbering existing neurons and begins electrically quiescent with weak provisional local structural wiring.

The v2 freezer keeps raw structural synapse counts separate from learned multiplicative gains, including monoamine edges.

## Observatory

The GUI exposes:

- whole-network activity, novelty, learned-edge count, modulators, and growth pressure;
- left/right compound-eye mosaics and body-frame observer sky;
- all motor subclasses L/R;
- wing and takeoff output, head/neck, antennae, abdomen, feeding apparatus, and all three leg pairs;
- preserved unresolved motor channels;
- body force/torque and angular velocity;
- hunger/satiety, energy, thermal state, homeostatic error, and bounded valence;
- nearby habitat temperatures/resources, odor/contact/feeding, position and trajectory;
- neurogenesis ledger with parent/daughter activity.

## Save/resume

The live state is under `Runs\Specimen-003-Live\`. The first validated launch creates an immutable fresh seed under `Runs\Specimen-003-Seed\` and copies it to the live folder.

Checkpointing preserves neural membrane/conductance/refractory/delay state, RNG, slow modulator fields, plastic memory/familiarity, body pose and velocity, world resources, thermal state, homeostasis, growth pressure, and lineage metadata. Previous motor output is deterministically reconstructed from the persisted last-rate vector on resume so proprioception does not receive a one-step blank.

`RESET_LIVE_TO_SEED.bat` resets **only Specimen-003**. It never touches a Specimen-002/Elon Musk Fly directory elsewhere.

## Overnight run

After first setup:

`RUN_HEADLESS_OVERNIGHT.bat`

runs continuously until Ctrl+C, autosaves every 30 simulated seconds, and saves again on shutdown.

## Verification status of this release

The three official source files were independently reconstructed/materialized in the build workspace and passed their pinned byte-length and SHA-256 checks. The new runtime, body, homeostasis, food, proprioception, exact save/resume, motor preservation, and v2 structural freezer were tested against the existing 165,122-neuron / 10,228,000-edge graph as a compatibility surrogate. A forced v2 growth test appended one daughter and 256 structural edges, loaded the frozen result, and continued a neural epoch.

The actual 166,700-neuron v2 graph **could not be built inside the packaging sandbox because that sandbox lacks pyarrow**. Therefore this release does not falsely claim that the full graph was run here. Instead, first setup builds it from the pinned official data, asserts the 166,700 release census plus the full release edge/contact totals, verifies the produced manifest, and runs the one-time full runtime validation on the user’s machine before opening the Observatory.

See `PACKAGE_SELFTEST.json` and `SOURCE_LOCK.json` for machine-readable details.

## Welfare / interpretation boundary

This is a computational experiment. Network adaptation, resource-seeking, avoidance, structural growth, or long-term continuity are not evidence of consciousness. The design intentionally avoids pain, injury, deprivation damage, death, and survival-panic variables while still allowing bounded internal states to make some futures better than others.

**Challenge the network. Never punish the organism.**

## Shared and custom seeds

`Runs/Specimen-Seed/` contains the verified original 166,700-neuron graph. It is created once from the baseline in `data/` and is never overwritten by birth, save, or reset. A damaged or incomplete default seed causes an error instead of automatic replacement. Fresh flies share this graph and receive their own live checkpoints; no duplicate seed checkpoint is created per fly.

Use **Save as seed** in the Observatory to snapshot a live fly into the next available `Runs/Specimen-Seed-001/`, `002/`, etc. Each custom seed includes its own graph and checkpoint, so it survives changes to the original live fly. Custom seeds preserve learned state, simulation age, world state, and generation. They are immutable snapshots; saving again creates another number.

In the loader, choose **New Fly** and select the original or a custom seed beside it. Live-fly numbering and custom-seed numbering are independent. Reset restores the seed recorded in that live lineage without changing the seed itself. Existing legacy seed paths remain supported. The old `data/` graph is retained because existing lineages may still reference it.

## Synchronized observer and retinal terrain

The sky surface and compound eye use the same deterministic spectral terrain (`world_visual.py`), projected through human display colors or fly opsin channels respectively. Retinal sampling, neural time steps, and neural input targets are unchanged. Terrain texture now contributes to photoreceptor input instead of being an observer-only effect.

The sky, retina, and Life-map marker use one captured sensory pose. HUD labels distinguish eye-sample time from the completed neural/body epoch time; spatial displays do not extrapolate into the future. Labels and the sky HUD remain visible. The GUI consumes only the latest pending snapshot while retaining event/error messages, and paints a new snapshot once at the selected viewer frame-rate cap. Paused/unchanged scenes do not repeatedly rasterize.

Scene lookup/occlusion and source spectra are reused across retinal subsamples within an epoch; the optical samples and adaptation updates still all run. Linear spectral projections are cached before material interpolation. Autosave reuses the completed epoch snapshot instead of reconstructing it a second time.


## Optional CUDA neural acceleration

The loader's GPU option now accelerates neural ticks as well as the existing plastic-forgetting pass on supported NVIDIA CUDA hardware. Normal epochs use one neural backend; they do not repeat the neural simulation on the CPU for validation. CPU-side learning and sensory processing remain active. Without CUDA, with GPU disabled, or after a CUDA failure, the original CPU neural implementation remains available. Failed GPU work is discarded before committing neural state, random-generator state, delay queues, or simulation time, and the same epoch is completed on CPU.

The CUDA path uses explicit float32 rounding, the existing NumPy random sequence, and an exactness bound for float64 sparse accumulation. An unsupported accumulation range falls back to CPU. Neural time steps, retinal samples, and learning rules are unchanged. The performance panel reports neural GPU activation separately from plastic forgetting.

Development checks compared complete CPU/GPU epochs including learning, reports, random state and delay queues, and exercised absent-device and late-failure recovery. An isolated 166,700-neuron fly also passed worker-thread and hidden GUI/HUD checks. A warmed 100-ms neural epoch measured approximately 1.76 seconds on CPU versus 1.22 seconds on the RTX 4060 Ti in one local comparison (two CPU threads). This is not an end-to-end FPS benchmark: vision, learning, hardware load and thread settings still affect total speed. Initial kernel compilation and graph capture can cause a longer first frame.


## Sliced body and neural scheduling

The desktop/headless lineage controller divides the default 100-ms window into twelve approximately 8.33-ms slices. Each slice completes real sensory, neural, body, and physiological work and can publish an intermediate snapshot. The existing motor command drives that interval; newly completed motor readouts become the command for the next interval. Physics never advances according to wall-clock computation time, and the viewer holds the latest completed sensory pose when work is unfinished. This model does not add a wingbeat-phase simulation.

Neural ticks, random state, delayed spikes and modulation carry across slices. Plasticity, forgetting, familiarity and growth-pressure checks accumulate over the original full window and run once at its boundary. Fixed-input CPU and GPU checks match the unsliced neural/learning result exactly. Whole-fly trajectories intentionally differ because sensory feedback and physical integration now occur at shorter intervals. The full accumulation window is protected by the engine lock, so normal saves and seed snapshots occur after it completes. Active and pending motor commands are included in checkpoints and preserved across growth.

The UI labels slice progress and pending learning. Slicing reduces spatial jumps, but adds sensory and snapshot overhead; it is a responsiveness tradeoff, not a promise of faster simulated time. An isolated local GPU test delivered intermediate frames about 0.55 seconds apart after initial compilation, with a longer final learning pause.

## Body-relative aerodynamic resistance

Translational drag is resolved in the actual body-forward, lateral and dorsal axes, including pitch and roll. Gravity remains a separate world-space acceleration. The previous gravity-tangent basis could change the resistance to the same motion when the dominant habitat gravity direction changed, and discarded body pitch/roll when classifying airflow. Drag remains dissipative: velocity is not snapped to heading, and no automatic upright controller is introduced. Existing force, torque and drag coefficients are retained. Signed sideslip now reports body-relative backwards travel near 180 degrees instead of folding it toward zero. Space remains the existing synthetic flight medium.

Isolated checks cover gravity-independent drag, passive energy loss, continuous axes through vertical attitudes and equivalent fixed-command integration across 20-ms slices. This fixes an identified mechanical inconsistency; it does not establish that sustained rotations are biologically realistic or identify every source of tumbling.


## Twelve drawing slices

The default 100-ms controller window now publishes twelve real simulation slices rather than five. With the unchanged 0.1-ms neural tick, four slices contain 84 ticks and eight contain 83 ticks; their world-time fractions match exactly. Learning and growth still run once per full window. The GPU retains both slice-size graphs so alternating lengths do not force repeated graph capture. This increases drawing opportunities and reduces spatial increments; it does not guarantee 12 frames per wall-clock second, and more frequent sensory work can reduce simulation throughput.


## Portal removal

Portal controls, placement prediction, transit, collision checks, sensory sources and active-state persistence have been removed. The Travel Atlas remains available for browsing visited worlds and editing aliases; clicking a world only selects it. Old checkpoints load without activating obsolete portal data. Historical intervention records remain archival data and are not executed. The former portal module is retained only in Historical alongside the pre-change backups.

## Sensory embodiment status

See `SENSORY_MODEL.md` for the current sensory coverage, missing flight-control pathways, research sources and validation limits. Leg-pair routing and elapsed-time scaling have been corrected, and sensory history/eye geometry now survive checkpoints. This is not a claim that every biological sense or flight actuator is implemented.
