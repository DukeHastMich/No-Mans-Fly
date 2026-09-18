# No Man's Fly WorldPack v2

This is a **simulation asset pack**, not concept art.

Six reusable templates are included: Earthlike Rocky, Marslike Rocky, Moonlike Rocky, Gas Giant,
Trans-Neptunian, and Rocky VI: The Voyage Home.

Each solid template includes displaced OBJ surface/collision meshes, aligned albedo/elevation/roughness/material/normal maps,
a numeric H×W×10 spectral-reflectance field at 330, 345, 355, 375, 405, 437, 478, 508, 550, 600 nm, UV reflectance, moisture,
volatile-emission, food, mineral-resource and temperature fields, discrete resource points, and a JSON sidecar.

The gas giant uses cloud/atmosphere shell meshes and explicitly disables landing/contact surface physics.

## Authoritative-coordinate rule

Viewer rendering, Yar's retinal ray hits, spectral sampling, environmental fields and contact/collision must all use the
same longitude/latitude UV coordinate and the same elevation field. There is no separate observer-only terrain.

## Neural-signal rule

Composition/material metadata does not directly become neural activity. It generates environmental conditions/source fields;
the existing sensory models must measure those conditions.

## Per-planet instances

Planet instances are intended to store only template ID, size/radius, rotation, palette, and environmental variation.
Examples are under `examples/instances/`.

`worldpack_loader.py` is a CPU reference sampler. `VALIDATION.json` records dimensions, ranges, mesh counts, seam checks,
and generated resource-point counts.
