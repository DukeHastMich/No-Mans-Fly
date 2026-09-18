# World pack integration: shared spectral surfaces

The optional extracted assets/No_Mans_Fly_WorldPack_v2 directory supplies six
512x256x10 reflectance fields to both eye sampling and observer rendering.
The bootstrap uses Earthlike Rocky; other body IDs deterministically select one
of the six templates. Radius and existing scalar albedo still vary per planet.
No arbitrary RGB recoloring is applied separately from the fly's spectra.

Numeric data is read with allow_pickle=False and checked for matching wavelengths,
shape, finite values and reflectance range. Supplied Python files are not executed.
Template arrays and receptor/display projections are cached and shared among
planets. Absence of the pack at startup retains the procedural surface fallback;
a present but corrupt pack raises an error instead of silently changing spectra.
Restart after adding, removing, or updating a pack.

This first integration uses spherical geometry consistently for eye, viewer and
contact. Elevation, OBJ meshes, normal maps, atmosphere/composition, environmental
fields and resource points are not activated. In particular, the gas giant is
currently a cloud-pattern visual template on the existing habitat physics, NOT
a simulated non-landable gas planet. These environmental changes require a
coordinated world-model migration. No existing checkpoint, resource or seed data
is rewritten. Rotation/palette instance overrides are not yet supported.

The sampler follows the pack reference longitude/latitude convention. Supplied
map borders are not perfectly seamless; original assets are preserved. No claim
of faster simulation is made: the pack increases texture resolution, while
neural and per-ray costs remain. Twelve-slice full-CNS GPU integration and hidden
UI rendering passed; a numerical test verifies projected receptor channels equal
projection of the same sampled spectra for all six templates.

Git exceptions retain the pack's required small spectral NPZ and temperature NPY
files without unignoring brain graphs or organism checkpoints. The duplicate ZIP
is ignored; the extracted runtime assets remain publishable.
