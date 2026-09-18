# Sensory embodiment audit

The connectome is not a complete sensory or muscle model. Neuron identities and synapses alone do not specify receptor geometry, directional tuning, muscle force curves, or environmental physics. This audit distinguishes the running model from missing capabilities; it does not claim biologically complete embodiment.

## Implemented corrections

- Leg movement, hair-plate and chordotonal feedback now use ProLN/MesoLN/MetaLN annotations to distinguish fore-, middle- and hind-leg pairs. Explicit side/instance annotations preserve left/right. Unknown sides remain pooled within their known pair; unknown nerves retain the previous generic approximation.
- Chordotonal movement magnitude now accounts for elapsed simulation time. Switching from 100-ms windows to shorter slices no longer weakens the same physical movement merely because it is sampled more often. Transfer gains remain engineering approximations.
- Proprioceptive movement/contact history survives checkpoints and structural growth. Legacy saves initialize missing history from restored body/motor state without inventing a movement or contact-onset event.
- Retinal adaptation, previous facet responses, microsaccade state and sensory time survive checkpoints and compatible structural growth.
- Saved eye axes and angles retain float64 precision. Float32 rounding previously changed nearest-neighbor choices in the optional interstitial sampler, producing a different next retinal frame after resume.

## Coverage and remaining gaps

| Sense/function | Running model | Important missing information or mechanism |
|---|---|---|
| Compound vision | Spatial spectral light, occlusion, shared terrain, photoreceptor mapping, adaptation | Head-relative eye motion; independent retinal sampling clock; validated receptor geometry throughout; polarization; optional virtual sampling is speculative |
| Ambient light | Rh6-like HBeyelet input | Complete irradiance/circadian photoreception physiology |
| Odor | Small fermentation mixture across named ORN classes and local gradients | Broad chemical repertoire, turbulent plume transport, many receptor classes |
| Taste/contact chemistry | Selected leg, labellar and pharyngeal populations; sugar/water/salt representations | Full taste repertoire and precise receptor/contact locations; some represented substances are always absent |
| Temperature/humidity | Bilateral thermal dynamics and humidity-sensitive inputs | Full peripheral transduction calibration and local microclimate physics |
| Hunger/internal state | Selected nutrient-sensitive populations, physiology and feeding feedback | Complete interoceptive/endocrine physiology |
| Leg mechanics/contact | Annotated leg-pair routing, movement-rate and contact envelopes | Real joint angles, individual sensillum tuning and per-foot force/strain geometry |
| Antennal wind/gravity | Opponent JO-C/E fore-aft/bilateral deflection approximation | Actual 3-D antennal geometry; body-Z contribution; gravity magnitude and inertial acceleration calibration |
| Halteres | Flight-gated angular-speed envelope with roll/yaw bias | Signed pitch coding, receptor-specific direction, high-rate oscillation/phase and strain mechanics |
| Hearing/vibration | JO-A/B identified but deliberately silent | Acoustic field, passive receiver vibration dynamics and high-rate neural delivery |
| Wing strain | Coarse load/activity envelopes | Individual wing-hinge/strain field mapping and wingbeat phase |
| Flight pitch actuation | No direct airborne pitch torque in current motor decoder | b1/b2 phase/activity-to-wing-kinematics mapping and force/moment model |

## Evidence limits

The immutable baseline contains 205 neurons labeled haltere; 148 have generic SApp type labels. The receptor and supertype fields are empty for all 205. Entry nerves identify the appendage, but do not provide per-receptor preferred directions. Assigning neurons to signed roll/pitch/yaw channels by array order would invent anatomy.

The peripheral nerve annotations distinguish leg pairs, which supports the routing correction above. They do not establish calibrated joint tuning. Likewise, research establishes b1/b2 involvement in pitch, but does not justify replacing the missing mechanics with a host-authored upright controller or arbitrary constant pitch torque.

## Research anchors

- [Whitehead et al., 2022 — neuromuscular pitch control](https://limbs.lcsr.jhu.edu/wp-content/papercite-data/pdf/whiteheadneuromuscular2022.pdf): b1/b2 manipulations affect pitch; phase-sensitive muscle mechanics matter. This establishes a missing actuator pathway, not a ready-made rate-to-torque coefficient for this simulator.
- [Haltere connectivity atlas](https://www.sciencedirect.com/science/article/pii/S0960982225016720): receptor/central connectivity organization provides an anatomical route toward better mapping; a crosswalk to this MaleCNS graph is still needed.
- [Johnston-organ subgroup physiology](https://pmc.ncbi.nlm.nih.gov/articles/PMC4023023/): vibration-sensitive and sustained-deflection populations are distinct. Auditory channels should not be filled with generic body speed.
- [Mechanosensory timing in flight](https://pmc.ncbi.nlm.nih.gov/articles/PMC7779504/): wing/haltere phase and steering-muscle timing are important, beyond the current slice-averaged model.

## Validation scope

Tests cover leg-pair/side isolation, movement-rate consistency across time steps, sensory history round trips, legacy resume behavior, and an isolated 166,700-neuron GPU run with save/resume and hidden GUI rendering. These validate implementation consistency, not biological equivalence or learned landing behavior. Live organism checkpoints and the immutable seed are not modified by development tests.

The next substantial flight improvement requires an explicit configurable physical model for antenna/haltere geometry and wing-hinge mechanics, with provenance and uncertainty for every unmeasured parameter. A complete set of senses remains unfinished.
