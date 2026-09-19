# Embodiment research audit — 18 September 2026

The simulator does **not yet provide a research-validated adult fly body**. An adult anatomical connectome does not by itself supply receptor transfer functions, synaptic efficacy, muscle mechanics or acquired physiological state. This audit follows environment → receptor → neural population → motor output → physical feedback. Passing a software test below establishes routing or numerical consistency, not biological equivalence.

## Subsequent isolated-probe corrections

See [Autonomy probes and corrections](AUTONOMY_PROBES.md). The later pass separates contact and aerodynamic actuation, removes unsupported abdominal external roll torque, removes odor-gradient mechanical antenna drive, and routes campaniform envelopes by appendage. Descriptions below of those former defects are retained as the audit baseline; the probe report records their current status. Signed pitch feedback and phase-aware bidirectional pitch are still unresolved.

## Corrections made in the initial audit pass

1. Johnston-organ subgroup selection now uses explicit JO type annotations across both coarse auditory and wind/gravity categories. Eight JO-C cells previously excluded from deflection input are restored. The census is 137 JO-A/B, 61 JO-C, 267 JO-E and 124 other/uncertain cells. Auditory populations remain silent until an acoustic transduction model exists.
2. Receptors outside the recognized fore/middle/hind leg nerve mapping no longer receive generic all-leg movement. This removes that input from 18 leg, 53 hair-plate and 33 chordotonal cells; some belong to other organs. Their neural connections remain intact.
3. The leg motor decoder no longer assigns an arbitrary positive direction to unidentified muscles. The 115 of 381 leg motor neurons without a direction in the current decoder remain visible in unresolved telemetry. Zero contributions preserve the existing pooling denominator. This removes invented actuation; it does not supply the missing muscle map.

No steering, exploration, landing or upright behavioral controller was added. These changes require restarting the running application. Existing saved organisms and the immutable seed were not edited.

## System-by-system findings

### 1. Halteres and flight stabilization — major unresolved defect

`proprioception.py` provides a flight-gated angular-speed envelope and a roll/yaw bias. A controlled test with otherwise identical states gives **identical sensory input for positive and negative pitch rotation**. The graph has 205 haltere-labeled cells, 148 with generic SApp labels. Current annotations do not establish individual directional tuning.

The [haltere atlas](https://pmc.ncbi.nlm.nih.gov/articles/PMC12872070/) describes structured receptor populations and connectivity. Its [annotation CSV](https://raw.githubusercontent.com/serene-da1/Dhawan-et-al-2025-/main/Data/Annotation%20tables/final_haltere_clusters.csv) contains root IDs, morphological clusters and campaniform fields. Those are useful anatomical evidence, but this pass has not established a crosswalk to the local MaleCNS neuron IDs or measured signed response functions.

Required before validation: explicit haltere oscillation, Coriolis/strain mechanics, receptor field mapping, phase-sensitive delivery and separate roll/pitch/yaw perturbation tests. Assigning axes by neuron array order would fabricate anatomy.

### 2. Wings and flight muscles — coarse approximation

`fly_body.py` converts pooled motor rates to power, thrust and torque. Reduced b1/b2 pitch actuation exists; bidirectional phase-sensitive control remains incomplete. The previous unsupported b3 pitch coupling has already been removed. Current gains and yaw/roll roles are not a measured aerodynamic model. Abdominal activation also contributes external roll torque without an articulated reaction model.

`proprioception.py` derives wing load from total commanded force, including leg force. Its 426 campaniform cells span wing, haltere and leg nerves but share a support/load envelope. Ground contact can therefore stimulate inappropriate appendage populations. This remains unresolved; replacing it requires separate actual organ strain, not another global gain.

[Pitch experiments](https://cohengroup.lassp.cornell.edu/userfiles/pubs/Whitehead_SciAdv_2022.pdf) and [wing proprioceptive anatomy](https://pmc.ncbi.nlm.nih.gov/articles/PMC12154767/) constrain pathways but do not supply the simulator's rate-to-torque coefficients. Acceptance requires unilateral muscle perturbations, wing kinematics and measured force/moment comparisons across phase and power.

### 3. Legs, joint proprioception and contact — incomplete body

Leg nerve routing distinguishes ProLN/MesoLN/MetaLN and annotated sides. The corrections above remove unjustified generic input and default forward actuation. However, scalar leg motor commands still stand in for joint positions. There are no articulated legs, per-foot collisions or muscle length/velocity dynamics. Flexion at different joints is collapsed into the same movement scalar.

[Leg proprioceptor experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/) distinguish position, directional movement and vibration channels. Current absolute activation/movement envelopes cannot reproduce those responses. Surface adhesion and passive body alignment are synthetic contact approximations, not validated stance mechanics.

Acceptance: joint-resolved body geometry; muscle attachment/action map; individual foot forces; separate position, velocity and vibration stimulus sweeps; no movement from unresolved muscles. A gait generator must not substitute for connectome output.

### 4. Antennal wind, gravity and hearing — partial routing

Explicit JO-C/E subgroup selection is repaired. [Subgroup physiology](https://pmc.ncbi.nlm.nih.gov/articles/PMC4023023/) supports opponent sustained deflection responses. The current antenna model combines airflow, a synthetic volatile-carrier cue and normalized gravity, projects only body X/Y, and omits physical antennal mechanics. It discards gravity magnitude and body-Z deflection. Acoustic vibration is absent; JO-A/B silence is a missing sense, not evidence of a correct hearing model.

Acceptance: antenna geometry and compliance, relative air velocity, gravity and inertial acceleration, passive vibration, frequency/intensity tuning and signed directional tests. Chemical gradients must not produce mechanical deflection without a modeled carrier flow.

### 5. Compound vision and ambient light — useful optics, incomplete transduction

`compound_eye.py` samples shared spectral world geometry with occlusion and mapped optic columns. Retinal history survives resume. Actual head motion is missing from eye orientation. Some field geometry and optional virtual sampling remain approximations. Photoreceptor input is converted to rate bins and injected into the generic spiking core; real fly photoreceptor voltage signaling is graded, as characterized in [photoreceptor measurements](https://pmc.ncbi.nlm.nih.gov/articles/PMC2232470/).

Acceptance: measured eye/head geometry, moving-head visual tests, receptor voltage/adaptation dynamics, calibrated contrast and temporal response. Looming should arise from image motion. HBeyelet illumination is only partial coverage of ambient/circadian photoreception; polarization and complete irradiance pathways remain absent.

### 6. Head, neck and abdomen — command proxies

Pooled motor activity drives observer pose and sensory envelopes. There is no complete articulated head/neck model; neck feedback follows commanded activation rather than actual stretch. [Neck motor experiments](https://www.nature.com/articles/s41586-024-07222-5) show why posture and proprioceptive context matter.

Acceptance: physical joint motion, muscle actions, inertial reaction, receptor stretch and head-relative eyes/antennae. A drawn head pose is not physical sensory feedback.

### 7. Touch and strain — nonlocal stimulation remains

The 2,206 mechanosensory-bristle cells span multiple appendages, yet receive a global airspeed/contact-onset envelope. Wing and campaniform load also use mixed-force proxies. A body landing should not automatically imply every mapped bristle or strain receptor was stimulated.

Acceptance: organ-local contact/airflow, receptor locations and deformation. The wing and leg research cited above provides anatomical constraints; receptor-specific thresholds remain to be sourced and calibrated.

### 8. Olfaction — narrow synthetic chemistry

`life_sensory.py` drives a small named ORN panel using handwritten response weights and synthetic fermentation mixtures. [DoOR 2.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC4766438/) provides empirical response information, but the current coefficients are not a direct validated fit to those data. Resource falloff and hashed chemical recipes do not constitute physical plume transport.

Acceptance: versioned receptor/odor response data, concentration units, transport, adaptation, inhibitory responses and held-out odor tests. Unrepresented receptor classes must remain explicit in coverage reports.

### 9. Taste, proboscis and ingestion — selected pathways

Selected leg, labellar and pharyngeal populations receive sugar/water/salt proxies; bitter/high-salt channels have no active environmental source. Contact is not localized to individual feet or sensilla. Feeding motor phases and saved cibarium fill are more specific than a single feed command, but remain reduced mechanics. The 39 aPhM neurons receive a shared fill/empty proxy.

The [adult gustatory connectome](https://www.janelia.org/publication/the-complete-gustatory-connectome-of-adult-drosophila-reveals-how-taste-guides-feeding) offers a basis for improving this mapping. Acceptance requires receptor-level contact chemistry, explicit ingestion mechanics, motor population provenance and separate chemical/mechanical stimulus tests. Nutrient availability must not directly select a motor action.

### 10. Temperature and humidity — approximate transduction

Named VP-associated populations receive bilateral thermal/humidity input. Their broad modality organization is supported by [thermo/hygrosensory circuit research](https://pmc.ncbi.nlm.nih.gov/articles/PMC7443704/). Temperature thresholds, firing gains and mixed cooling/dry coefficients are engineering choices, not established quantitative fits. World heat transfer and sensillum moisture dynamics remain simplified.

Acceptance: independent warm/cool/dry/moist ramps, adaptation and cross-sensitivity checks, measured transfer functions and local heat/moisture physics.

### 11. Internal physiology — partial and partly imposed

`homeostasis.py` represents hunger, energy and selected nutrient signals. [DH44 research](https://pmc.ncbi.nlm.nih.gov/articles/PMC4697866/) supports nutrient sensitivity, but does not validate the current hunger-times-ingestion rate formula. Exact NPF/Hugin population targeting and response laws need separate validation. The externally supplied ER5 keepalive waveform is an imposed neural-state input. Automatic food rescue is a world intervention. Both must be distinguished from emergent physiology.

Acceptance: population-specific evidence, energy/ingestion accounting, controlled internal-state response tests and explicit configuration/provenance of experimental interventions. Global host reinforcement remains disabled.

### 12. Neural dynamics, plasticity and growth — model assumptions

The full graph preserves anatomical connectivity, but generic LIF neurons, transmitter sign assignments, count-to-weight conversion and bounded plasticity are modeling assumptions. Numerical CPU/GPU equivalence validates the same equations, not biological accuracy. Anatomical adulthood alone does not guarantee adult behavior under these dynamics.

Automatic duplication has been retired in favor of staged growth proposals. A proposal is not demonstrated beneficial growth; replay/selection and safe promotion remain unfinished. Sensory amplification must not serve as a fitness proxy. Acceptance requires circuit response comparisons and independent behavioral/physiological validation before growth claims.

### 13. World and body scale — synthetic environment

The world uses small spherical habitats, synthetic atmosphere/chemistry and dominant-body gravity. Shared world-pack spectra improve visual consistency; displacement meshes and chemical metadata are not a complete physical terrain/atmosphere implementation. Texture detail does not repair a missing body model.

[NeuroMechFly v2](https://www.nature.com/articles/s41592-024-02497-y) demonstrates an articulated physics approach with sensory feedback. Its controllers are not a substitute for this connectome. Acceptance here requires consistent units, collision geometry, air/force scaling and contact mechanics before comparisons with fruit-fly trajectories are meaningful.

## Verification and remaining work

Passed: explicit JO population selection and opponent stimulation; auditory isolation; fore/middle/hind routing isolation; absence of generic input to unmapped limb receptors; sensory-only stimulus targets; unresolved-muscle neutrality and telemetry; sensory-history persistence. A separate isolated 166,700-neuron GPU run completed twelve slices, save/resume, and hidden rendering of sky, HUD, eyes and life map. Existing live checkpoints were not advanced or saved.

Observed diagnostic failure: opposite pitch rotation produces identical mechanosensory input under a controlled neutral-body stimulus. The audit does not claim that the three routing corrections cure corkscrewing or produce exploration.

Implementation order: establish appendage/muscle/receptor identity and provenance; add physical flight and joint state with localized feedback; validate signed stimulus responses; then calibrate environmental transduction and circuit dynamics. The unresolved systems above are substantial implementation work, not completed fixes. No system should be labeled biologically validated solely because its neurons receive nonzero input.
