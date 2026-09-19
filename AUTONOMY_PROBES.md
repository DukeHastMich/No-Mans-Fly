# Isolated autonomy probes and corrections

These are causal measurements of the current simulator, not measurements of a biological fly. No live organism or seed was modified.

## Protocol

The 166,700-neuron baseline graph was reset to resting state for each trial. Plasticity and the ER5 keepalive scaffold were disabled. Seventeen annotated haltere/JO groups were stimulated at 5, 25 and 75 Hz per cell, for 100 ms followed by 100 ms without stimulation, using random seeds 7 and 19. Two unstimulated controls were included: 104 trials total. Group sizes differ, so equal per-cell rate does not mean equal total input. Unknown sides were not fabricated.

Separately, each of 815 motor neurons was set to 80 Hz against unpowered and 40-Hz power-muscle backgrounds, both grounded and airborne: 3,260 decoder interventions. These measure the motor decoder, not neural propagation or a complete flight trajectory.

## High-dose sensory responses

Ranges below span the two random seeds during the 75-Hz pulse. Motor torque is in normalized model units. The control produced no active neurons or torque.

| Stimulated population | Cells | Active neurons | Pitch torque | Yaw torque |
|---|---:|---:|---:|---:|
| JO-C/L | 39 | 300 to 1887 | -0.0002733 to -0.000127 | -5.044e-05 to -2.331e-05 |
| JO-C/R | 22 | 88 to 182 | 0 to 0 | 0 to 0 |
| JO-E/L | 157 | 3273 to 3549 | -0.005252 to -0.0003992 | 3.252e-06 to 0.0002968 |
| JO-E/R | 110 | 2721 to 3378 | -0.0002414 to -0.0001072 | -2.221e-06 to 6.238e-05 |
| haltere/SApp/L | 75 | 1499 to 2526 | -0.002333 to -0.001208 | 0.0006961 to 0.0008578 |
| haltere/SApp/R | 73 | 2518 to 2614 | -0.002243 to -0.002213 | 0.0005308 to 0.001685 |
| haltere/SNpp12/unknown | 2 | 401 to 1622 | -0.0007017 to 0 | 0 to 0.0001957 |
| haltere/SNpp14/unknown | 6 | 20 to 22 | 0 to 0 | 0 to 0 |
| haltere/SNpp15/unknown | 6 | 26 to 28 | 0 to 0 | 0 to 0 |
| haltere/SNpp20/unknown | 6 | 7 to 10 | 0 to 0 | 0 to 0 |
| haltere/SNpp21/unknown | 4 | 18 to 19 | 0 to 0 | 0 to 0 |
| haltere/SNpp23/unknown | 6 | 6 to 6 | 0 to 0 | 0 to 0 |
| haltere/SNpp25/unknown | 7 | 13 to 13 | 0 to 0 | 0 to 0 |
| haltere/SNpp34/unknown | 8 | 14 to 16 | 0 to 0 | 0 to 0 |
| haltere/SNpp35/unknown | 6 | 23 to 28 | 0 to 0 | 0 to 0 |
| haltere/SNxx25/unknown | 2 | 3 to 3 | 0 to 0 | 0 to 0 |
| haltere/untyped/unknown | 4 | 6 to 8 | 0 to 0 | 0 to 0 |

## Interpretation

The populations have distinct effects, and some responses vary substantially between seeds. A short negative response is not proof that a neuron has no function. Testing resting state alone does not characterize an active flight network; longer trials, background-state sweeps, latency resolution and more repetitions remain necessary.

The decoder intervention sweep confirms that b1/b2 can supply negative body-Y (nose-up) pitch, but no tested motor channel supplies positive aerodynamic pitch. Choosing sensory signs to compensate would fit a deficient actuator. Receptor preferred direction still requires peripheral anatomy/physiology; these probes do not establish it. The [haltere connectivity atlas](https://pubmed.ncbi.nlm.nih.gov/41564880/) provides relevant anatomy, but a validated crosswalk and phase-sensitive mechanics remain unfinished.

## Corrections in this pass

- Split flight and contact actuator components. Wing activity no longer changes the calibration of leg force. Contact force/torque cannot continue in free flight. A bounded final contact interval supplies a reduced launch impulse. Jump force uses the existing full-power reference scale, expressed in contact units; jump muscle/contact dynamics still need calibration.
- Removed direct abdominal activation-to-whole-body roll torque. Abdominal neural activity and the effector remain; articulated inertia and aerodynamic reactions are not implemented.
- Removed chemical-gradient vectors from antennal mechanical stimulation. Chemical signals still drive their olfactory pathways. A real ambient wind field is not yet present.
- Split campaniform load envelopes by annotated appendage nerve. Ground support no longer directly stimulates wing/haltere strain populations, and mixed leg force no longer drives the wing-load envelope. These are still reduced organ-level envelopes, not measured per-receptor strain.

## Verification and limits

Mechanical regressions check independent force scaling, absence of airborne contact actuation, absence of abdominal external torque, and organ-local ground strain. Full integration checks cover twelve neural/physics slices, save/resume and hidden GUI rendering. The probe outputs include every motor rate, stimulus ID and recovery measurement for review.

Signed haltere pitch feedback, bidirectional phase-aware wing mechanics, articulated leg/head proprioception, localized touch, acoustics and the remaining sensory audit items are NOT completed by this pass. No exploration, target seeking or upright behavior controller was added.

## Source fingerprints

- fly_body.py: `9e0835dee9126ab0f3cfc6da62551238ee9a044c0acc0f5b30211ec37485e6d8`
- proprioception.py: `d29c949a1367288bee4b7be1eb83aee6e0312e5002d7fcf95c7db2d5d9cadb18`
- living_universe.py: `490e274d875407ddfc20b357750e0cd638dc8b1fd35a60119eb8adb63964980b`
- specimen003_experiment.py: `dfab2436777b4953b4101fa06c3f37e6a6a31e10e32befeaaaf8836c22b68b76`
