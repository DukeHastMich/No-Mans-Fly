# Reduced flight mechanics and actual atlas traces

This patch restores a missing motor-to-mechanics path; it does not establish that
the resulting trajectories are biologically validated fruit-fly flight.

Bilateral b1/b2 recruitment contributes a reduced pitch command, weighted by
power on each wing. Negative body-y torque is nose-up (x forward, z dorsal).
The coefficient 0.018 is an engineering calibration, not a measured force curve.

The previous b3 antagonist term has been removed. In a saved live state, b1/b2
were silent and right b3 fired at approximately 120 Hz. That term alone generated
continuous positive pitch torque and contributed to a corkscrew. Inferring a
scalar pitch antagonist from b3 hinge anatomy was not justified. b3 is retained
in neural state, telemetry and the existing turn-recruitment calculation.

This leaves a limitation: a validated bidirectional, phase-aware pitch actuator
is still missing. b1/b2 rate recruitment alone does not solve it. No tonic-rate
subtraction, randomized steering, heading target or exploration controller was
added to conceal persistent motor output. Removing an artificial torque does
not guarantee exploration, landing or feeding.

Existing muscle-specific yaw torque is now multiplied by mean wing power, so
steering activity cannot supply aerodynamic yaw while the wings are unpowered.
The existing thrust and drag envelopes are retained pending aerodynamic
calibration. This patch does not claim to solve every cause of sustained flight,
missing sensory feedback, or learned behavior. Unknown muscles remain unresolved.

Research basis: Whitehead et al., Neuromuscular embodiment of feedback control
elements in Drosophila flight (2022), establishes b1/b2 contributions to pitch:
https://cohengroup.lassp.cornell.edu/userfiles/pubs/Whitehead_SciAdv_2022.pdf
The reduced rate mapping and numerical gains above are model assumptions,
not coefficients measured in that paper.

The atlas now uses equal X/Y scale and actual recorded physics positions, rather
than joining planet centers. Its cyan trace contains up to 20,000 recent mechanical
positions from the current session. Old encounter records remain as world markers;
no historical fly path is invented. Restart begins a fresh trace. Checkpoints,
seeds, and organism state formats are unchanged.

Verification: the regression checks that b3-only firing creates no pitch torque,
that b1/b2 retain their powered pitch contribution, that inactive wings supply
no aerodynamic torque, and that surface leg pitch remains available. A replay
of captured motor rates checks the removed pitch component and preserves force,
roll and yaw commands. Existing saved organisms are not modified.
