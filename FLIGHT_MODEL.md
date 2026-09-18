# Reduced flight mechanics and actual atlas traces

This patch restores a missing motor-to-mechanics path; it does not establish that
the resulting trajectories are biologically validated fruit-fly flight.

Bilateral b1/b2 recruitment advances a reduced wing-hinge stroke command;
b3 recruitment opposes it. Each side uses 0.5*(b1+b2)-b3, bounded to [-1,1].
Its contribution is multiplied by that side's power-muscle command. The mean
produces body-y torque with negative sign for nose-up (x forward, z dorsal).
The coefficient 0.018 is an engineering calibration in the existing actuator
units. Antagonist balance is neutral. Increasing or decreasing that balance
permits pitch in either direction, without a target heading or gravity autopilot.
The rate-to-recruitment conversion does not recover actual wingbeat spike phase.

Existing muscle-specific yaw torque is now multiplied by mean wing power, so
steering activity cannot supply aerodynamic yaw while the wings are unpowered.
The existing thrust and drag envelopes are retained pending aerodynamic
calibration. This patch does not claim to solve every cause of sustained flight,
missing sensory feedback, or learned behavior. Unknown muscles remain unresolved.

Research basis: Whitehead et al., Neuromuscular embodiment of feedback control
elements in Drosophila flight (2022), establishes b1/b2 contributions to pitch:
https://cohengroup.lassp.cornell.edu/userfiles/pubs/Whitehead_SciAdv_2022.pdf
The simplified antagonist mapping and numerical gains above are model assumptions,
not coefficients measured in that paper.

The atlas now uses equal X/Y scale and actual recorded physics positions, rather
than joining planet centers. Its cyan trace contains up to 20,000 recent mechanical
positions from the current session. Old encounter records remain as world markers;
no historical fly path is invented. Restart begins a fresh trace. Checkpoints,
seeds, and organism state formats are unchanged.

Verification: synthetic motor tests check zero-input forces, unpowered steering,
positive/negative pitch, balanced antagonists, bilateral symmetry and quaternion
sign. A temporary full-CNS GPU run checks twelve physics slices, save/resume,
actual atlas trace creation and hidden UI rendering. These tests do not prove
long-term stability of an existing learned organism.
