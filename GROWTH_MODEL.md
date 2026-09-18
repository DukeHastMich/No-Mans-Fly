# Growth model: defects repaired, evolutionary trials staged

The desktop no longer automatically duplicates highly active neurons. Its
Growth trials armed control records proposals in the live lineage's GrowthTrials
folder. A proposal does not change the live topology, neuron count, or weights.
Ordinary synaptic learning remains active when Learning is enabled.

## Repairs

* Only positive plastic saturation counts toward capacity pressure; depression
  no longer masquerades as a request for more excitation.
* Persistently active disconnected sources can request connection recruitment
  without needing nonexistent outgoing edges to saturate.
* The explicit full-graph duplication utility retains fractional structural
  strength even when measured synapse counts round to one. New edges require
  donor transfers; their strength is deducted from existing structural gains.
* Budgeted graphs carry an explicit schema marker. Legacy gain repair preserves
  these intentional fractional gains instead of resetting them to unity.
* Old pressure evidence is invalidated on load by the new semantics version.

The structural budget constrains nominal synaptic strength, not firing rates,
learned effective strength, or behavior. A topology change can still alter all
three and must be tested. Old unbudgeted overlays cannot be compiled through
the revised full-graph compiler. Existing live graphs are not retroactively
rewired, and the immutable seed is untouched.

## Direction toward evolutionary growth

Proposals use same-type, same-side, same-entry-nerve, same-transmitter peers to
identify compatible existing destinations. Missing annotation blocks a proposal.
This is annotated circuit locality, not measured geometric proximity. Direct
motor and sensory targets are excluded. Each proposed connection is capped at
5% of a donor's structural strength and 0.05 mV per event, whichever is smaller.
These are experimental engineering limits, not measured biological constants.

The evaluation helper compares matched repeated stimulus responses and quiet
responses. It requires improved normalized discrimination, limits saturation,
and rejects elevated quiet firing. Uniform response amplification alone fails.
Passing means eligible for extended trials, never automatic live promotion.

Still required before autonomous evolutionary growth: an isolated matched
control/mutant replay runner; held-out input scheduling; actual budgeted proposal
application and rollback; stability and retention tests across longer episodes;
selection and pruning of unsuccessful mutations; and evidence that extra neurons
add capability beyond new connections. The current helper cannot establish
behavioral utility or biological accuracy. Repeated weak stimulation is a
candidate mechanism to test, not a guarantee of meaningful learning.

## Validation

Focused regressions cover positive-only pressure, disconnected recruitment,
fractional weak edges, donor budgets, overlay persistence, invalid transfer
rejection, proposal compatibility, uniform-gain rejection and quiet-state gates.
An isolated 166,700-neuron integration run verified unchanged live topology on
proposal, persisted proposal records, cooldown, twelve GPU/physics slices,
checkpoint save/resume, and hidden UI/HUD rendering. Existing live organisms
were not advanced or saved by these tests.
