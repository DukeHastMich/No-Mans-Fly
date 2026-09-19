# Fidelity-preserving performance pass

## Measured result

Windows, RTX 4060 Ti, two CPU neural workers, original 166,700-neuron seed,
GPU neural backend, learning enabled, twelve slices per 100-ms window,
world-pack spectral assets, intermediate snapshots enabled, no GUI painting.
Each fresh temporary instance ran three windows; the first was warm-up.
No existing live organism was advanced or saved.

| Configuration | Warm window 1 | Warm window 2 | Mean |
| --- | ---: | ---: | ---: |
| Reference | 8.242 s | 7.794 s | 8.018 s |
| Weight reuse + scalar rotations | 6.721 s | 6.702 s | 6.711 s |
| Above + tiled optics | 5.039 s | 5.321 s | 5.180 s |

The combined result is approximately 35.4% less wall time, or 1.55x throughput.
These short sequential local measurements can be influenced by concurrent load.
They are not a speed guarantee, an FPS measurement, or real-time performance.
A one-thread BLAS experiment produced only a small additional difference in the
intermediate configuration; no global BLAS settings were changed.

## Changes

1. Rebuild the 25.6-million-edge delivery-scale array once per explicit learning
   window, not twelve times. Standalone epochs rebuild normally. Learning reset
   and memory loading invalidate the window cache. Each new window recomputes
   the scale after the preceding consolidation.
2. Upload the unchanged scale and regenerate GPU delivery weights only when the
   CPU scale token changes. The exact accumulation bound is checked for each new
   scale. GPU delivery weights are shared across the 83/84-tick configurations.
   Failed work still falls back to uncommitted CPU state.
3. Evaluate point-source optical acceptance in 64-row tiles, retaining every ray,
   source, optical term, and the original matrix multiplication. Smaller temporary
   arrays reduce memory traffic without reducing optical resolution.
4. Spell out the existing two cross products for single-vector quaternion
   rotations. Batch rotation behavior is unchanged. Arithmetic order is preserved.

No time step, neuron, connection, plasticity rule, retinal sample, or physics
slice was removed. No prediction, skipped neural work, or lower precision was
introduced. Runtime caches do not alter the checkpoint format.

## Validation

Three full-CNS windows produced identical reference/optimized hashes for neural
membrane/conductance/refractory and modulation arrays, learned deltas, familiarity,
last rates, delayed spikes, random-generator state, state-wave state, eye temporal
history, body pose/velocity, world state, pending motor commands and homeostasis.
Synthetic tests separately check multi-window CPU/GPU equality, 83/84-tick cache
switching, reset invalidation and late GPU failure before commit. 10,000 random
scalar rotations and inverse rotations match exactly. Optical acceptance matches
bit-for-bit across tile boundaries, including a 1,772-by-530 ray/source case.

These checks cover tested inputs and this runtime; they are not a proof for every
platform or all future changes. Preserve cache invalidation if future work permits
structural or learned-weight mutation inside a learning window.

## Remaining opportunities

The baseline profile attributed roughly 40% of wall time to vision, 27% to neural
execution, 15% to learning and 12% to snapshot construction. These categories
change after optimization and should be re-profiled before another pass.

Promising next experiments are fused or GPU optical evaluation with matching
rounding, batched plasticity updates preserving reduction/float16 semantics,
and lighter observer snapshots. Keeping more neural state resident on the GPU
could reduce transfers, but needs the same checkpoint and failure-recovery
contract. None of those unimplemented changes is included in the figures above.
