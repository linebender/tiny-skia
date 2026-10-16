# Sweep gradient normalization reproducer

Based on upstream `main` at `5d4754777746eef0828be166896eaf482c49f8f2`
(tiny-skia 0.12.0). This reproducer uses only tiny-skia: no fonts, Nova,
Skrifa, GPU backend, or image comparison library.

Run the example and regression tests:

```sh
cargo run --example sweep_gradient_normalization -- target/sweep-gradient-after.png
cargo test --test sweep_gradient_normalization
```

The example fills a 256×256 rectangle with an opaque red-to-blue sweep,
centered at `(128.5, 128.5)`, from 0° to 180°, using `SpreadMode::Pad` and
identity transforms. Screen angles increase clockwise from the right.
The half-pixel center makes the sampled pixels lie exactly on cardinal rays.

| Sample | Expected RGBA | Unpatched upstream | Patched |
| --- | --- | --- | --- |
| 90° midpoint, `(128, 224)` | approximately `[128, 0, 128, 255]` | `[191, 0, 64, 255]` | `[128, 0, 128, 255]` |
| 180° endpoint, `(32, 128)` | `[0, 0, 255, 255]` | `[128, 0, 128, 255]` | `[0, 0, 255, 255]` |
| 270° padded end, `(128, 32)` | `[0, 0, 255, 255]` | `[64, 0, 191, 255]` | `[0, 0, 255, 255]` |

The original implementation spreads the transition over a whole turn rather
than completing it at 180°. The top half should remain the final blue color.

`SweepGradient::push_stages` computes `scale = 2` and `bias = 0` for this
case. Its `scale != 1.0 && bias != 0.0` condition therefore skips the
normalization stage. Either term can require normalization independently;
the patch changes `&&` to `||`.

The regression tests cover both independent cases with default pipeline
selection and forced high-quality rendering:

- 0°–180°: nonidentity scale, zero bias.
- 90°–450°: identity scale, nonzero bias.

Both tests failed against the original source and pass with the one-line fix.
Existing sweep fixtures cover 135°–225° (both terms nonidentity) and 0°–360°
(both terms identity), so they do not exercise these independent cases.

## Independent reference

The same 256×256 example was rendered with native Skia using
`skia-python==144.0.post2`, `GradientShader.MakeSweep`, clamp tiling, identical
colors/positions/angles, and antialiasing/dithering disabled. The patched
tiny-skia PNG matches all 65,536 reference pixels exactly. The unpatched PNG
differs by more than two channel values at 64,533 pixels; its maximum channel
error is 128. Native Skia also produces the asserted cardinal colors for
the 90°–450° case.

This matches [Skia's documented angular range and tiling semantics](https://api.skia.org/classSkGradientShader.html):
the start/end angles correspond to gradient positions zero/one, and the tiling
mode controls the region outside that range. Its
[sweep implementation](https://github.com/google/skia/blob/main/src/shaders/gradients/SkSweepGradient.cpp)
applies both normalization terms after converting coordinates to a unit angle.
The stage is only redundant when **both** scale is one and bias is zero.

Before/after images and their labeled comparison are local outputs under
`target/sweep-gradient-{before,after,comparison}.png`; they are not source assets.
