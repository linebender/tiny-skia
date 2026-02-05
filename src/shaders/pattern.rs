// Copyright 2006 The Android Open Source Project
// Copyright 2020 Yevhenii Reizner
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use tiny_skia_path::NormalizedF32;

use crate::{BlendMode, ColorSpace, PixmapRef, Shader, SpreadMode, Transform};

use crate::mipmap::compute_required_levels;
use crate::pipeline;
use crate::pipeline::RasterPipelineBuilder;

#[cfg(all(not(feature = "std"), feature = "no-std-float"))]
use tiny_skia_path::NoStdFloat;

/// Controls how much filtering to be done when transforming images.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum FilterQuality {
    /// Nearest-neighbor. Low quality, but fastest.
    Nearest,
    /// Bilinear.
    Bilinear,
    /// Bicubic. High quality, but slow.
    Bicubic,
}

/// Controls how a pixmap should be blended.
///
/// Like `Paint`, but for `Pixmap`.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct PixmapPaint {
    /// Pixmap opacity.
    ///
    /// Must be in 0..=1 range.
    ///
    /// Default: 1.0
    pub opacity: f32,

    /// Pixmap blending mode.
    ///
    /// Default: SourceOver
    pub blend_mode: BlendMode,

    /// Specifies how much filtering to be done when transforming images.
    ///
    /// Default: Nearest
    pub quality: FilterQuality,
}

impl Default for PixmapPaint {
    fn default() -> Self {
        PixmapPaint {
            opacity: 1.0,
            blend_mode: BlendMode::default(),
            quality: FilterQuality::Nearest,
        }
    }
}

/// A pattern shader.
///
/// Essentially a `SkImageShader`.
///
/// Unlike Skia, we do not support FilterQuality::Medium, because it involves
/// mipmap generation, which adds too much complexity.
#[derive(Clone, PartialEq, Debug)]
pub struct Pattern<'a> {
    pub(crate) pixmap: PixmapRef<'a>,
    quality: FilterQuality,
    spread_mode: SpreadMode,
    pub(crate) opacity: NormalizedF32,
    pub(crate) transform: Transform,
}

impl<'a> Pattern<'a> {
    /// Creates a new pattern shader.
    ///
    /// `opacity` will be clamped to the 0..=1 range.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        pixmap: PixmapRef<'a>,
        spread_mode: SpreadMode,
        quality: FilterQuality,
        opacity: f32,
        transform: Transform,
    ) -> Shader<'a> {
        Shader::Pattern(Pattern {
            pixmap,
            spread_mode,
            quality,
            opacity: NormalizedF32::new_clamped(opacity),
            transform,
        })
    }

    pub(crate) fn push_stages(&self, cs: ColorSpace, p: &mut RasterPipelineBuilder) -> bool {
        let mut transform = self.transform;
        let mut quality = self.quality;

        // Minimizing scale via mipmap
        let mut pixmap_width = self.pixmap.width() as f32;
        let mut pixmap_height = self.pixmap.height() as f32;
        let (scale_x, scale_y) = transform.get_scale();
        if scale_x < 0.5 && scale_y < 0.5 && quality != FilterQuality::Nearest {
            let (levels, prescale_x, prescale_y) =
                compute_required_levels(self.pixmap, scale_x, scale_y);
            p.set_required_mipmap_levels(levels);
            transform = transform.pre_scale(prescale_x, prescale_y);
            pixmap_width /= prescale_x;
            pixmap_height /= prescale_y;
        }

        let ts = match transform.invert() {
            Some(v) => v,
            None => {
                log::warn!("failed to invert a pattern transform. Nothing will be rendered");
                return false;
            }
        };

        p.push(pipeline::Stage::SeedShader);

        p.push_transform(ts);

        if ts.is_identity() || ts.is_translate() {
            quality = FilterQuality::Nearest;
        }

        if quality == FilterQuality::Bilinear {
            if ts.is_translate() {
                if ts.tx == ts.tx.trunc() && ts.ty == ts.ty.trunc() {
                    // When the matrix is just an integer translate, bilerp == nearest neighbor.
                    quality = FilterQuality::Nearest;
                }
            }
        }

        match quality {
            FilterQuality::Nearest => {
                p.ctx.limit_x = pipeline::TileCtx {
                    scale: pixmap_width,
                    inv_scale: 1.0 / pixmap_width,
                };

                p.ctx.limit_y = pipeline::TileCtx {
                    scale: pixmap_height,
                    inv_scale: 1.0 / pixmap_height,
                };

                match self.spread_mode {
                    SpreadMode::Pad => { /* The gather() stage will clamp for us. */ }
                    SpreadMode::Repeat => p.push(pipeline::Stage::Repeat),
                    SpreadMode::Reflect => p.push(pipeline::Stage::Reflect),
                }

                p.push(pipeline::Stage::Gather);
            }
            FilterQuality::Bilinear => {
                p.ctx.sampler = pipeline::SamplerCtx {
                    spread_mode: self.spread_mode,
                    inv_width: 1.0 / pixmap_width,
                    inv_height: 1.0 / pixmap_height,
                };
                p.push(pipeline::Stage::Bilinear);
            }
            FilterQuality::Bicubic => {
                p.ctx.sampler = pipeline::SamplerCtx {
                    spread_mode: self.spread_mode,
                    inv_width: 1.0 / pixmap_width,
                    inv_height: 1.0 / pixmap_height,
                };
                p.push(pipeline::Stage::Bicubic);

                // Bicubic filtering naturally produces out of range values on both sides of [0,1].
                p.push(pipeline::Stage::Clamp0);
                p.push(pipeline::Stage::ClampA);
            }
        }

        // Unlike Skia, we do not support global opacity and only Pattern allows it.
        if self.opacity != NormalizedF32::ONE {
            debug_assert_eq!(
                core::mem::size_of_val(&self.opacity),
                4,
                "alpha must be f32"
            );
            p.ctx.current_coverage = self.opacity.get();
            p.push(pipeline::Stage::Scale1Float);
        }

        if let Some(stage) = cs.expand_stage() {
            // TODO: it would be better to gamma-expand prior to sampling, but
            // there isn't currently an intermediate pipeline stage for that.
            p.push(stage);
        }

        true
    }
}
