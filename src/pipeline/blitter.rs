// Copyright 2016 Google Inc.
// Copyright 2020 Yevhenii Reizner
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use crate::{
    BlendMode, Color, DynamicPixmapRef, LengthU32, Paint, Pixel, PremultipliedColorU8, Shader,
};
use crate::{ALPHA_U8_OPAQUE, ALPHA_U8_TRANSPARENT};

use crate::alpha_runs::AlphaRun;
use crate::blitter::{Blitter, Mask};
use crate::color::AlphaU8;
use crate::geom::ScreenIntRect;
use crate::mask::SubMaskRef;
use crate::math::LENGTH_U32_ONE;
use crate::pipeline::highp::HighPixel;
use crate::pipeline::{self, RasterPipelineBuilder, RasterPipelineGeneric};
use crate::pixmap::SubPixmapMutGeneric;

pub struct RasterPipelineBlitter<'a, 'b: 'a, P: HighPixel> {
    mask: Option<SubMaskRef<'a>>,
    pixmap_src: DynamicPixmapRef<'a>,
    pixmap: &'a mut SubPixmapMutGeneric<'b, P>,
    memset2d_color: Option<P>,
    blit_anti_h_rp: RasterPipelineGeneric<P>,
    blit_rect_rp: RasterPipelineGeneric<P>,
    blit_mask_rp: RasterPipelineGeneric<P>,
    is_mask: bool,
}

impl<'a, 'b: 'a, P: HighPixel> RasterPipelineBlitter<'a, 'b, P> {
    pub fn new(
        paint: &Paint<'a>,
        mask: Option<SubMaskRef<'a>>,
        pixmap: &'a mut SubPixmapMutGeneric<'b, P>,
    ) -> Option<Self> {
        // Make sure that `mask` has the same size as `pixmap`.
        if let Some(mask) = mask {
            if mask.size.width() != pixmap.size.width()
                || mask.size.height() != pixmap.size.height()
            {
                log::warn!("Pixmap and Mask are expected to have the same size");
                return None;
            }
        }

        // Fast-reject.
        // This is basically SkInterpretXfermode().
        match paint.blend_mode {
            // `Destination` keep the pixmap unchanged. Nothing to do here.
            BlendMode::Destination => return None,
            BlendMode::DestinationIn if paint.shader.is_opaque() && paint.is_solid_color() => {
                return None
            }
            _ => {}
        }

        // We can strength-reduce SourceOver into Source when opaque.
        let mut blend_mode = paint.blend_mode;
        if paint.shader.is_opaque() && blend_mode == BlendMode::SourceOver && mask.is_none() {
            blend_mode = BlendMode::Source;
        }

        // When we're drawing a constant color in Source mode, we can sometimes just memset.
        let mut memset2d_color = None;
        if paint.is_solid_color() && blend_mode == BlendMode::Source && mask.is_none() {
            // Unlike Skia, our shader cannot be constant.
            // Therefore there is no need to run a raster pipeline to get shader's color.
            if let Shader::SolidColor(ref color) = paint.shader {
                memset2d_color = Some(P::from_color(*color));
            }
        };

        // Clear is just a transparent color memset.
        if blend_mode == BlendMode::Clear && !paint.anti_alias && mask.is_none() {
            blend_mode = BlendMode::Source;
            memset2d_color = Some(P::from_color(Color::TRANSPARENT));
        }

        let blit_anti_h_rp = {
            let mut p = RasterPipelineBuilder::new();
            p.set_force_hq_pipeline(paint.force_hq_pipeline);
            if !paint.shader.push_stages(paint.colorspace, &mut p) {
                return None;
            }

            if mask.is_some() {
                p.push(pipeline::Stage::MaskU8);
            }

            if blend_mode.should_pre_scale_coverage() {
                p.push(pipeline::Stage::Scale1Float);
                p.push(pipeline::Stage::LoadDestination);
                if let Some(stage) = paint.colorspace.expand_dest_stage() {
                    p.push(stage);
                }
                if let Some(blend_stage) = blend_mode.to_stage() {
                    p.push(blend_stage);
                }
            } else {
                p.push(pipeline::Stage::LoadDestination);
                if let Some(stage) = paint.colorspace.expand_dest_stage() {
                    p.push(stage);
                }
                if let Some(blend_stage) = blend_mode.to_stage() {
                    p.push(blend_stage);
                }

                p.push(pipeline::Stage::Lerp1Float);
            }

            if let Some(stage) = paint.colorspace.compress_stage() {
                p.push(stage);
            }
            p.push(pipeline::Stage::Store);

            p.compile()
        };

        let blit_rect_rp = {
            let mut p = RasterPipelineBuilder::new();
            p.set_force_hq_pipeline(paint.force_hq_pipeline);
            if !paint.shader.push_stages(paint.colorspace, &mut p) {
                return None;
            }

            if mask.is_some() {
                p.push(pipeline::Stage::MaskU8);
            }

            if blend_mode == BlendMode::SourceOver && mask.is_none() {
                if let Some(stage) = paint.colorspace.compress_stage() {
                    p.push(stage);
                }
                // TODO: ignore when dither_rate is non-zero
                p.push(pipeline::Stage::SourceOverRgba);
            } else {
                if blend_mode != BlendMode::Source {
                    p.push(pipeline::Stage::LoadDestination);
                    if let Some(blend_stage) = blend_mode.to_stage() {
                        if let Some(stage) = paint.colorspace.expand_dest_stage() {
                            p.push(stage);
                        }
                        p.push(blend_stage);
                    }
                }

                if let Some(stage) = paint.colorspace.compress_stage() {
                    p.push(stage);
                }
                p.push(pipeline::Stage::Store);
            }

            p.compile()
        };

        let blit_mask_rp = {
            let mut p = RasterPipelineBuilder::new();
            p.set_force_hq_pipeline(paint.force_hq_pipeline);
            if !paint.shader.push_stages(paint.colorspace, &mut p) {
                return None;
            }

            if mask.is_some() {
                p.push(pipeline::Stage::MaskU8);
            }

            if blend_mode.should_pre_scale_coverage() {
                p.push(pipeline::Stage::ScaleU8);
                p.push(pipeline::Stage::LoadDestination);
                if let Some(stage) = paint.colorspace.expand_dest_stage() {
                    p.push(stage);
                }
                if let Some(blend_stage) = blend_mode.to_stage() {
                    p.push(blend_stage);
                }
            } else {
                p.push(pipeline::Stage::LoadDestination);
                if let Some(stage) = paint.colorspace.expand_dest_stage() {
                    p.push(stage);
                }
                if let Some(blend_stage) = blend_mode.to_stage() {
                    p.push(blend_stage);
                }

                p.push(pipeline::Stage::LerpU8);
            }

            if let Some(stage) = paint.colorspace.compress_stage() {
                p.push(stage);
            }
            p.push(pipeline::Stage::Store);

            p.compile()
        };

        let pixmap_src = match paint.shader {
            Shader::Pattern(ref patt) => patt.pixmap,
            // Just a dummy one.
            _ => DynamicPixmapRef::dummy(),
        };

        Some(RasterPipelineBlitter {
            mask,
            pixmap_src,
            pixmap,
            memset2d_color,
            blit_anti_h_rp,
            blit_rect_rp,
            blit_mask_rp,
            is_mask: false,
        })
    }
}

impl<'a, 'b: 'a> RasterPipelineBlitter<'a, 'b, PremultipliedColorU8> {
    pub fn new_mask(pixmap: &'a mut SubPixmapMutGeneric<'b, PremultipliedColorU8>) -> Option<Self> {
        let color = Color::WHITE.premultiply();

        let memset2d_color = Some(PremultipliedColorU8::from_color(Color::WHITE));

        let blit_anti_h_rp = {
            let mut p = RasterPipelineBuilder::new();
            p.push_uniform_color(color);
            p.push(pipeline::Stage::LoadDestinationU8);
            p.push(pipeline::Stage::Lerp1Float);
            p.push(pipeline::Stage::StoreU8);
            p.compile()
        };

        let blit_rect_rp = {
            let mut p = RasterPipelineBuilder::new();
            p.push_uniform_color(color);
            p.push(pipeline::Stage::StoreU8);
            p.compile()
        };

        let blit_mask_rp = {
            let mut p = RasterPipelineBuilder::new();
            p.push_uniform_color(color);
            p.push(pipeline::Stage::LoadDestinationU8);
            p.push(pipeline::Stage::LerpU8);
            p.push(pipeline::Stage::StoreU8);
            p.compile()
        };

        Some(RasterPipelineBlitter {
            mask: None,
            pixmap_src: DynamicPixmapRef::dummy(),
            pixmap,
            memset2d_color,
            blit_anti_h_rp,
            blit_rect_rp,
            blit_mask_rp,
            is_mask: true,
        })
    }
}

impl<P: HighPixel> Blitter for RasterPipelineBlitter<'_, '_, P> {
    fn blit_h(&mut self, x: u32, y: u32, width: LengthU32) {
        let r = ScreenIntRect::from_xywh_safe(x, y, width, LENGTH_U32_ONE);
        self.blit_rect(&r);
    }

    fn blit_anti_h(&mut self, mut x: u32, y: u32, aa: &mut [AlphaU8], runs: &mut [AlphaRun]) {
        let mask_ctx = self.mask.map(|c| c.mask_ctx()).unwrap_or_default();

        let mut aa_offset = 0;
        let mut run_offset = 0;
        let mut run_opt = runs[0];
        while let Some(run) = run_opt {
            let width = LengthU32::from(run);

            match aa[aa_offset] {
                ALPHA_U8_TRANSPARENT => {}
                ALPHA_U8_OPAQUE => {
                    self.blit_h(x, y, width);
                }
                alpha => {
                    self.blit_anti_h_rp.ctx.current_coverage = alpha as f32 * (1.0 / 255.0);

                    let rect = ScreenIntRect::from_xywh_safe(x, y, width, LENGTH_U32_ONE);
                    self.blit_anti_h_rp.run(
                        &rect,
                        pipeline::AAMaskCtx::default(),
                        mask_ctx,
                        self.pixmap_src,
                        self.pixmap,
                    );
                }
            }

            x += width.get();
            run_offset += usize::from(run.get());
            aa_offset += usize::from(run.get());
            run_opt = runs[run_offset];
        }
    }

    fn blit_v(&mut self, x: u32, y: u32, height: LengthU32, alpha: AlphaU8) {
        let bounds = ScreenIntRect::from_xywh_safe(x, y, LENGTH_U32_ONE, height);

        let mask = Mask {
            image: [alpha, alpha],
            bounds,
            row_bytes: 0, // so we reuse the 1 "row" for all of height
        };

        self.blit_mask(&mask, &bounds);
    }

    fn blit_anti_h2(&mut self, x: u32, y: u32, alpha0: AlphaU8, alpha1: AlphaU8) {
        let bounds = ScreenIntRect::from_xywh(x, y, 2, 1).unwrap();

        let mask = Mask {
            image: [alpha0, alpha1],
            bounds,
            row_bytes: 2,
        };

        self.blit_mask(&mask, &bounds);
    }

    fn blit_anti_v2(&mut self, x: u32, y: u32, alpha0: AlphaU8, alpha1: AlphaU8) {
        let bounds = ScreenIntRect::from_xywh(x, y, 1, 2).unwrap();

        let mask = Mask {
            image: [alpha0, alpha1],
            bounds,
            row_bytes: 1,
        };

        self.blit_mask(&mask, &bounds);
    }

    fn blit_rect(&mut self, rect: &ScreenIntRect) {
        if let Some(c) = self.memset2d_color {
            let real_width = self.pixmap.real_width;
            if self.is_mask {
                let mask_data: &mut [u8] = bytemuck::cast_slice_mut(self.pixmap.data);
                for y in 0..rect.height() {
                    let start = real_width * (rect.y() + y) as usize + rect.x() as usize;
                    let end = start + rect.width() as usize;
                    mask_data[start..end].fill(255);
                }
            } else {
                for y in 0..rect.height() {
                    let start = (real_width * (rect.y() + y) as usize + rect.x() as usize)
                        * P::BYTES_PER_PIXEL;
                    let end = start + rect.width() as usize * P::BYTES_PER_PIXEL;
                    let slice: &mut [P] =
                        bytemuck::cast_slice_mut(&mut self.pixmap.data[start..end]);
                    slice.fill(c);
                }
            }

            return;
        }

        let mask_ctx = self.mask.map(|c| c.mask_ctx()).unwrap_or_default();

        self.blit_rect_rp.run(
            rect,
            pipeline::AAMaskCtx::default(),
            mask_ctx,
            self.pixmap_src,
            self.pixmap,
        );
    }

    fn blit_mask(&mut self, mask: &Mask, clip: &ScreenIntRect) {
        let aa_mask_ctx = pipeline::AAMaskCtx {
            pixels: mask.image,
            stride: mask.row_bytes,
            shift: (mask.bounds.left() + mask.bounds.top() * mask.row_bytes) as usize,
        };

        let mask_ctx = self.mask.map(|c| c.mask_ctx()).unwrap_or_default();

        self.blit_mask_rp
            .run(clip, aa_mask_ctx, mask_ctx, self.pixmap_src, self.pixmap);
    }
}
