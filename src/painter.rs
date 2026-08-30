// Copyright 2006 The Android Open Source Project
// Copyright 2020 Yevhenii Reizner
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use crate::*;

use tiny_skia_path::{PathStroker, Scalar, SCALAR_MAX};

use crate::geom::ScreenIntRect;
use crate::mask::SubMaskRef;
use crate::pipeline::{HighPixel, RasterPipelineBlitter, RasterPipelineBuilder};
use crate::pixmap::SubPixmapMutGeneric;
use crate::scan;

use crate::geom::IntSizeExt;
#[cfg(all(not(feature = "std"), feature = "no-std-float"))]
use tiny_skia_path::NoStdFloat;

/// A path filling rule.
#[derive(Copy, Clone, Default, PartialEq, Debug)]
pub enum FillRule {
    /// Specifies that "inside" is computed by a non-zero sum of signed edge crossings.
    #[default]
    Winding,
    /// Specifies that "inside" is computed by an odd number of edge crossings.
    EvenOdd,
}

/// Controls how a shape should be painted.
#[derive(Clone, PartialEq, Debug)]
pub struct Paint<'a> {
    /// A paint shader.
    ///
    /// Default: black color
    pub shader: Shader<'a>,

    /// Paint blending mode.
    ///
    /// Default: SourceOver
    pub blend_mode: BlendMode,

    /// Enables anti-aliased painting.
    ///
    /// Default: true
    pub anti_alias: bool,

    /// Colorspace for blending.
    ///
    /// This enables gamma correction during the blend operation.  While skia supports
    /// full color-space conversions, we only support a few (simple) cases.  Note that
    /// any color space other than Linear will force using the high-quality pipeline.
    ///
    /// Default: Linear
    pub colorspace: ColorSpace,

    /// Forces the high quality/precision rendering pipeline.
    ///
    /// `tiny-skia`, just like Skia, has two rendering pipelines:
    /// one uses `f32` and another one uses `u16`. `u16` one is usually way faster,
    /// but less precise. Which can lead to slight differences.
    ///
    /// By default, `tiny-skia` will choose the pipeline automatically,
    /// depending on a blending mode and other parameters.
    /// But you can force the high quality one using this flag.
    ///
    /// This feature is especially useful during testing.
    ///
    /// Unlike high quality pipeline, the low quality one doesn't support all
    /// rendering stages, therefore we cannot force it like hq one.
    ///
    /// Default: false
    pub force_hq_pipeline: bool,
}

impl Default for Paint<'_> {
    fn default() -> Self {
        Paint {
            shader: Shader::SolidColor(Color::BLACK),
            blend_mode: BlendMode::default(),
            anti_alias: true,
            colorspace: ColorSpace::default(),
            force_hq_pipeline: false,
        }
    }
}

impl Paint<'_> {
    /// Sets a paint source to a solid color.
    pub fn set_color(&mut self, color: Color) {
        self.shader = Shader::SolidColor(color);
    }

    /// Sets a paint source to a solid color.
    ///
    /// `self.shader = Shader::SolidColor(Color::from_rgba8(50, 127, 150, 200));` shorthand.
    pub fn set_color_rgba8(&mut self, r: u8, g: u8, b: u8, a: u8) {
        self.set_color(Color::from_rgba8(r, g, b, a))
    }

    /// Checks that the paint source is a solid color.
    pub fn is_solid_color(&self) -> bool {
        matches!(self.shader, Shader::SolidColor(_))
    }
}

impl<P: HighPixel> PixmapGeneric<P> {
    /// Draws a filled rectangle onto the pixmap.
    ///
    /// See [`PixmapMut::fill_rect`](struct.PixmapMut.html#method.fill_rect) for details.
    pub fn fill_rect(
        &mut self,
        rect: Rect,
        paint: &Paint,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        self.as_mut().fill_rect(rect, paint, transform, mask);
    }

    /// Draws a filled path onto the pixmap.
    ///
    /// See [`PixmapMut::fill_path`](struct.PixmapMut.html#method.fill_path) for details.
    pub fn fill_path(
        &mut self,
        path: &Path,
        paint: &Paint,
        fill_rule: FillRule,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        self.as_mut()
            .fill_path(path, paint, fill_rule, transform, mask);
    }

    /// Draws a stroked path onto the pixmap.
    ///
    /// See [`PixmapMut::stroke_path`](struct.PixmapMut.html#method.stroke_path) for details.
    pub fn stroke_path(
        &mut self,
        path: &Path,
        paint: &Paint,
        stroke: &Stroke,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        self.as_mut()
            .stroke_path(path, paint, stroke, transform, mask);
    }

    /// Draws a pixmap onto the pixmap.
    ///
    /// See [`PixmapMut::draw_pixmap`](struct.PixmapMut.html#method.draw_pixmap) for details.
    pub fn draw_pixmap<'b>(
        &mut self,
        x: i32,
        y: i32,
        pixmap: PixmapRefGeneric<'b, P>,
        paint: &PixmapPaint,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        self.as_mut()
            .draw_pixmap(x, y, pixmap, paint, transform, mask);
    }

    /// Applies a mask.
    ///
    /// See [`PixmapMut::apply_mask`](struct.PixmapMut.html#method.apply_mask) for details.
    pub fn apply_mask(&mut self, mask: &Mask) {
        self.as_mut().apply_mask(mask);
    }
}

impl<'a, P: HighPixel> PixmapMutGeneric<'a, P> {
    // TODO: accept NonZeroRect?
    /// Draws a filled rectangle onto the pixmap.
    ///
    /// This function is usually slower than filling a rectangular path,
    /// but it produces better results. Mainly it doesn't suffer from weird
    /// clipping of horizontal/vertical edges.
    ///
    /// Used mainly to render a pixmap onto a pixmap.
    pub fn fill_rect(
        &mut self,
        rect: Rect,
        paint: &Paint,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        // TODO: we probably can use tiler for rect too
        if transform.is_identity() && !DrawTiler::required(self.width(), self.height()) {
            // TODO: ignore rects outside the pixmap

            let clip = self.size().to_screen_int_rect(0, 0);

            let mask = mask.map(|mask| mask.as_submask());
            let mut subpix = self.as_subpixmap();
            let mut blitter = match RasterPipelineBlitter::new(paint, mask, &mut subpix) {
                Some(v) => v,
                None => return, // nothing to do, all good
            };

            if paint.anti_alias {
                scan::fill_rect_aa(&rect, &clip, &mut blitter);
            } else {
                scan::fill_rect(&rect, &clip, &mut blitter);
            }
        } else {
            let path = PathBuilder::from_rect(rect);
            self.fill_path(&path, paint, FillRule::Winding, transform, mask);
        }
    }

    /// Draws a filled path onto the pixmap.
    pub fn fill_path(
        &mut self,
        path: &Path,
        paint: &Paint,
        fill_rule: FillRule,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        if transform.is_identity() {
            // This is sort of similar to SkDraw::drawPath

            // Skip empty paths and horizontal/vertical lines.
            let path_bounds = path.bounds();
            if path_bounds.width().is_nearly_zero() || path_bounds.height().is_nearly_zero() {
                log::warn!("empty paths and horizontal/vertical lines cannot be filled");
                return;
            }

            if is_too_big_for_math(path) {
                log::warn!("path coordinates are too big");
                return;
            }

            // TODO: ignore paths outside the pixmap

            if let Some(tiler) = DrawTiler::new(self.width(), self.height()) {
                let mut path = path.clone(); // TODO: avoid cloning
                let mut paint = paint.clone();

                for tile in tiler {
                    let ts = Transform::from_translate(-(tile.x() as f32), -(tile.y() as f32));
                    path = match path.transform(ts) {
                        Some(v) => v,
                        None => {
                            log::warn!("path transformation failed");
                            return;
                        }
                    };
                    paint.shader.transform(ts);

                    let clip_rect = tile.size().to_screen_int_rect(0, 0);
                    let mut subpix = match self.subpixmap(tile.to_int_rect()) {
                        Some(v) => v,
                        None => continue, // technically unreachable
                    };

                    let submask = mask.and_then(|mask| mask.submask(tile.to_int_rect()));
                    let mut blitter = match RasterPipelineBlitter::new(&paint, submask, &mut subpix)
                    {
                        Some(v) => v,
                        None => continue, // nothing to do, all good
                    };

                    // We're ignoring "errors" here, because `fill_path` will return `None`
                    // when rendering a tile that doesn't have a path on it.
                    // Which is not an error in this case.
                    if paint.anti_alias {
                        scan::path_aa::fill_path(&path, fill_rule, &clip_rect, &mut blitter);
                    } else {
                        scan::path::fill_path(&path, fill_rule, &clip_rect, &mut blitter);
                    }

                    let ts = Transform::from_translate(tile.x() as f32, tile.y() as f32);
                    path = match path.transform(ts) {
                        Some(v) => v,
                        None => return, // technically unreachable
                    };
                    paint.shader.transform(ts);
                }
            } else {
                let clip_rect = self.size().to_screen_int_rect(0, 0);
                let submask = mask.map(|mask| mask.as_submask());
                let mut subpix = self.as_subpixmap();
                let mut blitter = match RasterPipelineBlitter::new(paint, submask, &mut subpix) {
                    Some(v) => v,
                    None => return, // nothing to do, all good
                };

                if paint.anti_alias {
                    scan::path_aa::fill_path(path, fill_rule, &clip_rect, &mut blitter);
                } else {
                    scan::path::fill_path(path, fill_rule, &clip_rect, &mut blitter);
                }
            }
        } else {
            let path = match path.clone().transform(transform) {
                Some(v) => v,
                None => {
                    log::warn!("path transformation failed");
                    return;
                }
            };

            let mut paint = paint.clone();
            paint.shader.transform(transform);

            self.fill_path(&path, &paint, fill_rule, Transform::identity(), mask)
        }
    }

    /// Strokes a path.
    ///
    /// Stroking is implemented using two separate algorithms:
    ///
    /// 1. If a stroke width is wider than 1px (after applying the transformation),
    ///    a path will be converted into a stroked path and then filled using `fill_path`.
    ///    Which means that we have to allocate a separate `Path`, that can be 2-3x larger
    ///    then the original path.
    /// 2. If a stroke width is thinner than 1px (after applying the transformation),
    ///    we will use hairline stroking, which doesn't involve a separate path allocation.
    ///
    /// Also, if a `stroke` has a dash array, then path will be converted into
    /// a dashed path first and then stroked. Which means a yet another allocation.
    pub fn stroke_path(
        &mut self,
        path: &Path,
        paint: &Paint,
        stroke: &Stroke,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        if stroke.width < 0.0 {
            log::warn!("negative stroke width isn't allowed");
            return;
        }

        let res_scale = PathStroker::compute_resolution_scale(&transform);

        let dash_path;
        let path = if let Some(ref dash) = stroke.dash {
            dash_path = match path.dash(dash, res_scale) {
                Some(v) => v,
                None => {
                    log::warn!("path dashing failed");
                    return;
                }
            };
            &dash_path
        } else {
            path
        };

        if let Some(coverage) = treat_as_hairline(paint, stroke, transform) {
            let mut paint = paint.clone();
            if coverage == 1.0 {
                // No changes to the `paint`.
            } else if paint.blend_mode.should_pre_scale_coverage() {
                // This is the old technique, which we preserve for now so
                // we don't change previous results (testing)
                // the new way seems fine, its just (a tiny bit) different.
                let scale = (coverage * 256.0) as i32;
                let new_alpha = (255 * scale) >> 8;
                paint.shader.apply_opacity(new_alpha as f32 / 255.0);
            }

            if let Some(tiler) = DrawTiler::new(self.width(), self.height()) {
                let mut path = path.clone(); // TODO: avoid cloning
                let mut paint = paint.clone();

                if !transform.is_identity() {
                    paint.shader.transform(transform);
                    path = match path.transform(transform) {
                        Some(v) => v,
                        None => {
                            log::warn!("path transformation failed");
                            return;
                        }
                    };
                }

                for tile in tiler {
                    let ts = Transform::from_translate(-(tile.x() as f32), -(tile.y() as f32));
                    path = match path.transform(ts) {
                        Some(v) => v,
                        None => {
                            log::warn!("path transformation failed");
                            return;
                        }
                    };
                    paint.shader.transform(ts);

                    let mut subpix = match self.subpixmap(tile.to_int_rect()) {
                        Some(v) => v,
                        None => continue, // technically unreachable
                    };
                    let submask = mask.and_then(|mask| mask.submask(tile.to_int_rect()));

                    // We're ignoring "errors" here, because `stroke_hairline` will return `None`
                    // when rendering a tile that doesn't have a path on it.
                    // Which is not an error in this case.
                    Self::stroke_hairline(&path, &paint, stroke.line_cap, submask, &mut subpix);

                    let ts = Transform::from_translate(tile.x() as f32, tile.y() as f32);
                    path = match path.transform(ts) {
                        Some(v) => v,
                        None => return,
                    };
                    paint.shader.transform(ts);
                }
            } else {
                let subpix = &mut self.as_subpixmap();
                let submask = mask.map(|mask| mask.as_submask());
                if !transform.is_identity() {
                    paint.shader.transform(transform);

                    // TODO: avoid clone
                    let path = match path.clone().transform(transform) {
                        Some(v) => v,
                        None => {
                            log::warn!("path transformation failed");
                            return;
                        }
                    };

                    Self::stroke_hairline(&path, &paint, stroke.line_cap, submask, subpix);
                } else {
                    Self::stroke_hairline(path, &paint, stroke.line_cap, submask, subpix);
                }
            }
        } else {
            let path = match path.stroke(stroke, res_scale) {
                Some(v) => v,
                None => {
                    log::warn!("path stroking failed");
                    return;
                }
            };

            self.fill_path(&path, paint, FillRule::Winding, transform, mask);
        }
    }

    /// A stroking for paths with subpixel/hairline width.
    fn stroke_hairline(
        path: &Path,
        paint: &Paint,
        line_cap: LineCap,
        mask: Option<SubMaskRef>,
        pixmap: &mut SubPixmapMutGeneric<'_, P>,
    ) {
        let clip = pixmap.size.to_screen_int_rect(0, 0);
        let mut blitter = match RasterPipelineBlitter::new(paint, mask, pixmap) {
            Some(v) => v,
            None => return, // nothing to do, all good
        };
        if paint.anti_alias {
            scan::hairline_aa::stroke_path(path, line_cap, &clip, &mut blitter);
        } else {
            scan::hairline::stroke_path(path, line_cap, &clip, &mut blitter);
        }
    }

    /// Draws a `Pixmap` on top of the current `Pixmap`.
    ///
    /// The same as filling a rectangle with a `pixmap` pattern.
    pub fn draw_pixmap<'b>(
        &mut self,
        x: i32,
        y: i32,
        pixmap: PixmapRefGeneric<'b, P>,
        paint: &PixmapPaint,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        let rect = pixmap.size().to_int_rect(x, y).to_rect();
        let patt_transform = Transform::from_translate(x as f32, y as f32);

        #[cfg(feature = "16bpc")]
        if P::BYTES_PER_PIXEL != 4 {
            if transform.is_identity() && mask.is_none() {
                let dst_w = self.width() as i32;
                let dst_h = self.height() as i32;
                let src_w = pixmap.width() as i32;
                let src_h = pixmap.height() as i32;

                let x0 = x.max(0);
                let y0 = y.max(0);
                let x1 = (x + src_w).min(dst_w);
                let y1 = (y + src_h).min(dst_h);

                if x0 < x1 && y0 < y1 {
                    let opacity = paint.opacity.clamp(0.0, 1.0);
                    let op16 = (opacity * 65535.0 + 0.5) as u32;

                    let src_pixels: &[PremultipliedColorU16] = bytemuck::cast_slice(pixmap.data());
                    let dst_pixels: &mut [PremultipliedColorU16] =
                        bytemuck::cast_slice_mut(self.data_mut());

                    let mode = paint.blend_mode;
                    for cy in y0..y1 {
                        let src_row = (cy - y) * src_w;
                        let dst_row = cy * dst_w;
                        for cx in x0..x1 {
                            let mut sp = src_pixels[(src_row + (cx - x)) as usize];
                            if op16 != 65535 {
                                let r = ((sp.red() as u32 * op16 + 32768) >> 16) as u16;
                                let g = ((sp.green() as u32 * op16 + 32768) >> 16) as u16;
                                let b = ((sp.blue() as u32 * op16 + 32768) >> 16) as u16;
                                let a = ((sp.alpha() as u32 * op16 + 32768) >> 16) as u16;
                                sp = PremultipliedColorU16::from_rgba_unchecked(r, g, b, a);
                            }

                            let dp = &mut dst_pixels[(dst_row + cx) as usize];
                            *dp = blend_u16_op(*dp, sp, mode);
                        }
                    }
                }
                return;
            }
        }

        let paint = Paint {
            shader: Pattern::from_pixmap(
                pixmap,
                SpreadMode::Pad,
                paint.quality,
                paint.opacity,
                patt_transform,
            ),
            blend_mode: paint.blend_mode,
            anti_alias: false,
            force_hq_pipeline: false,
            colorspace: ColorSpace::default(),
        };

        self.fill_rect(rect, &paint, transform, mask);
    }

    /// Applies a masks.
    ///
    /// When a `Mask` is passed to drawing methods, it will be used to mask-out
    /// content we're about to draw.
    /// This method masks-out an already drawn content.
    /// It's not as fast, but can be useful when a mask is not available during drawing.
    ///
    /// This method is similar to filling the whole pixmap with an another,
    /// mask-like pixmap using the `DestinationOut` blend mode.
    ///
    /// `Mask` must have the same size as `Pixmap`. No transform or offset are allowed.
    pub fn apply_mask(&mut self, mask: &Mask) {
        if self.size() != mask.size() {
            log::warn!("Pixmap and Mask are expected to have the same size");
            return;
        }

        // Just a dummy.
        let pixmap_src = DynamicPixmapRef::dummy();

        let mut p = RasterPipelineBuilder::new();
        p.push(pipeline::Stage::LoadMaskU8);
        p.push(pipeline::Stage::LoadDestination);
        p.push(pipeline::Stage::DestinationIn);
        p.push(pipeline::Stage::Store);
        let mut p = p.compile();
        let rect = self.size().to_screen_int_rect(0, 0);
        p.run(
            &rect,
            pipeline::AAMaskCtx::default(),
            mask.as_submask().mask_ctx(),
            pixmap_src,
            &mut self.as_subpixmap(),
        );
    }
}

fn treat_as_hairline(paint: &Paint, stroke: &Stroke, mut ts: Transform) -> Option<f32> {
    fn fast_len(p: Point) -> f32 {
        let mut x = p.x.abs();
        let mut y = p.y.abs();
        if x < y {
            core::mem::swap(&mut x, &mut y);
        }

        x + y.half()
    }

    debug_assert!(stroke.width >= 0.0);

    if stroke.width == 0.0 {
        return Some(1.0);
    }

    if !paint.anti_alias {
        return None;
    }

    // We don't care about translate.
    ts.tx = 0.0;
    ts.ty = 0.0;

    // We need to try to fake a thick-stroke with a modulated hairline.
    let mut points = [
        Point::from_xy(stroke.width, 0.0),
        Point::from_xy(0.0, stroke.width),
    ];
    ts.map_points(&mut points);

    let len0 = fast_len(points[0]);
    let len1 = fast_len(points[1]);

    if len0 <= 1.0 && len1 <= 1.0 {
        return Some(len0.ave(len1));
    }

    None
}

/// Sometimes in the drawing pipeline, we have to perform math on path coordinates, even after
/// the path is in device-coordinates. Tessellation and clipping are two examples. Usually this
/// is pretty modest, but it can involve subtracting/adding coordinates, or multiplying by
/// small constants (e.g. 2,3,4). To try to preflight issues where these optionations could turn
/// finite path values into infinities (or NaNs), we allow the upper drawing code to reject
/// the path if its bounds (in device coordinates) is too close to max float.
pub(crate) fn is_too_big_for_math(path: &Path) -> bool {
    // This value is just a guess. smaller is safer, but we don't want to reject largish paths
    // that we don't have to.
    const SCALE_DOWN_TO_ALLOW_FOR_SMALL_MULTIPLIES: f32 = 0.25;
    const MAX: f32 = SCALAR_MAX * SCALE_DOWN_TO_ALLOW_FOR_SMALL_MULTIPLIES;

    let b = path.bounds();

    // use ! expression so we return true if bounds contains NaN
    !(b.left() >= -MAX && b.top() >= -MAX && b.right() <= MAX && b.bottom() <= MAX)
}

/// Splits the target pixmap into a list of tiles.
///
/// Skia/tiny-skia uses a lot of fixed-point math during path rendering.
/// Probably more for precision than performance.
/// And our fixed-point types are limited by 8192 and 32768.
/// Which means that we cannot render a path larger than 8192 onto a pixmap.
/// When pixmap is smaller than 8192, the path will be automatically clipped anyway,
/// but for large pixmaps we have to render in tiles.
pub(crate) struct DrawTiler {
    image_width: u32,
    image_height: u32,
    x_offset: u32,
    y_offset: u32,
    finished: bool,
}

impl DrawTiler {
    // 8K is 1 too big, since 8K << supersample == 32768 which is too big for Fixed.
    const MAX_DIMENSIONS: u32 = 8192 - 1;

    fn required(image_width: u32, image_height: u32) -> bool {
        image_width > Self::MAX_DIMENSIONS || image_height > Self::MAX_DIMENSIONS
    }

    pub(crate) fn new(image_width: u32, image_height: u32) -> Option<Self> {
        if Self::required(image_width, image_height) {
            Some(DrawTiler {
                image_width,
                image_height,
                x_offset: 0,
                y_offset: 0,
                finished: false,
            })
        } else {
            None
        }
    }
}

impl Iterator for DrawTiler {
    type Item = ScreenIntRect;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // TODO: iterate only over tiles that actually affected by the shape

        if self.x_offset < self.image_width && self.y_offset < self.image_height {
            let h = if self.y_offset < self.image_height {
                (self.image_height - self.y_offset).min(Self::MAX_DIMENSIONS)
            } else {
                self.image_height
            };

            let r = ScreenIntRect::from_xywh(
                self.x_offset,
                self.y_offset,
                (self.image_width - self.x_offset).min(Self::MAX_DIMENSIONS),
                h,
            );

            self.x_offset += Self::MAX_DIMENSIONS;
            if self.x_offset >= self.image_width {
                self.x_offset = 0;
                self.y_offset += Self::MAX_DIMENSIONS;
            }

            return r;
        }

        None
    }
}

#[cfg(feature = "16bpc")]
impl DynamicPixmap {
    /// Draws a filled rectangle onto the dynamic pixmap.
    pub fn fill_rect(
        &mut self,
        rect: Rect,
        paint: &Paint,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        match self {
            DynamicPixmap::U8(p) => p.fill_rect(rect, paint, transform, mask),
            DynamicPixmap::U16(p) => p.fill_rect(rect, paint, transform, mask),
        }
    }

    /// Draws a filled path onto the dynamic pixmap.
    pub fn fill_path(
        &mut self,
        path: &Path,
        paint: &Paint,
        fill_rule: FillRule,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        match self {
            DynamicPixmap::U8(p) => p.fill_path(path, paint, fill_rule, transform, mask),
            DynamicPixmap::U16(p) => p.fill_path(path, paint, fill_rule, transform, mask),
        }
    }

    /// Strokes a path onto the dynamic pixmap.
    pub fn stroke_path(
        &mut self,
        path: &Path,
        paint: &Paint,
        stroke: &Stroke,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        match self {
            DynamicPixmap::U8(p) => p.stroke_path(path, paint, stroke, transform, mask),
            DynamicPixmap::U16(p) => p.stroke_path(path, paint, stroke, transform, mask),
        }
    }

    /// Draws a `Pixmap` on top of the dynamic pixmap.
    pub fn draw_pixmap(
        &mut self,
        x: i32,
        y: i32,
        pixmap: PixmapRef,
        paint: &PixmapPaint,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        match self {
            DynamicPixmap::U8(p) => p.draw_pixmap(x, y, pixmap, paint, transform, mask),
            DynamicPixmap::U16(p) => {
                let u8_pixels = pixmap.pixels();
                let mut u16_pixmap = PixmapU16::new(pixmap.width(), pixmap.height()).unwrap();
                for (src, dst) in u8_pixels.iter().zip(u16_pixmap.pixels_mut().iter_mut()) {
                    *dst = PremultipliedColorU16::from_color_u8(*src);
                }
                p.draw_pixmap(x, y, u16_pixmap.as_ref(), paint, transform, mask);
            }
        }
    }

    /// Applies a mask to the dynamic pixmap.
    pub fn apply_mask(&mut self, mask: &Mask) {
        match self {
            DynamicPixmap::U8(p) => p.apply_mask(mask),
            DynamicPixmap::U16(p) => p.apply_mask(mask),
        }
    }
}

#[cfg(feature = "16bpc")]
impl DynamicPixmapMut<'_> {
    /// Draws a filled rectangle onto the dynamic mutable pixmap.
    pub fn fill_rect(
        &mut self,
        rect: Rect,
        paint: &Paint,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        match self {
            DynamicPixmapMut::U8(p) => p.fill_rect(rect, paint, transform, mask),
            DynamicPixmapMut::U16(p) => p.fill_rect(rect, paint, transform, mask),
        }
    }

    /// Draws a filled path onto the dynamic mutable pixmap.
    pub fn fill_path(
        &mut self,
        path: &Path,
        paint: &Paint,
        fill_rule: FillRule,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        match self {
            DynamicPixmapMut::U8(p) => p.fill_path(path, paint, fill_rule, transform, mask),
            DynamicPixmapMut::U16(p) => p.fill_path(path, paint, fill_rule, transform, mask),
        }
    }

    /// Strokes a path onto the dynamic mutable pixmap.
    pub fn stroke_path(
        &mut self,
        path: &Path,
        paint: &Paint,
        stroke: &Stroke,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        match self {
            DynamicPixmapMut::U8(p) => p.stroke_path(path, paint, stroke, transform, mask),
            DynamicPixmapMut::U16(p) => p.stroke_path(path, paint, stroke, transform, mask),
        }
    }

    /// Draws a `Pixmap` on top of the dynamic mutable pixmap.
    pub fn draw_pixmap(
        &mut self,
        x: i32,
        y: i32,
        pixmap: PixmapRef,
        paint: &PixmapPaint,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        match self {
            DynamicPixmapMut::U8(p) => p.draw_pixmap(x, y, pixmap, paint, transform, mask),
            DynamicPixmapMut::U16(p) => {
                let u8_pixels = pixmap.pixels();
                let mut u16_pixmap = PixmapU16::new(pixmap.width(), pixmap.height()).unwrap();
                for (src, dst) in u8_pixels.iter().zip(u16_pixmap.pixels_mut().iter_mut()) {
                    *dst = PremultipliedColorU16::from_color_u8(*src);
                }
                p.draw_pixmap(x, y, u16_pixmap.as_ref(), paint, transform, mask);
            }
        }
    }

    /// Applies a mask to the dynamic mutable pixmap.
    pub fn apply_mask(&mut self, mask: &Mask) {
        match self {
            DynamicPixmapMut::U8(p) => p.apply_mask(mask),
            DynamicPixmapMut::U16(p) => p.apply_mask(mask),
        }
    }
}

#[cfg(feature = "16bpc")]
#[inline]
fn blend_u16_op(
    dst: PremultipliedColorU16,
    src: PremultipliedColorU16,
    mode: BlendMode,
) -> PremultipliedColorU16 {
    match mode {
        BlendMode::Source => src,
        BlendMode::Destination => dst,
        BlendMode::Clear => PremultipliedColorU16::TRANSPARENT,
        BlendMode::SourceOver => {
            let inv_a = 65535 - src.alpha() as u32;
            let r = src.red() as u32 + ((dst.red() as u32 * inv_a + 32768) >> 16);
            let g = src.green() as u32 + ((dst.green() as u32 * inv_a + 32768) >> 16);
            let b = src.blue() as u32 + ((dst.blue() as u32 * inv_a + 32768) >> 16);
            let a = src.alpha() as u32 + ((dst.alpha() as u32 * inv_a + 32768) >> 16);
            PremultipliedColorU16::from_rgba_unchecked(
                r.min(65535) as u16,
                g.min(65535) as u16,
                b.min(65535) as u16,
                a.min(65535) as u16,
            )
        }
        BlendMode::DestinationOver => {
            let inv_a = 65535 - dst.alpha() as u32;
            let r = dst.red() as u32 + ((src.red() as u32 * inv_a + 32768) >> 16);
            let g = dst.green() as u32 + ((src.green() as u32 * inv_a + 32768) >> 16);
            let b = dst.blue() as u32 + ((src.blue() as u32 * inv_a + 32768) >> 16);
            let a = dst.alpha() as u32 + ((src.alpha() as u32 * inv_a + 32768) >> 16);
            PremultipliedColorU16::from_rgba_unchecked(
                r.min(65535) as u16,
                g.min(65535) as u16,
                b.min(65535) as u16,
                a.min(65535) as u16,
            )
        }
        BlendMode::SourceIn => {
            let a = dst.alpha() as u32;
            let r = ((src.red() as u32 * a + 32768) >> 16) as u16;
            let g = ((src.green() as u32 * a + 32768) >> 16) as u16;
            let b = ((src.blue() as u32 * a + 32768) >> 16) as u16;
            let a = ((src.alpha() as u32 * a + 32768) >> 16) as u16;
            PremultipliedColorU16::from_rgba_unchecked(r, g, b, a)
        }
        BlendMode::DestinationIn => {
            let a = src.alpha() as u32;
            let r = ((dst.red() as u32 * a + 32768) >> 16) as u16;
            let g = ((dst.green() as u32 * a + 32768) >> 16) as u16;
            let b = ((dst.blue() as u32 * a + 32768) >> 16) as u16;
            let a = ((dst.alpha() as u32 * a + 32768) >> 16) as u16;
            PremultipliedColorU16::from_rgba_unchecked(r, g, b, a)
        }
        BlendMode::SourceOut => {
            let inv_a = 65535 - dst.alpha() as u32;
            let r = ((src.red() as u32 * inv_a + 32768) >> 16) as u16;
            let g = ((src.green() as u32 * inv_a + 32768) >> 16) as u16;
            let b = ((src.blue() as u32 * inv_a + 32768) >> 16) as u16;
            let a = ((src.alpha() as u32 * inv_a + 32768) >> 16) as u16;
            PremultipliedColorU16::from_rgba_unchecked(r, g, b, a)
        }
        BlendMode::DestinationOut => {
            let inv_a = 65535 - src.alpha() as u32;
            let r = ((dst.red() as u32 * inv_a + 32768) >> 16) as u16;
            let g = ((dst.green() as u32 * inv_a + 32768) >> 16) as u16;
            let b = ((dst.blue() as u32 * inv_a + 32768) >> 16) as u16;
            let a = ((dst.alpha() as u32 * inv_a + 32768) >> 16) as u16;
            PremultipliedColorU16::from_rgba_unchecked(r, g, b, a)
        }
        BlendMode::SourceAtop => {
            let da = dst.alpha() as u32;
            let inv_sa = 65535 - src.alpha() as u32;
            let r = ((src.red() as u32 * da + dst.red() as u32 * inv_sa + 32768) >> 16) as u16;
            let g = ((src.green() as u32 * da + dst.green() as u32 * inv_sa + 32768) >> 16) as u16;
            let b = ((src.blue() as u32 * da + dst.blue() as u32 * inv_sa + 32768) >> 16) as u16;
            let a = dst.alpha();
            PremultipliedColorU16::from_rgba_unchecked(r, g, b, a)
        }
        BlendMode::DestinationAtop => {
            let sa = src.alpha() as u32;
            let inv_da = 65535 - dst.alpha() as u32;
            let r = ((dst.red() as u32 * sa + src.red() as u32 * inv_da + 32768) >> 16) as u16;
            let g = ((dst.green() as u32 * sa + src.green() as u32 * inv_da + 32768) >> 16) as u16;
            let b = ((dst.blue() as u32 * sa + src.blue() as u32 * inv_da + 32768) >> 16) as u16;
            let a = src.alpha();
            PremultipliedColorU16::from_rgba_unchecked(r, g, b, a)
        }
        BlendMode::Xor => {
            let inv_da = 65535 - dst.alpha() as u32;
            let inv_sa = 65535 - src.alpha() as u32;
            let r = ((src.red() as u32 * inv_da + dst.red() as u32 * inv_sa + 32768) >> 16) as u16;
            let g =
                ((src.green() as u32 * inv_da + dst.green() as u32 * inv_sa + 32768) >> 16) as u16;
            let b =
                ((src.blue() as u32 * inv_da + dst.blue() as u32 * inv_sa + 32768) >> 16) as u16;
            let a =
                ((src.alpha() as u32 * inv_da + dst.alpha() as u32 * inv_sa + 32768) >> 16) as u16;
            PremultipliedColorU16::from_rgba_unchecked(r, g, b, a)
        }
        BlendMode::Plus => {
            let r = (dst.red() as u32 + src.red() as u32).min(65535) as u16;
            let g = (dst.green() as u32 + src.green() as u32).min(65535) as u16;
            let b = (dst.blue() as u32 + src.blue() as u32).min(65535) as u16;
            let a = (dst.alpha() as u32 + src.alpha() as u32).min(65535) as u16;
            PremultipliedColorU16::from_rgba_unchecked(r, g, b, a)
        }
        _ => {
            let sr = src.red() as f32 / 65535.0;
            let sg = src.green() as f32 / 65535.0;
            let sb = src.blue() as f32 / 65535.0;
            let sa = src.alpha() as f32 / 65535.0;

            let dr = dst.red() as f32 / 65535.0;
            let dg = dst.green() as f32 / 65535.0;
            let db = dst.blue() as f32 / 65535.0;
            let da = dst.alpha() as f32 / 65535.0;

            let (r, g, b) = match mode {
                BlendMode::Hue => {
                    let (mut rr, mut gg, mut bb) = (sr * sa, sg * sa, sb * sa);
                    set_sat(&mut rr, &mut gg, &mut bb, sat(dr, dg, db) * sa);
                    set_lum(&mut rr, &mut gg, &mut bb, lum(dr, dg, db) * sa);
                    clip_color(&mut rr, &mut gg, &mut bb, sa * da);
                    (
                        sr * (1.0 - da) + dr * (1.0 - sa) + rr,
                        sg * (1.0 - da) + dg * (1.0 - sa) + gg,
                        sb * (1.0 - da) + db * (1.0 - sa) + bb,
                    )
                }
                BlendMode::Saturation => {
                    let (mut rr, mut gg, mut bb) = (dr * sa, dg * sa, db * sa);
                    set_sat(&mut rr, &mut gg, &mut bb, sat(sr, sg, sb) * da);
                    set_lum(&mut rr, &mut gg, &mut bb, lum(dr, dg, db) * sa);
                    clip_color(&mut rr, &mut gg, &mut bb, sa * da);
                    (
                        sr * (1.0 - da) + dr * (1.0 - sa) + rr,
                        sg * (1.0 - da) + dg * (1.0 - sa) + gg,
                        sb * (1.0 - da) + db * (1.0 - sa) + bb,
                    )
                }
                BlendMode::Color => {
                    let (mut rr, mut gg, mut bb) = (sr * da, sg * da, sb * da);
                    set_lum(&mut rr, &mut gg, &mut bb, lum(dr, dg, db) * sa);
                    clip_color(&mut rr, &mut gg, &mut bb, sa * da);
                    (
                        sr * (1.0 - da) + dr * (1.0 - sa) + rr,
                        sg * (1.0 - da) + dg * (1.0 - sa) + gg,
                        sb * (1.0 - da) + db * (1.0 - sa) + bb,
                    )
                }
                BlendMode::Luminosity => {
                    let (mut rr, mut gg, mut bb) = (dr * sa, dg * sa, db * sa);
                    set_lum(&mut rr, &mut gg, &mut bb, lum(sr, sg, sb) * da);
                    clip_color(&mut rr, &mut gg, &mut bb, sa * da);
                    (
                        sr * (1.0 - da) + dr * (1.0 - sa) + rr,
                        sg * (1.0 - da) + dg * (1.0 - sa) + gg,
                        sb * (1.0 - da) + db * (1.0 - sa) + bb,
                    )
                }
                _ => {
                    let blend_ch = |s: f32, d: f32| -> f32 {
                        match mode {
                            BlendMode::Multiply => s * (1.0 - da) + d * (1.0 - sa) + s * d,
                            BlendMode::Screen => s + d - s * d,
                            BlendMode::Darken => s + d - (s * da).max(d * sa),
                            BlendMode::Lighten => s + d - (s * da).min(d * sa),
                            BlendMode::Difference => s + d - 2.0 * (s * da).min(d * sa),
                            BlendMode::Exclusion => s + d - 2.0 * s * d,
                            BlendMode::ColorDodge => {
                                if d <= 0.0 {
                                    s * (1.0 - da)
                                } else if s >= sa {
                                    s + d * (1.0 - sa)
                                } else {
                                    sa * da.min((d * sa) / (sa - s))
                                        + s * (1.0 - da)
                                        + d * (1.0 - sa)
                                }
                            }
                            BlendMode::ColorBurn => {
                                if d >= da {
                                    d + s * (1.0 - da)
                                } else if s <= 0.0 {
                                    d * (1.0 - sa)
                                } else {
                                    sa * (da - da.min((da - d) * sa / s))
                                        + s * (1.0 - da)
                                        + d * (1.0 - sa)
                                }
                            }
                            BlendMode::Overlay => {
                                let b = if 2.0 * d <= da {
                                    2.0 * s * d
                                } else {
                                    sa * da - 2.0 * (da - d) * (sa - s)
                                };
                                s * (1.0 - da) + d * (1.0 - sa) + b
                            }
                            BlendMode::HardLight => {
                                let b = if 2.0 * s <= sa {
                                    2.0 * s * d
                                } else {
                                    sa * da - 2.0 * (da - d) * (sa - s)
                                };
                                s * (1.0 - da) + d * (1.0 - sa) + b
                            }
                            BlendMode::SoftLight => {
                                let m = if da > 0.0 { d / da } else { 0.0 };
                                let s2 = 2.0 * s;
                                let m4 = 4.0 * m;
                                let dark_src = d * (sa + (s2 - sa) * (1.0 - m));
                                let dark_dst = (m4 * m4 + m4) * (m - 1.0) + 7.0 * m;
                                let lite_dst = m.sqrt() - m;
                                let lite_src = d * sa
                                    + da * (s2 - sa)
                                        * if 4.0 * d <= da { dark_dst } else { lite_dst };
                                s * (1.0 - da)
                                    + d * (1.0 - sa)
                                    + if s2 <= sa { dark_src } else { lite_src }
                            }
                            _ => s + d * (1.0 - sa),
                        }
                    };
                    (blend_ch(sr, dr), blend_ch(sg, dg), blend_ch(sb, db))
                }
            };

            let a = sa + da - sa * da;

            let r16 = (r.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            let g16 = (g.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            let b16 = (b.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            let a16 = (a.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            PremultipliedColorU16::from_rgba_unchecked(r16, g16, b16, a16)
        }
    }
}

#[cfg(feature = "16bpc")]
#[inline(always)]
fn sat(r: f32, g: f32, b: f32) -> f32 {
    r.max(g.max(b)) - r.min(g.min(b))
}

#[cfg(feature = "16bpc")]
#[inline(always)]
fn lum(r: f32, g: f32, b: f32) -> f32 {
    r * 0.30 + g * 0.59 + b * 0.11
}

#[cfg(feature = "16bpc")]
#[inline(always)]
fn set_sat(r: &mut f32, g: &mut f32, b: &mut f32, s: f32) {
    let mn = r.min(g.min(*b));
    let mx = r.max(g.max(*b));
    let sat = mx - mn;
    if sat > 0.0 {
        *r = (*r - mn) * s / sat;
        *g = (*g - mn) * s / sat;
        *b = (*b - mn) * s / sat;
    } else {
        *r = 0.0;
        *g = 0.0;
        *b = 0.0;
    }
}

#[cfg(feature = "16bpc")]
#[inline(always)]
fn set_lum(r: &mut f32, g: &mut f32, b: &mut f32, l: f32) {
    let diff = l - lum(*r, *g, *b);
    *r += diff;
    *g += diff;
    *b += diff;
}

#[cfg(feature = "16bpc")]
#[inline(always)]
fn clip_color(r: &mut f32, g: &mut f32, b: &mut f32, a: f32) {
    let mn = r.min(g.min(*b));
    let mx = r.max(g.max(*b));
    let l = lum(*r, *g, *b);

    if mn < 0.0 {
        if (l - mn).abs() > 0.0 {
            *r = l + (*r - l) * l / (l - mn);
            *g = l + (*g - l) * l / (l - mn);
            *b = l + (*b - l) * l / (l - mn);
        } else {
            *r = l;
            *g = l;
            *b = l;
        }
    }

    if mx > a {
        if (mx - l).abs() > 0.0 {
            *r = l + (*r - l) * (a - l) / (mx - l);
            *g = l + (*g - l) * (a - l) / (mx - l);
            *b = l + (*b - l) * (a - l) / (mx - l);
        } else {
            *r = l;
            *g = l;
            *b = l;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MAX_DIM: u32 = DrawTiler::MAX_DIMENSIONS;

    #[test]
    fn skip() {
        assert!(DrawTiler::new(100, 500).is_none());
    }

    #[test]
    fn horizontal() {
        let mut iter = DrawTiler::new(10000, 500).unwrap();
        assert_eq!(iter.next(), ScreenIntRect::from_xywh(0, 0, MAX_DIM, 500));
        assert_eq!(
            iter.next(),
            ScreenIntRect::from_xywh(MAX_DIM, 0, 10000 - MAX_DIM, 500)
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn vertical() {
        let mut iter = DrawTiler::new(500, 10000).unwrap();
        assert_eq!(iter.next(), ScreenIntRect::from_xywh(0, 0, 500, MAX_DIM));
        assert_eq!(
            iter.next(),
            ScreenIntRect::from_xywh(0, MAX_DIM, 500, 10000 - MAX_DIM)
        );
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn rect() {
        let mut iter = DrawTiler::new(10000, 10000).unwrap();
        // Row 1
        assert_eq!(
            iter.next(),
            ScreenIntRect::from_xywh(0, 0, MAX_DIM, MAX_DIM)
        );
        assert_eq!(
            iter.next(),
            ScreenIntRect::from_xywh(MAX_DIM, 0, 10000 - MAX_DIM, MAX_DIM)
        );
        // Row 2
        assert_eq!(
            iter.next(),
            ScreenIntRect::from_xywh(0, MAX_DIM, MAX_DIM, 10000 - MAX_DIM)
        );
        assert_eq!(
            iter.next(),
            ScreenIntRect::from_xywh(MAX_DIM, MAX_DIM, 10000 - MAX_DIM, 10000 - MAX_DIM)
        );
        assert_eq!(iter.next(), None);
    }
}
