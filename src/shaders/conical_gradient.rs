// Copyright 2006 The Android Open Source Project
// Copyright 2020 Yevhenii Reizner
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.


use alloc::vec;
use alloc::vec::Vec;

use tiny_skia_path::Scalar;

use crate::{ColorSpace, GradientStop, Point, Shader, SpreadMode, Transform};

use super::gradient::{Gradient, DEGENERATE_THRESHOLD};
use crate::pipeline;
use crate::pipeline::RasterPipelineBuilder;
use crate::wide::u32x8;

#[cfg(all(not(feature = "std"), feature = "no-std-float"))]
use tiny_skia_path::NoStdFloat;

#[derive(Copy, Clone, PartialEq, Debug, Default)]
struct FocalData {
    r1: f32,      // r1 after mapping focal point to (0, 0)
    focal_x: f32, // f
    is_swapped: bool,
}

impl FocalData {
    // From: https://source.chromium.org/chromium/chromium/src/+/main:third_party/skia/src/shaders/gradients/SkConicalGradient.cpp;l=36;drc=075316994c97ee86961b369bb2bff246aaa9d6c4
    fn set(&mut self, mut r0: f32, mut r1: f32, matrix: &mut Transform) -> bool {
        self.is_swapped = false;
        self.focal_x = r0 / (r0 - r1);

        if (self.focal_x - 1.0).is_nearly_zero() {
            // swap r0, r1
            *matrix = matrix.post_translate(-1.0, 0.0).post_scale(-1.0, 1.0);
            core::mem::swap(&mut r0, &mut r1);

            self.focal_x = 0.0; // because r0 is now 0
            self.is_swapped = true;
        }

        // Map {focal point, (1, 0)} to {(0, 0), (1, 0)}
        let from = [Point::from_xy(self.focal_x, 0.0), Point::from_xy(1.0, 0.0)];
        let to = [Point::from_xy(0.0, 0.0), Point::from_xy(1.0, 0.0)];

        let focal_matrix = match ts_from_poly_to_poly(from[0], from[1], to[0], to[1]) {
            Some(m) => m,
            None => return false,
        };

        *matrix = matrix.post_concat(focal_matrix);
        self.r1 = r1 / (1.0 - self.focal_x).abs(); // focalMatrix has a scale of 1/(1-f).

        // The following transformations are just to accelerate the shader computation by saving
        // some arithmetic operations.
        if self.is_focal_on_circle() {
            *matrix = matrix.post_scale(0.5, 0.5);
        } else {
            *matrix = matrix.post_scale(
                self.r1 / (self.r1 * self.r1 - 1.0),
                1.0 / (self.r1 * self.r1 - 1.0).abs().sqrt(),
            );
        }

        *matrix = matrix.post_scale((1.0 - self.focal_x).abs(), (1.0 - self.focal_x).abs()); // scale |1 - f|

        true
    }

    fn is_focal_on_circle(&self) -> bool {
        (1.0 - self.r1).is_nearly_zero()
    }

    fn is_well_behaved(&self) -> bool {
        !self.is_focal_on_circle() && self.r1 > 1.0
    }

    fn is_natively_focal(&self) -> bool {
        self.focal_x.is_nearly_zero()
    }
}

#[derive(Clone, PartialEq, Debug)]
enum GradientType {
    Radial,
    Strip,
    Focal(FocalData),
}

/// A 2-point conical gradient shader.
#[derive(Clone, PartialEq, Debug)]
pub struct ConicalGradient {
    pub(crate) base: Gradient,
    center1: Point,
    center2: Point,
    radius1: f32,
    radius2: f32,
    gradient_type: GradientType,
}

impl ConicalGradient {
    /// Creates a new 2-point conical gradient shader.
    ///
    /// From: https://source.chromium.org/chromium/chromium/src/+/main:third_party/skia/src/shaders/gradients/SkConicalGradient.cpp;l=251;drc=075316994c97ee86961b369bb2bff246aaa9d6c4
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        start_point: Point,
        start_radius: f32,
        end_point: Point,
        end_radius: f32,
        stops: Vec<GradientStop>,
        mode: SpreadMode,
        transform: Transform,
    ) -> Option<Shader<'static>> {
        if start_radius < 0.0 || end_radius < 0.0 {
            return None;
        }

        transform.invert()?;

        match stops.as_slice() {
            [] => return None,
            [stop] => return Some(Shader::SolidColor(stop.color)),
            _ => {}
        }

        let length = (start_point - end_point).length();
        if length.is_nearly_zero_within_tolerance(DEGENERATE_THRESHOLD) {
            if start_radius.is_nearly_equal_within_tolerance(end_radius, DEGENERATE_THRESHOLD) {
                // Degenerate case, where the interpolation region area approaches zero. The proper
                // behavior depends on the tile mode, which is consistent with the default degenerate
                // gradient behavior, except when mode = clamp and the radii > 0.
                if mode == SpreadMode::Pad && end_radius > DEGENERATE_THRESHOLD {
                    // The interpolation region becomes an infinitely thin ring at the radius, so the
                    // final gradient will be the first color repeated from p=0 to 1, and then a hard
                    // stop switching to the last color at p=1.
                    let front = stops.first()?;
                    let back = stops.last()?;
                    let new_stops = vec![
                        GradientStop::new(0.0, front.color),
                        GradientStop::new(1.0, front.color),
                        GradientStop::new(1.0, back.color),
                    ];
                    return ConicalGradient::new_radial(
                        start_point,
                        end_point,
                        end_radius,
                        new_stops,
                        mode,
                        transform,
                    );
                }
                // TODO: Consider making a degenerate gradient
                return None;
            }

            if start_radius.is_nearly_zero_within_tolerance(DEGENERATE_THRESHOLD) {
                return ConicalGradient::new_radial(
                    start_point,
                    end_point,
                    end_radius,
                    stops,
                    mode,
                    transform,
                );
            }
        }

        create(
            start_point,
            start_radius,
            end_point,
            end_radius,
            stops,
            mode,
            transform,
        )
    }

    /// Creates a new radial gradient shader.
    ///
    /// Returns `Shader::SolidColor` when:
    /// - `stops.len()` == 1
    ///
    /// Returns `None` when:
    ///
    /// - `stops` is empty
    /// - `radius` <= 0
    /// - `transform` is not invertible
    #[allow(clippy::new_ret_no_self)]
    pub fn new_radial(
        start: Point,
        end: Point,
        radius: f32,
        stops: Vec<GradientStop>,
        mode: SpreadMode,
        transform: Transform,
    ) -> Option<Shader<'static>> {
        let length = (end - start).length();
        if !length.is_finite() {
            return None;
        }

        if length.is_nearly_zero_within_tolerance(DEGENERATE_THRESHOLD) {
            // If the center positions are the same, then the gradient
            // is the radial variant of a 2 pt conical gradient,
            // an actual radial gradient (startRadius == 0),
            // or it is fully degenerate (startRadius == endRadius).

            let inv = radius.invert();
            let mut ts = Transform::from_translate(-start.x, -start.y);
            ts = ts.post_scale(inv, inv);

            // We can treat this gradient as radial, which is faster. If we got here, we know
            // that endRadius is not equal to 0, so this produces a meaningful gradient
            Some(Shader::ConicalGradient(ConicalGradient {
                base: Gradient::new(stops, mode, transform, ts),
                center1: start,
                center2: end,
                radius1: 0.0,
                radius2: radius,
                gradient_type: GradientType::Radial,
            }))
        } else {
            // From SkTwoPointConicalGradient::Create
            let mut ts = ts_from_poly_to_poly(
                start,
                end,
                Point::from_xy(0.0, 0.0),
                Point::from_xy(1.0, 0.0),
            )?;

            let d_center = (start - end).length();
            let r1 = radius / d_center;
            let focal_data = FocalData { r1, focal_x: 0.0, is_swapped: false };

            // The following transformations are just to accelerate the shader computation by saving
            // some arithmetic operations.
            if focal_data.is_focal_on_circle() {
                ts = ts.post_scale(0.5, 0.5);
            } else {
                ts = ts.post_scale(r1 / (r1 * r1 - 1.0), 1.0 / ((r1 * r1 - 1.0).abs()).sqrt());
            }

            Some(Shader::ConicalGradient(ConicalGradient {
                base: Gradient::new(stops, mode, transform, ts),
                center1: start,
                center2: end,
                radius1: 0.0,
                radius2: radius,
                gradient_type: GradientType::Focal(focal_data),
            }))
        }
    }

    // From: https://source.chromium.org/chromium/chromium/src/+/main:third_party/skia/src/shaders/gradients/SkConicalGradient.cpp;l=194;drc=075316994c97ee86961b369bb2bff246aaa9d6c4
    pub(crate) fn push_stages(&self, cs: ColorSpace, p: &mut RasterPipelineBuilder) -> bool {
        let (p0, p1) = match self.gradient_type {
            GradientType::Radial => {
                if self.radius1 == 0.0 {
                    (1.0, 0.0)
                } else {
                    let d_radius = self.radius2 - self.radius1;
                    // For concentric gradients: t = t * scale + bias
                    let p0 = self.radius1.max(self.radius2) / d_radius;
                    let p1 = -self.radius1 / d_radius;
                    (p0, p1)
                }
            }
            GradientType::Strip => {
                let scaled_r0 = self.radius1 / (self.center1 - self.center2).length();
                (scaled_r0 * scaled_r0, 0.0 /*unused*/)
            }
            GradientType::Focal(fd) => (1.0 / fd.r1, fd.focal_x),
        };

        p.ctx.two_point_conical_gradient = pipeline::TwoPointConicalGradientCtx {
            mask: u32x8::default(),
            p0,
            p1,
        };

        self.base.push_stages(
            p,
            cs,
            &|p| {
                match self.gradient_type {
                    GradientType::Radial => {
                        p.push(pipeline::Stage::XYToRadius);
                        // Apply scale/bias to map t from [0, 1] based on r_max to proper t where
                        // t=0 at r0 and t=1 at r1
                        if (p0, p1) != (1.0, 0.0) {
                            p.push(pipeline::Stage::ApplyConcentricScaleBias);
                        }
                    }
                    GradientType::Strip => {
                        p.push(pipeline::Stage::XYTo2PtConicalStrip);
                        p.push(pipeline::Stage::Mask2PtConicalNan);
                    }
                    GradientType::Focal(fd) => {
                        if fd.is_focal_on_circle() {
                            p.push(pipeline::Stage::XYTo2PtConicalFocalOnCircle);
                        } else if fd.is_well_behaved() {
                            p.push(pipeline::Stage::XYTo2PtConicalWellBehaved);
                        } else if fd.is_swapped || (1.0 - fd.focal_x) < 0.0 {
                            p.push(pipeline::Stage::XYTo2PtConicalSmaller);
                        } else {
                            p.push(pipeline::Stage::XYTo2PtConicalGreater);
                        }

                        if !fd.is_well_behaved() {
                            p.push(pipeline::Stage::Mask2PtConicalDegenerates);
                        }

                        if (1.0 - fd.focal_x) < 0.0 {
                            p.push(pipeline::Stage::NegateX);
                        }

                        if !fd.is_natively_focal() {
                            p.push(pipeline::Stage::Alter2PtConicalCompensateFocal);
                        }

                        if fd.is_swapped {
                            p.push(pipeline::Stage::Alter2PtConicalUnswap);
                        }
                    }
                }
            },
            &|p| match self.gradient_type {
                GradientType::Strip => p.push(pipeline::Stage::ApplyVectorMask),
                GradientType::Focal(fd) if !fd.is_well_behaved() => {
                    p.push(pipeline::Stage::ApplyVectorMask)
                }
                _ => {}
            },
        )
    }
}

// https://source.chromium.org/chromium/chromium/src/+/main:third_party/skia/src/shaders/gradients/SkConicalGradient.cpp;l=76;drc=075316994c97ee86961b369bb2bff246aaa9d6c4
fn create(
    c0: Point,
    r0: f32,
    c1: Point,
    r1: f32,
    stops: Vec<GradientStop>,
    mode: SpreadMode,
    transform: Transform,
) -> Option<Shader<'static>> {
    let mut gradient_type;
    let mut gradient_matrix;

    if (c0 - c1).length().is_nearly_zero() {
        if r0.max(r1).is_nearly_zero() || r0.is_nearly_equal(r1) {
            // Degenerate case; avoid dividing by zero. Should have been caught
            // by caller but just in case, recheck here.
            return None;
        }

        // Concentric case: we can pretend we're radial (with a tiny twist).
        let scale = 1.0 / r0.max(r1);
        gradient_matrix = Transform::from_translate(-c1.x, -c1.y).post_scale(scale, scale);
        gradient_type = GradientType::Radial;
    } else {
        gradient_matrix = map_to_unit_x(c0, c1)?;
        gradient_type = if (r0 - r1).is_nearly_zero() {
            GradientType::Strip
        } else {
            GradientType::Focal(FocalData::default())
        };
    }
    if let GradientType::Focal(ref mut focal_data) = &mut gradient_type {
        let d_center = (c0 - c1).length();
        if !focal_data.set(r0 / d_center, r1 / d_center, &mut gradient_matrix) {
            return None;
        }
    }

    Some(Shader::ConicalGradient(ConicalGradient {
        base: Gradient::new(stops, mode, transform, gradient_matrix),
        center1: c0,
        center2: c1,
        radius1: r0,
        radius2: r1,
        gradient_type,
    }))
}

// From: https://source.chromium.org/chromium/chromium/src/+/main:third_party/skia/src/shaders/gradients/SkConicalGradient.cpp;l=69;drc=075316994c97ee86961b369bb2bff246aaa9d6c4
fn map_to_unit_x(origin: Point, x_is_one: Point) -> Option<Transform> {
    ts_from_poly_to_poly(
        origin,
        x_is_one,
        Point::from_xy(0.0, 0.0),
        Point::from_xy(1.0, 0.0),
    )
}

fn ts_from_poly_to_poly(src1: Point, src2: Point, dst1: Point, dst2: Point) -> Option<Transform> {
    let tmp = from_poly2(src1, src2);
    let res = tmp.invert()?;
    let tmp = from_poly2(dst1, dst2);
    Some(tmp.pre_concat(res))
}

fn from_poly2(p0: Point, p1: Point) -> Transform {
    Transform::from_row(
        p1.y - p0.y,
        p0.x - p1.x,
        p1.x - p0.x,
        p1.y - p0.y,
        p0.x,
        p0.y,
    )
}
