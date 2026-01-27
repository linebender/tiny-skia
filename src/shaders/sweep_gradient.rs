// Copyright 2006 The Android Open Source Project
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use alloc::vec::Vec;

use tiny_skia_path::Scalar;

use crate::{ColorSpace, GradientStop, Point, Shader, SpreadMode, Transform};

use super::gradient::{Gradient, DEGENERATE_THRESHOLD};
use crate::pipeline::{RasterPipelineBuilder, Stage};

#[cfg(all(not(feature = "std"), feature = "no-std-float"))]
use tiny_skia_path::NoStdFloat;

/// A radial gradient.
#[derive(Clone, PartialEq, Debug)]
pub struct SweepGradient {
    pub(crate) base: Gradient,
    t0: f32,
    t1: f32,
}

impl SweepGradient {
    /// Creates a new 2-point conical gradient shader.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(
        center: Point,
        start_angle: f32,
        end_angle: f32,
        stops: Vec<GradientStop>,
        mut mode: SpreadMode,
        transform: Transform,
    ) -> Option<Shader<'static>> {
        if !start_angle.is_finite() || !end_angle.is_finite() || start_angle > end_angle {
            return None;
        }

        match stops.as_slice() {
            [] => return None,
            [stop] => return Some(Shader::SolidColor(stop.color)),
            _ => (),
        }
        transform.invert()?;
        if start_angle.is_nearly_equal_within_tolerance(end_angle, DEGENERATE_THRESHOLD) {
            if mode == SpreadMode::Pad && end_angle > DEGENERATE_THRESHOLD {
                // In this case, the first color is repeated from 0 to the angle, then a hardstop
                // switches to the last color (all other colors are compressed to the infinitely
                // thin interpolation region).
                let front_color = stops.first().unwrap().color;
                let back_color = stops.last().unwrap().color;
                let mut new_stops = stops;
                new_stops.clear();
                new_stops.extend_from_slice(&[
                    GradientStop::new(0.0, front_color),
                    GradientStop::new(1.0, front_color),
                    GradientStop::new(1.0, back_color),
                ]);
                return SweepGradient::new(center, 0.0, end_angle, new_stops, mode, transform);
            }
            // TODO: Consider making a degenerate fallback shader similar to Skia. Tiny Skia
            // currently opts to return `None` in some places.
            return None;
        }
        if start_angle <= 0.0 && end_angle >= 360.0 {
            mode = SpreadMode::Pad;
        }
        let t0 = start_angle / 360.0;
        let t1 = end_angle / 360.0;
        Some(Shader::SweepGradient(SweepGradient {
            base: Gradient::new(
                stops,
                mode,
                transform,
                Transform::from_translate(-center.x, -center.y),
            ),
            t0,
            t1,
        }))
    }

    pub(crate) fn is_opaque(&self) -> bool {
        self.base.colors_are_opaque
    }

    pub(crate) fn push_stages(&self, cs: ColorSpace, p: &mut RasterPipelineBuilder) -> bool {
        let scale = 1.0 / (self.t1 - self.t0);
        let bias = -scale * self.t0;
        p.ctx.two_point_conical_gradient.p0 = scale;
        p.ctx.two_point_conical_gradient.p1 = bias;
        self.base.push_stages(
            p,
            cs,
            &|p| {
                p.push(Stage::XYToUnitAngle);
                if scale != 1.0 && bias != 0.0 {
                    p.push(Stage::ApplyConcentricScaleBias)
                }
            },
            &|_| {},
        )
    }
}
