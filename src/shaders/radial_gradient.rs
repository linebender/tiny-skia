// Copyright 2006 The Android Open Source Project
// Copyright 2020 Yevhenii Reizner
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use alloc::vec::Vec;

use crate::{ConicalGradient, GradientStop, Point, Shader, SpreadMode, Transform};

#[cfg(all(not(feature = "std"), feature = "no-std-float"))]
use tiny_skia_path::NoStdFloat;


/// A radial gradient shader.
///
/// This is not `SkRadialGradient` like in Skia, but rather `SkTwoPointConicalGradient`
/// without the start radius.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct RadialGradient;

impl RadialGradient {
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
    pub fn new(
        start: Point,
        end: Point,
        radius: f32,
        stops: Vec<GradientStop>,
        mode: SpreadMode,
        transform: Transform,
    ) -> Option<Shader<'static>> {
        ConicalGradient::new(start, 0.0, end, radius, stops, mode, transform)
    }
}
