// Copyright 2006 The Android Open Source Project
// Copyright 2020 Evgeniy Reizner
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// NOTE: this is not SkPathBuilder, but rather a reimplementation of SkPath.

use crate::{Point, Bounds, Path, Rect};

use crate::checked_geom_ext::BoundsExt;
use crate::geometry;
use crate::path::PathVerb;
use crate::scalar::{Scalar, SCALAR_ROOT_2_OVER_2};


#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PathDirection {
    /// Clockwise direction for adding closed contours.
    CW,
    /// Counter-clockwise direction for adding closed contours.
    CCW,
}


/// A path builder.
#[derive(Default, Debug)]
pub struct PathBuilder {
    pub(crate) verbs: Vec<PathVerb>,
    pub(crate) points: Vec<Point>,
    pub(crate) last_move_to_index: usize,
    pub(crate) move_to_required: bool,
}

impl PathBuilder {
    /// Creates a new builder.
    #[inline]
    pub fn new() -> Self {
        PathBuilder {
            verbs: Vec::new(),
            points: Vec::new(),
            last_move_to_index: 0,
            move_to_required: true,
        }
    }

    /// Creates a new builder with a specified capacity.
    ///
    /// Number of points depends on a verb type:
    ///
    /// - Move - 1
    /// - Line - 1
    /// - Quad - 2
    /// - Cubic - 3
    /// - Close - 0
    #[inline]
    pub fn with_capacity(verbs_capacity: usize, points_capacity: usize) -> Self {
        PathBuilder {
            verbs: Vec::with_capacity(verbs_capacity),
            points: Vec::with_capacity(points_capacity),
            last_move_to_index: 0,
            move_to_required: true,
        }
    }

    /// Creates a new `Path` from `Bounds`.
    ///
    /// Never fails since `Bounds` is always valid.
    ///
    /// Segments are created clockwise: TopLeft -> TopRight -> BottomRight -> BottomLeft
    ///
    /// The contour is closed.
    #[inline]
    pub fn from_bound(bounds: Bounds) -> Path {
        let verbs = vec![
            PathVerb::Move,
            PathVerb::Line,
            PathVerb::Line,
            PathVerb::Line,
            PathVerb::Close,
        ];

        let points = vec![
            Point::from_xy(bounds.left(), bounds.top()),
            Point::from_xy(bounds.right(), bounds.top()),
            Point::from_xy(bounds.right(), bounds.bottom()),
            Point::from_xy(bounds.left(), bounds.bottom()),
        ];

        Path {
            bounds,
            verbs,
            points,
        }
    }

    /// Returns the current number of segments in the builder.
    pub fn reserve(&mut self, additional_verbs: usize, additional_points: usize) {
        self.verbs.reserve(additional_verbs);
        self.points.reserve(additional_points);
    }

    /// Returns the current number of segments in the builder.
    pub fn len(&self) -> usize {
        self.verbs.len()
    }

    /// Checks if the builder has any segments added.
    pub fn is_empty(&self) -> bool {
        self.verbs.is_empty()
    }

    /// Adds beginning of a contour.
    ///
    /// Multiple continuous MoveTo segments are not allowed.
    /// If the previous segment was also MoveTo, it will be overwritten with the current one.
    pub fn move_to(&mut self, x: f32, y: f32) {
        if let Some(PathVerb::Move) = self.verbs.last() {
            let last_idx = self.points.len() - 1;
            self.points[last_idx] = Point::from_xy(x, y);
        } else {
            self.last_move_to_index = self.points.len();
            self.move_to_required = false;

            self.verbs.push(PathVerb::Move);
            self.points.push(Point::from_xy(x, y));
        }
    }

    #[inline(never)]
    fn inject_move_to_if_needed(&mut self) {
        if self.move_to_required {
            match self.points.get(self.last_move_to_index).cloned() {
                Some(p) => self.move_to(p.x, p.y),
                None => self.move_to(0.0, 0.0),
            }
        }
    }

    /// Adds a line from the last point.
    ///
    /// - If Path is empty - adds Move(0, 0) first.
    /// - If Path ends with Close - adds Move(last_x, last_y) first.
    pub fn line_to(&mut self, x: f32, y: f32) {
        self.inject_move_to_if_needed();

        self.verbs.push(PathVerb::Line);
        self.points.push(Point::from_xy(x, y));
    }

    /// Adds a quad curve from the last point to `x`, `y`.
    ///
    /// - If Path is empty - adds Move(0, 0) first.
    /// - If Path ends with Close - adds Move(last_x, last_y) first.
    pub fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.inject_move_to_if_needed();

        self.verbs.push(PathVerb::Quad);
        self.points.push(Point::from_xy(x1, y1));
        self.points.push(Point::from_xy(x, y));
    }

    // We do not support conic segments, but Skia still relies on them from time to time.
    // This method will simply convert the input data into quad segments.
    pub(crate) fn conic_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, weight: f32) {
        self.inject_move_to_if_needed();

        // TODO: use SkAutoConicToQuads

        let last = self.last_point().unwrap();
        let conic = geometry::Conic::new(last, Point::from_xy(x1, y1), Point::from_xy(x2, y2), weight);
        let mut points = [Point::zero(); 32]; // 1 + 2 * (1<<3) = 17, so 32 should be enough
        let count = conic.chop_into_quads_pow2(3, &mut points);

        // Points are ordered as: 0 - 1 2 - 3 4 - 5 6 - ..
        // `count` is a number of pairs +1
        let mut offset = 1;
        for _ in 0..count {
            let pt1 = points[offset + 0];
            let pt2 = points[offset + 1];
            self.quad_to(pt1.x, pt1.y, pt2.x, pt2.y);
            offset += 2;
        }
    }

    pub(crate) fn conic_points_to(&mut self, pt1: Point, pt2: Point, weight: f32) {
        self.conic_to(pt1.x, pt1.y, pt2.x, pt2.y, weight);
    }

    /// Adds a cubic curve from the last point to `x`, `y`.
    ///
    /// - If Path is empty - adds Move(0, 0) first.
    /// - If Path ends with Close - adds Move(last_x, last_y) first.
    pub fn cubic_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.inject_move_to_if_needed();

        self.verbs.push(PathVerb::Cubic);
        self.points.push(Point::from_xy(x1, y1));
        self.points.push(Point::from_xy(x2, y2));
        self.points.push(Point::from_xy(x, y));
    }

    /// Closes the current contour.
    ///
    /// A closed contour connects the first and the last Point
    /// with a line, forming a continuous loop.
    ///
    /// Does nothing when Path is empty or already closed.
    ///
    /// Open and closed contour will be filled the same way.
    /// Stroking an open contour will add LineCap at contour's start and end.
    /// Stroking an closed contour will add LineJoin at contour's start and end.
    pub fn close(&mut self) {
        // don't add a close if it's the first verb or a repeat
        if !self.verbs.is_empty() {
            if self.verbs.last().cloned() != Some(PathVerb::Close) {
                self.verbs.push(PathVerb::Close);
            }
        }

        self.move_to_required = true;
    }

    #[inline]
    pub(crate) fn last_point(&self) -> Option<Point> {
        self.points.last().cloned()
    }

    #[inline]
    pub(crate) fn set_last_point(&mut self, pt: Point) {
        match self.points.last_mut() {
            Some(last) => *last = pt,
            None => self.move_to(pt.x, pt.y),
        }
    }

    pub(crate) fn is_zero_length_since_point(&self, start_pt_index: usize) -> bool {
        let count = self.points.len() - start_pt_index;
        if count < 2 {
            return true;
        }

        let first = self.points[start_pt_index];
        for i in 1..count {
            if first != self.points[start_pt_index + i] {
                return false;
            }
        }

        true
    }

    fn push_oval(&mut self, oval: &Rect) {
        let cx = oval.left().half() + oval.right().half();
        let cy = oval.top().half() + oval.bottom().half();

        let oval_points = [
            Point::from_xy(cx, oval.bottom()),
            Point::from_xy(oval.left(), cy),
            Point::from_xy(cx, oval.top()),
            Point::from_xy(oval.right(), cy),
        ];

        let rect_points = [
            Point::from_xy(oval.right(), oval.bottom()),
            Point::from_xy(oval.left(), oval.bottom()),
            Point::from_xy(oval.left(), oval.top()),
            Point::from_xy(oval.right(), oval.top()),
        ];

        let weight = SCALAR_ROOT_2_OVER_2;
        self.move_to(oval_points[3].x, oval_points[3].y);
        for (p1, p2) in rect_points.iter().zip(oval_points.iter()) {
            self.conic_points_to(*p1, *p2, weight);
        }
        self.close();
    }

    #[inline]
    pub(crate) fn push_circle(&mut self, x: f32, y: f32, r: f32) {
        if let Some(r) = Rect::from_xywh(x - r, y - r, r + r, r + r) {
            self.push_oval(&r);
        }
    }

    pub(crate) fn push_path(&mut self, other: &PathBuilder) {
        if other.is_empty() {
            return;
        }

        if self.last_move_to_index != 0 {
            self.last_move_to_index = self.points.len() + other.last_move_to_index;
        }

        self.verbs.extend_from_slice(&other.verbs);
        self.points.extend_from_slice(&other.points);
    }

    /// Appends, in a reverse order, the first contour of path ignoring path's last point.
    pub(crate) fn reverse_path_to(&mut self, other: &PathBuilder) {
        if other.is_empty() {
            return;
        }

        debug_assert_eq!(other.verbs[0], PathVerb::Move);

        let mut points_offset = other.points.len() - 1;
        for verb in other.verbs.iter().rev() {
            match verb {
                PathVerb::Move => {
                    // if the path has multiple contours, stop after reversing the last
                    break;
                }
                PathVerb::Line => {
                    // We're moving one point back manually, to prevent points_offset overflow.
                    let pt = other.points[points_offset - 1];
                    points_offset -= 1;
                    self.line_to(pt.x, pt.y);
                }
                PathVerb::Quad => {
                    let pt1 = other.points[points_offset - 1];
                    let pt2 = other.points[points_offset - 2];
                    points_offset -= 2;
                    self.quad_to(pt1.x, pt1.y, pt2.x, pt2.y);
                }
                PathVerb::Cubic => {
                    let pt1 = other.points[points_offset - 1];
                    let pt2 = other.points[points_offset - 2];
                    let pt3 = other.points[points_offset - 3];
                    points_offset -= 3;
                    self.cubic_to(pt1.x, pt1.y, pt2.x, pt2.y, pt3.x, pt3.y);
                }
                PathVerb::Close => {}
            }
        }
    }

    /// Reset the builder.
    ///
    /// Memory is not deallocated.
    pub fn clear(&mut self) {
        self.verbs.clear();
        self.points.clear();
        self.last_move_to_index = 0;
        self.move_to_required = true;
    }

    /// Finishes the builder and returns a `Path`.
    ///
    /// Returns `None` when `Path` is empty or has zero bounds.
    pub fn finish(self) -> Option<Path> {
        if self.is_empty() {
            return None;
        }

        // Just a move to? Bail.
        if self.verbs.len() == 1 {
            return None;
        }

        let bounds = Bounds::from_points(&self.points)?;

        Some(Path {
            bounds,
            verbs: self.verbs,
            points: self.points,
        })
    }
}
