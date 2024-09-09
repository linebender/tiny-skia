// Copyright 2006 The Android Open Source Project
// Copyright 2020 Yevhenii Reizner
// Copyright 2024 Jeremy James
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use alloc::vec::Vec;

use crate::pixmap::{Pixmap, PixmapRef};
use crate::PremultipliedColorU8;

#[cfg(all(not(feature = "std"), feature = "no-std-float"))]
use tiny_skia_path::NoStdFloat;

/// Mipmaps are used to scaling down source images quickly to be used instead
/// of a pixmap as source for bilinear or bicubic scaling
///
/// These are created from a `PixmapRef` as a base source which can be fetched
/// using level `0`
///
#[derive(Debug)]
pub struct Mipmaps<'a> {
    levels: Vec<Pixmap>,
    base_pixmap: PixmapRef<'a>,
}

impl<'a> Mipmaps<'a> {
    /// Allocates a new set of mipmaps from a base pixmap
    pub fn new(p: PixmapRef<'a>) -> Self {
        Mipmaps {
            levels: Vec::new(),
            base_pixmap: p,
        }
    }

    /// Fetch a mipmap to be used - or base pixmap if zero is given
    pub fn get(&self, level: usize) -> PixmapRef {
        return if level > 0 {
            self.levels.get(level - 1).unwrap().as_ref()
        } else {
            self.base_pixmap
        };
    }

    /// Ensure this many levels of mipmap are available, returning
    /// an index to be used with get()
    pub fn build(&mut self, required_levels: usize) -> usize {
        let mut src_level = self.levels.len();
        let mut src_pixmap = self.get(src_level);
        let mut level_width = src_pixmap.width();
        let mut level_height = src_pixmap.height();

        while src_level < required_levels {
            level_width = (level_width as f32 / 2.0).floor() as u32;
            level_height = (level_height as f32 / 2.0).floor() as u32;

            // Scale image down
            let mut dst_pixmap = Pixmap::new(level_width, level_height).unwrap();
            let dst_width = dst_pixmap.width() as usize;
            let dst_height = dst_pixmap.height() as usize;
            let dst_pixels = dst_pixmap.pixels_mut();

            let src_pixels = src_pixmap.pixels();
            let src_width = src_pixmap.width() as usize;
            let src_height = src_pixmap.height() as usize;

            //  To produce each mip level, we need to filter down by 1/2 (e.g. 100x100 -> 50,50)
            //  If the starting dimension is odd, we floor the size of the lower level (e.g. 101 -> 50)
            //  In those (odd) cases, we use a triangle filter, with 1-pixel overlap between samplings,
            //  else for even cases, we just use a 2x box filter.
            //
            //  This produces 4 possible isotropic filters: 2x2 2x3 3x2 3x3 where WxH indicates the number of
            //  src pixels we need to sample in each dimension to produce 1 dst pixel.
            let downsample = match (src_width & 1 == 0, src_height & 1 == 0) {
                (true, true) => downsample_2_2,
                (true, false) => downsample_2_3,
                (false, true) => downsample_3_2,
                (false, false) => downsample_3_3,
            };

            let mut src_y = 0;
            for dst_y in 0..dst_height {
                downsample(src_pixels, src_y, src_width, dst_pixels, dst_y, dst_width);
                src_y += 2;
            }

            self.levels.push(dst_pixmap);
            src_pixmap = self.levels.get(src_level).unwrap().as_ref();
            src_level += 1;
        }

        src_level
    }
}

/// Determine how many Mipmap levels will be needed for a given source and
/// a given (approximate) scaling being applied to the source
///
/// Return the number of levels, and a pre-scale that should be applied to
/// a transform that will 'correct' it to the right size of source
///
/// Note that this is different from Skia since only required levels will
/// be generated
pub fn compute_required_levels(
    base_pixmap: PixmapRef,
    scale_x: f32,
    scale_y: f32,
) -> (usize, f32, f32) {
    let mut required_levels: usize = 0;
    let mut level_width = base_pixmap.width();
    let mut level_height = base_pixmap.height();
    let mut prescale_x: f32 = 1.0;
    let mut prescale_y: f32 = 1.0;

    // Keep generating levels whilst required scale is
    // smaller than half of previous level size
    while scale_x * prescale_x < 0.5
        && level_width > 1
        && scale_y * prescale_y < 0.5
        && level_height > 1
    {
        required_levels += 1;
        level_width = (level_width as f32 / 2.0).floor() as u32;
        level_height = (level_height as f32 / 2.0).floor() as u32;
        prescale_x = base_pixmap.width() as f32 / level_width as f32;
        prescale_y = base_pixmap.height() as f32 / level_height as f32;
    }

    (required_levels, prescale_x, prescale_y)
}

// Downsamples to match Skia (non-SIMD)
macro_rules! sum_channel {
    ($channel:ident, $($p:ident),+ ) => {
        0u16 $( + $p.$channel() as u16 )+
    };
}

fn downsample_2_2(
    src_pixels: &[PremultipliedColorU8],
    src_y: usize,
    src_width: usize,
    dst_pixels: &mut [PremultipliedColorU8],
    dst_y: usize,
    dst_width: usize,
) {
    let mut src_x = 0;
    for dst_x in 0..dst_width {
        let p1 = src_pixels[src_y * src_width + src_x];
        let p2 = src_pixels[src_y * src_width + src_x + 1];
        let p3 = src_pixels[(src_y + 1) * src_width + src_x];
        let p4 = src_pixels[(src_y + 1) * src_width + src_x + 1];

        let r = (sum_channel!(red, p1, p2, p3, p4) >> 2) as u8;
        let g = (sum_channel!(green, p1, p2, p3, p4) >> 2) as u8;
        let b = (sum_channel!(blue, p1, p2, p3, p4) >> 2) as u8;
        let a = (sum_channel!(alpha, p1, p2, p3, p4) >> 2) as u8;
        dst_pixels[dst_y * dst_width + dst_x] =
            PremultipliedColorU8::from_rgba_unchecked(r, g, b, a);

        src_x += 2;
    }
}

fn downsample_2_3(
    src_pixels: &[PremultipliedColorU8],
    src_y: usize,
    src_width: usize,
    dst_pixels: &mut [PremultipliedColorU8],
    dst_y: usize,
    dst_width: usize,
) {
    // Given pixels:
    // a0 b0 c0 d0 ...
    // a1 b1 c1 d1 ...
    // a2 b2 c2 d2 ...
    // We want:
    // (a0 + 2*a1 + a2 + b0 + 2*b1 + b2) / 8
    // (c0 + 2*c1 + c2 + d0 + 2*d1 + d2) / 8
    // ...

    let mut src_x = 0;
    for dst_x in 0..dst_width {
        let p1 = src_pixels[src_y * src_width + src_x];
        let p2 = src_pixels[src_y * src_width + src_x + 1];
        let p3 = src_pixels[(src_y + 1) * src_width + src_x];
        let p4 = src_pixels[(src_y + 1) * src_width + src_x + 1];
        let p5 = src_pixels[(src_y + 2) * src_width + src_x];
        let p6 = src_pixels[(src_y + 2) * src_width + src_x + 1];

        let r = (sum_channel!(red, p1, p3, p3, p5, p2, p4, p4, p6) >> 3) as u8;
        let g = (sum_channel!(green, p1, p3, p3, p5, p2, p4, p4, p6) >> 3) as u8;
        let b = (sum_channel!(blue, p1, p3, p3, p5, p2, p4, p4, p6) >> 3) as u8;
        let a = (sum_channel!(alpha, p1, p3, p3, p5, p2, p4, p4, p6) >> 3) as u8;
        dst_pixels[dst_y * dst_width + dst_x] =
            PremultipliedColorU8::from_rgba_unchecked(r, g, b, a);

        src_x += 2;
    }
}

fn downsample_3_2(
    src_pixels: &[PremultipliedColorU8],
    src_y: usize,
    src_width: usize,
    dst_pixels: &mut [PremultipliedColorU8],
    dst_y: usize,
    dst_width: usize,
) {
    // Given pixels:
    // a0 b0 c0 d0 e0 ...
    // a1 b1 c1 d1 e1 ...
    // We want:
    // (a0 + 2*b0 + c0 + a1 + 2*b1 + c1) / 8
    // (c0 + 2*d0 + e0 + c1 + 2*d1 + e1) / 8
    // ...

    let mut src_x = 0;
    for dst_x in 0..dst_width {
        let p1 = src_pixels[src_y * src_width + src_x];
        let p2 = src_pixels[src_y * src_width + src_x + 1];
        let p3 = src_pixels[src_y * src_width + src_x + 2];
        let p4 = src_pixels[(src_y + 1) * src_width + src_x];
        let p5 = src_pixels[(src_y + 1) * src_width + src_x + 1];
        let p6 = src_pixels[(src_y + 1) * src_width + src_x + 2];

        let r = (sum_channel!(red, p1, p2, p2, p3, p4, p5, p5, p6) >> 3) as u8;
        let g = (sum_channel!(green, p1, p2, p2, p3, p4, p5, p5, p6) >> 3) as u8;
        let b = (sum_channel!(blue, p1, p2, p2, p3, p4, p5, p5, p6) >> 3) as u8;
        let a = (sum_channel!(alpha, p1, p2, p2, p3, p4, p5, p5, p6) >> 3) as u8;
        dst_pixels[dst_y * dst_width + dst_x] =
            PremultipliedColorU8::from_rgba_unchecked(r, g, b, a);

        src_x += 2;
    }
}

fn downsample_3_3(
    src_pixels: &[PremultipliedColorU8],
    src_y: usize,
    src_width: usize,
    dst_pixels: &mut [PremultipliedColorU8],
    dst_y: usize,
    dst_width: usize,
) {
    // Given pixels:
    // a0 b0 c0 d0 e0 ...
    // a1 b1 c1 d1 e1 ...
    // a2 b2 c2 d2 e2 ...
    // We want:
    // (a0 + 2*b0 + c0 + 2*a1 + 4*b1 + 2*c1 + a2 + 2*b2 + c2) / 16
    // (c0 + 2*d0 + e0 + 2*c1 + 4*d1 + 2*e1 + c2 + 2*d2 + e2) / 16
    // ...

    let mut src_x = 0;
    for dst_x in 0..dst_width {
        let p1 = src_pixels[src_y * src_width + src_x];
        let p2 = src_pixels[src_y * src_width + src_x + 1];
        let p3 = src_pixels[src_y * src_width + src_x + 2];
        let p4 = src_pixels[(src_y + 1) * src_width + src_x];
        let p5 = src_pixels[(src_y + 1) * src_width + src_x + 1];
        let p6 = src_pixels[(src_y + 1) * src_width + src_x + 2];
        let p7 = src_pixels[(src_y + 2) * src_width + src_x];
        let p8 = src_pixels[(src_y + 2) * src_width + src_x + 1];
        let p9 = src_pixels[(src_y + 2) * src_width + src_x + 2];

        let r = (sum_channel!(red, p1, p2, p2, p3, p4, p4, p5, p5, p5, p5, p6, p6, p7, p8, p8, p9)
            >> 4) as u8;
        let g =
            (sum_channel!(green, p1, p2, p2, p3, p4, p4, p5, p5, p5, p5, p6, p6, p7, p8, p8, p9)
                >> 4) as u8;
        let b = (sum_channel!(blue, p1, p2, p2, p3, p4, p4, p5, p5, p5, p5, p6, p6, p7, p8, p8, p9)
            >> 4) as u8;
        let a =
            (sum_channel!(alpha, p1, p2, p2, p3, p4, p4, p5, p5, p5, p5, p6, p6, p7, p8, p8, p9)
                >> 4) as u8;
        dst_pixels[dst_y * dst_width + dst_x] =
            PremultipliedColorU8::from_rgba_unchecked(r, g, b, a);

        src_x += 2;
    }
}
