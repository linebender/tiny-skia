// Copyright 2006 The Android Open Source Project
// Copyright 2020 Yevhenii Reizner
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use alloc::vec;
use alloc::vec::Vec;

use core::convert::TryFrom;
use core::num::NonZeroUsize;

use tiny_skia_path::IntSize;

use crate::{Color, IntRect};

#[cfg(feature = "16bpc")]
use crate::color::{ColorU16, PremultipliedColorU16, ALPHA_U16_OPAQUE, ALPHA_U16_TRANSPARENT};
use crate::color::{ColorU8, PremultipliedColor, PremultipliedColorU8};
use crate::color::{ALPHA_U8_OPAQUE, ALPHA_U8_TRANSPARENT};
use crate::geom::{IntSizeExt, ScreenIntRect};

#[cfg(feature = "png-format")]
use crate::color::premultiply_u8;

/// Number of bytes per pixel for 8bpc.
pub const BYTES_PER_PIXEL: usize = 4;

/// Sealed trait representing a pixel format with premultiplied color channels.
pub trait Pixel:
    Copy + Clone + PartialEq + Eq + bytemuck::Pod + bytemuck::Zeroable + 'static
{
    /// Demultiplied color type.
    type Color: Copy + Clone + PartialEq + Eq + core::fmt::Debug;
    /// Alpha channel type.
    type Alpha: Copy + Clone + Into<u32>;
    /// Bytes per pixel.
    const BYTES_PER_PIXEL: usize;
    /// Fully opaque alpha value.
    const ALPHA_OPAQUE: Self::Alpha;
    /// Fully transparent alpha value.
    const ALPHA_TRANSPARENT: Self::Alpha;

    /// Converts an RGBA float color into this premultiplied pixel type.
    fn from_color(c: Color) -> Self;
    /// Converts a floating-point premultiplied color into this pixel type.
    fn from_premultiplied(c: PremultipliedColor) -> Self;
    /// Returns demultiplied color.
    fn demultiply(self) -> Self::Color;
    /// Returns whether pixel is opaque.
    fn is_opaque(self) -> bool;
    /// Converts pixel to 8-bit premultiplied color.
    fn to_u8(&self) -> PremultipliedColorU8;
    /// Converts 8-bit premultiplied color to this pixel type.
    fn from_u8(c: PremultipliedColorU8) -> Self;
}

impl Pixel for PremultipliedColorU8 {
    type Color = ColorU8;
    type Alpha = u8;
    const BYTES_PER_PIXEL: usize = 4;
    const ALPHA_OPAQUE: u8 = ALPHA_U8_OPAQUE;
    const ALPHA_TRANSPARENT: u8 = ALPHA_U8_TRANSPARENT;

    #[inline(always)]
    fn from_color(c: Color) -> Self {
        c.premultiply().to_color_u8()
    }

    #[inline(always)]
    fn from_premultiplied(c: PremultipliedColor) -> Self {
        c.to_color_u8()
    }

    #[inline(always)]
    fn demultiply(self) -> Self::Color {
        PremultipliedColorU8::demultiply(&self)
    }

    #[inline(always)]
    fn is_opaque(self) -> bool {
        PremultipliedColorU8::is_opaque(&self)
    }

    #[inline(always)]
    fn to_u8(&self) -> PremultipliedColorU8 {
        *self
    }

    #[inline(always)]
    fn from_u8(c: PremultipliedColorU8) -> Self {
        c
    }
}

#[cfg(feature = "16bpc")]
impl Pixel for PremultipliedColorU16 {
    type Color = ColorU16;
    type Alpha = u16;
    const BYTES_PER_PIXEL: usize = 8;
    const ALPHA_OPAQUE: u16 = ALPHA_U16_OPAQUE;
    const ALPHA_TRANSPARENT: u16 = ALPHA_U16_TRANSPARENT;

    #[inline(always)]
    fn from_color(c: Color) -> Self {
        c.premultiply().to_color_u16()
    }

    #[inline(always)]
    fn from_premultiplied(c: PremultipliedColor) -> Self {
        c.to_color_u16()
    }

    #[inline(always)]
    fn demultiply(self) -> Self::Color {
        PremultipliedColorU16::demultiply(&self)
    }

    #[inline(always)]
    fn is_opaque(self) -> bool {
        PremultipliedColorU16::is_opaque(&self)
    }

    #[inline(always)]
    fn to_u8(&self) -> PremultipliedColorU8 {
        self.to_color_u8()
    }

    #[inline(always)]
    fn from_u8(c: PremultipliedColorU8) -> Self {
        PremultipliedColorU16::from_color_u8(c)
    }
}

/// A container that owns premultiplied RGBA pixels.
///
/// The data is not aligned, therefore width == stride.
#[derive(Clone, PartialEq)]
pub struct PixmapGeneric<P: Pixel> {
    data: Vec<u8>,
    size: IntSize,
    _marker: core::marker::PhantomData<P>,
}

/// 8-bit per channel RGBA Pixmap.
pub type Pixmap = PixmapGeneric<PremultipliedColorU8>;

/// 16-bit per channel RGBA Pixmap.
#[cfg(feature = "16bpc")]
pub type PixmapU16 = PixmapGeneric<PremultipliedColorU16>;

impl<P: Pixel> PixmapGeneric<P> {
    /// Allocates a new pixmap.
    ///
    /// A pixmap is filled with transparent black by default, aka (0, 0, 0, 0).
    ///
    /// Zero size in an error.
    pub fn new(width: u32, height: u32) -> Option<Self> {
        let size = IntSize::from_wh(width, height)?;
        let data_len = data_len_for_size::<P>(size)?;

        Some(PixmapGeneric {
            data: vec![0; data_len],
            size,
            _marker: core::marker::PhantomData,
        })
    }

    /// Creates a new pixmap by taking ownership over an image buffer.
    pub fn from_vec(data: Vec<u8>, size: IntSize) -> Option<Self> {
        let data_len = data_len_for_size::<P>(size)?;
        if data.len() != data_len {
            return None;
        }

        Some(PixmapGeneric {
            data,
            size,
            _marker: core::marker::PhantomData,
        })
    }

    /// Returns a container that references Pixmap's data.
    pub fn as_ref(&self) -> PixmapRefGeneric<'_, P> {
        PixmapRefGeneric {
            data: &self.data,
            size: self.size,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns a container that references Pixmap's mutable data.
    pub fn as_mut(&mut self) -> PixmapMutGeneric<'_, P> {
        PixmapMutGeneric {
            data: &mut self.data,
            size: self.size,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns pixmap's width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.size.width()
    }

    /// Returns pixmap's height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.size.height()
    }

    /// Returns pixmap's size.
    #[inline]
    pub fn size(&self) -> IntSize {
        self.size
    }

    /// Fills the entire pixmap with a specified color.
    pub fn fill(&mut self, color: Color) {
        let c = P::from_color(color);
        for p in self.as_mut().pixels_mut() {
            *p = c;
        }
    }

    /// Returns the internal data.
    pub fn data(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Returns the mutable internal data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.data.as_mut_slice()
    }

    /// Returns a pixel color.
    pub fn pixel(&self, x: u32, y: u32) -> Option<P> {
        let idx = self.width().checked_mul(y)?.checked_add(x)?;
        self.pixels().get(idx as usize).cloned()
    }

    /// Returns a mutable slice of pixels.
    pub fn pixels_mut(&mut self) -> &mut [P] {
        bytemuck::cast_slice_mut(self.data_mut())
    }

    /// Returns a slice of pixels.
    pub fn pixels(&self) -> &[P] {
        bytemuck::cast_slice(self.data())
    }

    /// Consumes the internal data.
    pub fn take(self) -> Vec<u8> {
        self.data
    }

    /// Returns a copy of the pixmap that intersects the `rect`.
    pub fn clone_rect(&self, rect: IntRect) -> Option<Self> {
        self.as_ref().clone_rect(rect)
    }
}

impl PixmapGeneric<PremultipliedColorU8> {
    /// Decodes a PNG data into an 8-bit `Pixmap`.
    #[cfg(feature = "png-format")]
    pub fn decode_png(data: &[u8]) -> Result<Self, png::DecodingError> {
        fn make_custom_png_error(msg: &str) -> png::DecodingError {
            std::io::Error::new(std::io::ErrorKind::Other, msg).into()
        }

        let mut decoder = png::Decoder::new(std::io::BufReader::new(std::io::Cursor::new(data)));
        decoder.set_transformations(png::Transformations::normalize_to_color8());
        let mut reader = decoder.read_info()?;
        let output_buffer_size = reader
            .output_buffer_size()
            .ok_or(png::DecodingError::LimitsExceeded)?;
        let mut img_data = vec![0; output_buffer_size];
        let info = reader.next_frame(&mut img_data)?;

        if info.bit_depth != png::BitDepth::Eight {
            return Err(make_custom_png_error("unsupported bit depth"));
        }

        let size = IntSize::from_wh(info.width, info.height)
            .ok_or_else(|| make_custom_png_error("invalid image size"))?;
        let data_len = data_len_for_size::<PremultipliedColorU8>(size)
            .ok_or_else(|| make_custom_png_error("image is too big"))?;

        img_data = match info.color_type {
            png::ColorType::Rgb => {
                let mut rgba_data = Vec::with_capacity(data_len);
                for rgb in img_data.chunks(3) {
                    rgba_data.push(rgb[0]);
                    rgba_data.push(rgb[1]);
                    rgba_data.push(rgb[2]);
                    rgba_data.push(ALPHA_U8_OPAQUE);
                }
                rgba_data
            }
            png::ColorType::Rgba => img_data,
            png::ColorType::Grayscale => {
                let mut rgba_data = Vec::with_capacity(data_len);
                for gray in img_data {
                    rgba_data.push(gray);
                    rgba_data.push(gray);
                    rgba_data.push(gray);
                    rgba_data.push(ALPHA_U8_OPAQUE);
                }
                rgba_data
            }
            png::ColorType::GrayscaleAlpha => {
                let mut rgba_data = Vec::with_capacity(data_len);
                for slice in img_data.chunks(2) {
                    let gray = slice[0];
                    let alpha = slice[1];
                    rgba_data.push(gray);
                    rgba_data.push(gray);
                    rgba_data.push(gray);
                    rgba_data.push(alpha);
                }
                rgba_data
            }
            png::ColorType::Indexed => {
                return Err(make_custom_png_error("indexed PNG is not supported"));
            }
        };

        for pixel in img_data.as_mut_slice().chunks_mut(BYTES_PER_PIXEL) {
            let a = pixel[3];
            pixel[0] = premultiply_u8(pixel[0], a);
            pixel[1] = premultiply_u8(pixel[1], a);
            pixel[2] = premultiply_u8(pixel[2], a);
        }

        Pixmap::from_vec(img_data, size)
            .ok_or_else(|| make_custom_png_error("failed to create a pixmap"))
    }

    /// Loads a PNG file into a `Pixmap`.
    #[cfg(feature = "png-format")]
    pub fn load_png<P: AsRef<std::path::Path>>(path: P) -> Result<Self, png::DecodingError> {
        let data = std::fs::read(path)?;
        Self::decode_png(&data)
    }

    /// Encodes pixmap into a PNG data.
    #[cfg(feature = "png-format")]
    pub fn encode_png(&self) -> Result<Vec<u8>, png::EncodingError> {
        self.as_ref().encode_png()
    }

    /// Saves pixmap as a PNG file.
    #[cfg(feature = "png-format")]
    pub fn save_png<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), png::EncodingError> {
        self.as_ref().save_png(path)
    }

    /// Consumes the pixmap and returns the internal data as demultiplied RGBA bytes.
    pub fn take_demultiplied(mut self) -> Vec<u8> {
        for pixel in self.pixels_mut() {
            let c = pixel.demultiply();
            *pixel =
                PremultipliedColorU8::from_rgba_unchecked(c.red(), c.green(), c.blue(), c.alpha());
        }
        self.data
    }
}

#[cfg(feature = "16bpc")]
impl PixmapGeneric<PremultipliedColorU16> {
    /// Consumes the pixmap and returns the internal data as demultiplied RGBA64 bytes.
    pub fn take_demultiplied(mut self) -> Vec<u8> {
        for pixel in self.pixels_mut() {
            let c = pixel.demultiply();
            *pixel =
                PremultipliedColorU16::from_rgba_unchecked(c.red(), c.green(), c.blue(), c.alpha());
        }
        self.data
    }

    /// Encodes pixmap into a 16-bit PNG data.
    #[cfg(feature = "png-format")]
    pub fn encode_png(&self) -> Result<Vec<u8>, png::EncodingError> {
        let demultiplied_data = self.clone().take_demultiplied();
        let mut data = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut data, self.width(), self.height());
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Sixteen);
            let mut writer = encoder.write_header()?;
            let be_bytes: Vec<u8> = demultiplied_data
                .chunks_exact(2)
                .flat_map(|chunk| {
                    let val = u16::from_ne_bytes([chunk[0], chunk[1]]);
                    val.to_be_bytes()
                })
                .collect();
            writer.write_image_data(&be_bytes)?;
        }
        Ok(data)
    }

    /// Saves pixmap as a 16-bit PNG file.
    #[cfg(feature = "png-format")]
    pub fn save_png<Pth: AsRef<std::path::Path>>(
        &self,
        path: Pth,
    ) -> Result<(), png::EncodingError> {
        let data = self.encode_png()?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Decodes a PNG image into a `PixmapU16`.
    #[cfg(feature = "png-format")]
    pub fn decode_png(data: &[u8]) -> Result<Self, png::DecodingError> {
        fn make_custom_png_error(msg: &str) -> png::DecodingError {
            std::io::Error::new(std::io::ErrorKind::Other, msg).into()
        }

        let decoder = png::Decoder::new(std::io::BufReader::new(std::io::Cursor::new(data)));
        let mut reader = decoder.read_info()?;
        let output_buffer_size = reader
            .output_buffer_size()
            .ok_or(png::DecodingError::LimitsExceeded)?;
        let mut img_data = vec![0; output_buffer_size];
        let info = reader.next_frame(&mut img_data)?;

        let size = IntSize::from_wh(info.width, info.height)
            .ok_or_else(|| make_custom_png_error("invalid image size"))?;
        let num_pixels = (info.width as usize) * (info.height as usize);
        let mut pixels = Vec::with_capacity(num_pixels);

        match (info.bit_depth, info.color_type) {
            (png::BitDepth::Sixteen, png::ColorType::Rgba) => {
                for chunk in img_data[..info.buffer_size()].chunks_exact(8) {
                    let r = u16::from_be_bytes([chunk[0], chunk[1]]);
                    let g = u16::from_be_bytes([chunk[2], chunk[3]]);
                    let b = u16::from_be_bytes([chunk[4], chunk[5]]);
                    let a = u16::from_be_bytes([chunk[6], chunk[7]]);
                    let c = ColorU16::from_rgba(r, g, b, a);
                    pixels.push(c.premultiply());
                }
            }
            (png::BitDepth::Sixteen, png::ColorType::Rgb) => {
                for chunk in img_data[..info.buffer_size()].chunks_exact(6) {
                    let r = u16::from_be_bytes([chunk[0], chunk[1]]);
                    let g = u16::from_be_bytes([chunk[2], chunk[3]]);
                    let b = u16::from_be_bytes([chunk[4], chunk[5]]);
                    let c = ColorU16::from_rgba(r, g, b, 65535);
                    pixels.push(c.premultiply());
                }
            }
            (png::BitDepth::Sixteen, png::ColorType::Grayscale) => {
                for chunk in img_data[..info.buffer_size()].chunks_exact(2) {
                    let gray = u16::from_be_bytes([chunk[0], chunk[1]]);
                    let c = ColorU16::from_rgba(gray, gray, gray, 65535);
                    pixels.push(c.premultiply());
                }
            }
            (png::BitDepth::Sixteen, png::ColorType::GrayscaleAlpha) => {
                for chunk in img_data[..info.buffer_size()].chunks_exact(4) {
                    let gray = u16::from_be_bytes([chunk[0], chunk[1]]);
                    let alpha = u16::from_be_bytes([chunk[2], chunk[3]]);
                    let c = ColorU16::from_rgba(gray, gray, gray, alpha);
                    pixels.push(c.premultiply());
                }
            }
            (png::BitDepth::Eight, png::ColorType::Rgba) => {
                for chunk in img_data[..info.buffer_size()].chunks_exact(4) {
                    let r = ((chunk[0] as u16) << 8) | (chunk[0] as u16);
                    let g = ((chunk[1] as u16) << 8) | (chunk[1] as u16);
                    let b = ((chunk[2] as u16) << 8) | (chunk[2] as u16);
                    let a = ((chunk[3] as u16) << 8) | (chunk[3] as u16);
                    let c = ColorU16::from_rgba(r, g, b, a);
                    pixels.push(c.premultiply());
                }
            }
            (png::BitDepth::Eight, png::ColorType::Rgb) => {
                for chunk in img_data[..info.buffer_size()].chunks_exact(3) {
                    let r = ((chunk[0] as u16) << 8) | (chunk[0] as u16);
                    let g = ((chunk[1] as u16) << 8) | (chunk[1] as u16);
                    let b = ((chunk[2] as u16) << 8) | (chunk[2] as u16);
                    let c = ColorU16::from_rgba(r, g, b, 65535);
                    pixels.push(c.premultiply());
                }
            }
            (png::BitDepth::Eight, png::ColorType::Grayscale) => {
                for &gray8 in &img_data[..info.buffer_size()] {
                    let gray = ((gray8 as u16) << 8) | (gray8 as u16);
                    let c = ColorU16::from_rgba(gray, gray, gray, 65535);
                    pixels.push(c.premultiply());
                }
            }
            (png::BitDepth::Eight, png::ColorType::GrayscaleAlpha) => {
                for chunk in img_data[..info.buffer_size()].chunks_exact(2) {
                    let gray = ((chunk[0] as u16) << 8) | (chunk[0] as u16);
                    let alpha = ((chunk[1] as u16) << 8) | (chunk[1] as u16);
                    let c = ColorU16::from_rgba(gray, gray, gray, alpha);
                    pixels.push(c.premultiply());
                }
            }
            _ => {
                return Err(make_custom_png_error("unsupported PNG format"));
            }
        }

        let raw_bytes: Vec<u8> = bytemuck::cast_slice(&pixels).to_vec();
        PixmapGeneric::from_vec(raw_bytes, size)
            .ok_or_else(|| make_custom_png_error("failed to create pixmap"))
    }
}

impl<P: Pixel> core::fmt::Debug for PixmapGeneric<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Pixmap")
            .field("data", &"...")
            .field("width", &self.size.width())
            .field("height", &self.size.height())
            .finish()
    }
}

/// A container that references premultiplied RGBA pixels.
#[derive(Clone, Copy, PartialEq)]
pub struct PixmapRefGeneric<'a, P: Pixel> {
    data: &'a [u8],
    size: IntSize,
    _marker: core::marker::PhantomData<P>,
}

/// 8-bit per channel Pixmap reference.
pub type PixmapRef<'a> = PixmapRefGeneric<'a, PremultipliedColorU8>;

/// 16-bit per channel Pixmap reference.
#[cfg(feature = "16bpc")]
pub type PixmapU16Ref<'a> = PixmapRefGeneric<'a, PremultipliedColorU16>;

impl<'a, P: Pixel> PixmapRefGeneric<'a, P> {
    /// Creates a new `PixmapRef` from bytes.
    pub fn from_bytes(data: &'a [u8], width: u32, height: u32) -> Option<Self> {
        let size = IntSize::from_wh(width, height)?;
        let data_len = data_len_for_size::<P>(size)?;
        if data.len() < data_len {
            return None;
        }

        Some(PixmapRefGeneric {
            data,
            size,
            _marker: core::marker::PhantomData,
        })
    }

    /// Creates a new `Pixmap` from the current data.
    pub fn to_owned(&self) -> PixmapGeneric<P> {
        PixmapGeneric {
            data: self.data.to_vec(),
            size: self.size,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns pixmap's width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.size.width()
    }

    /// Returns pixmap's height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.size.height()
    }

    /// Returns pixmap's size.
    pub(crate) fn size(&self) -> IntSize {
        self.size
    }

    /// Returns pixmap's rect.
    pub(crate) fn rect(&self) -> ScreenIntRect {
        self.size.to_screen_int_rect(0, 0)
    }

    /// Returns the internal data.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Returns a pixel color.
    pub fn pixel(&self, x: u32, y: u32) -> Option<P> {
        let idx = self.width().checked_mul(y)?.checked_add(x)?;
        self.pixels().get(idx as usize).cloned()
    }

    /// Returns a slice of pixels.
    pub fn pixels(&self) -> &'a [P] {
        bytemuck::cast_slice(self.data())
    }

    /// Returns a copy of the pixmap that intersects the `rect`.
    pub fn clone_rect(&self, rect: IntRect) -> Option<PixmapGeneric<P>> {
        let rect = self.rect().to_int_rect().intersect(&rect)?;
        let mut new = PixmapGeneric::<P>::new(rect.width(), rect.height())?;
        {
            let old_pixels = self.pixels();
            let mut new_mut = new.as_mut();
            let new_pixels = new_mut.pixels_mut();

            for y in 0..rect.height() {
                for x in 0..rect.width() {
                    let old_idx = (y + rect.y() as u32) * self.width() + (x + rect.x() as u32);
                    let new_idx = y * rect.width() + x;
                    new_pixels[new_idx as usize] = old_pixels[old_idx as usize];
                }
            }
        }

        Some(new)
    }
}

impl PixmapRefGeneric<'_, PremultipliedColorU8> {
    /// Encodes pixmap into a PNG data.
    #[cfg(feature = "png-format")]
    pub fn encode_png(&self) -> Result<Vec<u8>, png::EncodingError> {
        let demultiplied_data = self.to_owned().take_demultiplied();
        let mut data = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut data, self.width(), self.height());
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_image_data(&demultiplied_data)?;
        }
        Ok(data)
    }

    /// Saves pixmap as a PNG file.
    #[cfg(feature = "png-format")]
    pub fn save_png<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), png::EncodingError> {
        let data = self.encode_png()?;
        std::fs::write(path, data)?;
        Ok(())
    }
}

impl<P: Pixel> core::fmt::Debug for PixmapRefGeneric<'_, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PixmapRef")
            .field("data", &"...")
            .field("width", &self.size.width())
            .field("height", &self.size.height())
            .finish()
    }
}

/// A container that references mutable premultiplied RGBA pixels.
#[derive(PartialEq)]
pub struct PixmapMutGeneric<'a, P: Pixel> {
    data: &'a mut [u8],
    size: IntSize,
    _marker: core::marker::PhantomData<P>,
}

/// 8-bit per channel Pixmap mutable reference.
pub type PixmapMut<'a> = PixmapMutGeneric<'a, PremultipliedColorU8>;

/// 16-bit per channel Pixmap mutable reference.
#[cfg(feature = "16bpc")]
pub type PixmapU16Mut<'a> = PixmapMutGeneric<'a, PremultipliedColorU16>;

impl<'a, P: Pixel> PixmapMutGeneric<'a, P> {
    /// Creates a new `PixmapMut` from bytes.
    pub fn from_bytes(data: &'a mut [u8], width: u32, height: u32) -> Option<Self> {
        let size = IntSize::from_wh(width, height)?;
        let data_len = data_len_for_size::<P>(size)?;
        if data.len() < data_len {
            return None;
        }

        Some(PixmapMutGeneric {
            data,
            size,
            _marker: core::marker::PhantomData,
        })
    }

    /// Creates a new `Pixmap` from the current data.
    pub fn to_owned(&self) -> PixmapGeneric<P> {
        PixmapGeneric {
            data: self.data.to_vec(),
            size: self.size,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns a container that references Pixmap's data.
    pub fn as_ref(&self) -> PixmapRefGeneric<'_, P> {
        PixmapRefGeneric {
            data: self.data,
            size: self.size,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns pixmap's width.
    #[inline]
    pub fn width(&self) -> u32 {
        self.size.width()
    }

    /// Returns pixmap's height.
    #[inline]
    pub fn height(&self) -> u32 {
        self.size.height()
    }

    /// Returns pixmap's size.
    pub(crate) fn size(&self) -> IntSize {
        self.size
    }

    /// Fills the entire pixmap with a specified color.
    pub fn fill(&mut self, color: Color) {
        let c = P::from_color(color);
        for p in self.pixels_mut() {
            *p = c;
        }
    }

    /// Returns the mutable internal data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// Returns a mutable slice of pixels.
    pub fn pixels_mut(&mut self) -> &mut [P] {
        bytemuck::cast_slice_mut(self.data_mut())
    }

    /// Creates `SubPixmapMut` that contains the whole `PixmapMut`.
    pub(crate) fn as_subpixmap(&mut self) -> SubPixmapMutGeneric<'_, P> {
        let width = self.width() as usize;
        SubPixmapMutGeneric {
            size: self.size(),
            real_width: width,
            data: self.data,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns a mutable reference to the pixmap region that intersects the `rect`.
    pub(crate) fn subpixmap(&mut self, rect: IntRect) -> Option<SubPixmapMutGeneric<'_, P>> {
        let rect = self.size.to_int_rect(0, 0).intersect(&rect)?;
        let row_bytes = self.width() as usize * P::BYTES_PER_PIXEL;
        let offset = rect.top() as usize * row_bytes + rect.left() as usize * P::BYTES_PER_PIXEL;

        Some(SubPixmapMutGeneric {
            size: rect.size(),
            real_width: self.width() as usize,
            data: &mut self.data[offset..],
            _marker: core::marker::PhantomData,
        })
    }
}

impl<P: Pixel> core::fmt::Debug for PixmapMutGeneric<'_, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PixmapMut")
            .field("data", &"...")
            .field("width", &self.size.width())
            .field("height", &self.size.height())
            .finish()
    }
}

/// A `PixmapMut` subregion.
pub struct SubPixmapMutGeneric<'a, P: Pixel> {
    /// Pixmap data.
    pub data: &'a mut [u8],
    /// Region size.
    pub size: IntSize,
    /// Width of the parent Pixmap.
    pub real_width: usize,
    /// Pixel type marker.
    pub _marker: core::marker::PhantomData<P>,
}

impl<P: Pixel> core::fmt::Debug for SubPixmapMutGeneric<'_, P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubPixmapMut")
            .field("data", &"...")
            .field("width", &self.size.width())
            .field("height", &self.size.height())
            .finish()
    }
}

/// 8-bit per channel SubPixmap mutable reference.
pub type SubPixmapMut<'a> = SubPixmapMutGeneric<'a, PremultipliedColorU8>;

/// 16-bit per channel SubPixmap mutable reference.
#[cfg(feature = "16bpc")]
pub type SubPixmapU16Mut<'a> = SubPixmapMutGeneric<'a, PremultipliedColorU16>;

impl<'a, P: Pixel> SubPixmapMutGeneric<'a, P> {
    /// Returns a mutable slice of pixels.
    pub fn pixels_mut(&mut self) -> &mut [P] {
        bytemuck::cast_slice_mut(self.data)
    }
}

/// Returns minimum bytes per row as usize.
fn min_row_bytes<P: Pixel>(size: IntSize) -> Option<NonZeroUsize> {
    let w = i32::try_from(size.width()).ok()?;
    let w = w.checked_mul(P::BYTES_PER_PIXEL as i32)?;
    NonZeroUsize::new(w as usize)
}

/// Returns storage size required by pixel array.
fn compute_data_len<P: Pixel>(size: IntSize, row_bytes: usize) -> Option<usize> {
    let h = size.height().checked_sub(1)?;
    let h = (h as usize).checked_mul(row_bytes)?;
    let w = (size.width() as usize).checked_mul(P::BYTES_PER_PIXEL)?;
    h.checked_add(w)
}

fn data_len_for_size<P: Pixel>(size: IntSize) -> Option<usize> {
    let row_bytes = min_row_bytes::<P>(size)?;
    compute_data_len::<P>(size, row_bytes.get())
}

/// A dynamically-typed pixmap holding either 8bpc (RGBA8) or 16bpc (RGBA16) pixel data.
#[cfg(feature = "16bpc")]
#[derive(Clone, PartialEq, Debug)]
pub enum DynamicPixmap {
    /// 8-bit per channel pixmap.
    U8(Pixmap),
    /// 16-bit per channel pixmap.
    U16(PixmapU16),
}

#[cfg(feature = "16bpc")]
impl DynamicPixmap {
    /// Returns the width of the pixmap.
    pub fn width(&self) -> u32 {
        match self {
            DynamicPixmap::U8(p) => p.width(),
            DynamicPixmap::U16(p) => p.width(),
        }
    }

    /// Returns the height of the pixmap.
    pub fn height(&self) -> u32 {
        match self {
            DynamicPixmap::U8(p) => p.height(),
            DynamicPixmap::U16(p) => p.height(),
        }
    }

    /// Returns a dynamic mutable reference.
    pub fn as_mut(&mut self) -> DynamicPixmapMut<'_> {
        match self {
            DynamicPixmap::U8(p) => DynamicPixmapMut::U8(p.as_mut()),
            DynamicPixmap::U16(p) => DynamicPixmapMut::U16(p.as_mut()),
        }
    }

    /// Encodes pixmap into PNG data.
    #[cfg(feature = "png-format")]
    pub fn encode_png(&self) -> Result<Vec<u8>, png::EncodingError> {
        match self {
            DynamicPixmap::U8(p) => p.encode_png(),
            DynamicPixmap::U16(p) => p.encode_png(),
        }
    }

    /// Saves pixmap as a PNG file.
    #[cfg(feature = "png-format")]
    pub fn save_png<Pth: AsRef<std::path::Path>>(
        &self,
        path: Pth,
    ) -> Result<(), png::EncodingError> {
        match self {
            DynamicPixmap::U8(p) => p.save_png(path),
            DynamicPixmap::U16(p) => p.save_png(path),
        }
    }
}

/// A dynamically-typed mutable pixmap reference.
#[cfg(feature = "16bpc")]
#[derive(PartialEq, Debug)]
pub enum DynamicPixmapMut<'a> {
    /// 8-bit per channel pixmap mutable reference.
    U8(PixmapMut<'a>),
    /// 16-bit per channel pixmap mutable reference.
    U16(PixmapU16Mut<'a>),
}

#[cfg(feature = "16bpc")]
impl DynamicPixmapMut<'_> {
    /// Returns the width of the pixmap.
    pub fn width(&self) -> u32 {
        match self {
            DynamicPixmapMut::U8(p) => p.width(),
            DynamicPixmapMut::U16(p) => p.width(),
        }
    }

    /// Returns the height of the pixmap.
    pub fn height(&self) -> u32 {
        match self {
            DynamicPixmapMut::U8(p) => p.height(),
            DynamicPixmapMut::U16(p) => p.height(),
        }
    }
}

/// A dynamically-typed immutable pixmap reference.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum DynamicPixmapRef<'a> {
    /// 8-bit per channel pixmap reference.
    U8(PixmapRef<'a>),
    /// 16-bit per channel pixmap reference.
    #[cfg(feature = "16bpc")]
    U16(PixmapU16Ref<'a>),
}

impl<'a> DynamicPixmapRef<'a> {
    /// Creates a dummy 1x1 8bpc pixmap reference.
    pub fn dummy() -> Self {
        DynamicPixmapRef::U8(PixmapRef::from_bytes(&[0, 0, 0, 0], 1, 1).unwrap())
    }

    /// Returns the width of the pixmap.
    pub fn width(&self) -> u32 {
        match self {
            DynamicPixmapRef::U8(p) => p.width(),
            #[cfg(feature = "16bpc")]
            DynamicPixmapRef::U16(p) => p.width(),
        }
    }

    /// Returns the height of the pixmap.
    pub fn height(&self) -> u32 {
        match self {
            DynamicPixmapRef::U8(p) => p.height(),
            #[cfg(feature = "16bpc")]
            DynamicPixmapRef::U16(p) => p.height(),
        }
    }
}
