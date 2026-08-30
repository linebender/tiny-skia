// Copyright 2006 The Android Open Source Project
// Copyright 2020 Yevhenii Reizner
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use core::convert::TryFrom;
use core::num::NonZeroUsize;

use tiny_skia_path::IntSize;

use crate::{Color, IntRect, PremultipliedColor};

use crate::color::PremultipliedColorU8;

#[cfg(feature = "png-format")]
use crate::color::{premultiply_u8, ALPHA_U8_OPAQUE};

/// This determines how pixels are encoded in memory and map to real values.
///
/// This does NOT include color space information.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub enum PixelType {
    /// 8 bit premultiplied integer values in R,G,B,A memory order.
    ///
    /// See [PremultipliedColorU8]
    ///
    /// Unit white is (255,255,255,255).
    Rgba8U,
}

impl PixelType {
    /// Number of bytes per pixel
    #[inline(always)]
    pub const fn size(&self) -> u8 {
        match self {
            PixelType::Rgba8U => 4,
        }
    }

    /// Required memory alignment of each pixel
    #[inline(always)]
    pub const fn alignment(&self) -> u8 {
        match self {
            PixelType::Rgba8U => 1,
        }
    }
}

/// Utility type for an owned buffer which may have a particular alignment
/// constraint, used by [Pixmap].
///
/// [PixelType]s with alignment requirements greater than 1 cannot be stored
/// directly in a [`Box<[u8]>`], because Rust's allocation rules do not guarantee
/// that [`Box<[u8]>`]'s data is aligned to a multiple of any value other than 1,
/// and allocator implementations (for example, a bump allocator sharded by
/// alignment requirement) could indeed produce [`Box<[u8]>`]s with memory address
/// value ≡ 1 mod 2.
///
/// Because strides need not be a multiple of the pixel size, transmuting the
/// entire contents to a slice of pixels will not work unless you know the Pixmap
/// that produced this was tightly packed (which [Pixmap::new] does by default).
///
/// If a new [PixelType] with a higher alignment requirement is added, this enum
/// will gain an option like `Align2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignedMemory {
    /// Used for [PixelType::Rgba8U]
    Align1(Box<[u8]>),
}

/// A container that owns premultiplied RGBA pixels.
///
/// The data is only guaranteed to be aligned to match the PixelType.
#[derive(Clone, PartialEq)]
pub struct Pixmap {
    data: AlignedMemory,
    size: IntSize,
    /// The number of bytes between the starts of two rows
    stride: usize,
    pixel_type: PixelType,
}

impl Pixmap {
    /// Allocates a new pixmap.
    ///
    /// A pixmap is filled with transparent black by default, aka (0, 0, 0, 0).
    ///
    /// Zero size in an error.
    ///
    /// Pixmap's width is limited by i32::MAX/4.
    pub fn new(width: u32, height: u32) -> Option<Self> {
        let size = IntSize::from_wh(width, height)?;
        let stride = min_row_bytes(size, PixelType::Rgba8U)?;
        let data_len = data_len_for_size(size, stride.get())?;

        // We cannot check that allocation was successful yet.
        // We have to wait for https://github.com/rust-lang/rust/issues/48043

        let data = AlignedMemory::Align1(vec![0; data_len].into_boxed_slice());

        Some(Pixmap {
            data,
            size,
            stride: stride.get(),
            pixel_type: PixelType::Rgba8U,
        })
    }

    /// Allocates a new pixmap with the provided parameters.
    ///
    /// A pixmap is filled with transparent black by default.
    ///
    /// Zero size in an error.
    ///
    /// Pixmap's width is limited by i32::MAX/pix_type.size().
    pub fn new_with_type(
        width: u32,
        height: u32,
        stride: usize,
        pixel_type: PixelType,
    ) -> Option<Self> {
        let size = IntSize::from_wh(width, height)?;
        let min_stride = min_row_bytes(size, pixel_type)?;
        if min_stride.get() > stride {
            return None;
        }
        let data_len = data_len_for_size(size, stride)?;

        let data = AlignedMemory::Align1(vec![0; data_len].into_boxed_slice());
        Some(Pixmap {
            data,
            size,
            stride,
            pixel_type,
        })
    }

    /// Allocates a new pixmap with the provided parameters.
    ///
    /// A pixmap is filled with transparent black by default.
    ///
    /// Zero size in an error.
    ///
    /// Pixmap's width is limited by i32::MAX/pix_type.size().
    pub fn from_mem_with_type(
        data: AlignedMemory,
        width: u32,
        height: u32,
        stride: usize,
        pixel_type: PixelType,
    ) -> Option<Self> {
        let size = IntSize::from_wh(width, height)?;
        let min_stride = min_row_bytes(size, pixel_type)?;
        if min_stride.get() > stride {
            return None;
        }
        let data_len = data_len_for_size(size, stride)?;

        let AlignedMemory::Align1(buf) = &data;

        if buf.len() != data_len {
            return None;
        }

        Some(Pixmap {
            data,
            size,
            stride,
            pixel_type,
        })
    }

    /// Creates a new pixmap by taking ownership over an image buffer
    /// containing tightly packed premultiplied RgbaU8 pixels.
    ///
    /// The size needs to match the data provided.
    ///
    /// Pixmap's width is limited by i32::MAX/4.
    pub fn from_vec(data: Vec<u8>, size: IntSize) -> Option<Self> {
        let stride = min_row_bytes(size, PixelType::Rgba8U)?;
        let data_len = data_len_for_size(size, stride.get())?;
        if data.len() != data_len {
            return None;
        }

        Some(Pixmap {
            data: AlignedMemory::Align1(data.into_boxed_slice()),
            size,
            stride: stride.get(),
            pixel_type: PixelType::Rgba8U,
        })
    }

    /// Decodes a PNG data into a `Pixmap`.
    ///
    /// Only 8-bit images are supported.
    /// Index PNGs are not supported.
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
        let stride = min_row_bytes(size, PixelType::Rgba8U)
            .ok_or_else(|| make_custom_png_error("image is too big"))?;
        let data_len = data_len_for_size(size, stride.get())
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

        // Premultiply alpha.
        //
        // We cannon use RasterPipeline here, which is faster,
        // because it produces slightly different results.
        // Seems like Skia does the same.
        //
        // Also, in our tests unsafe version (no bound checking)
        // had roughly the same performance. So we keep the safe one.
        for pixel in img_data
            .as_mut_slice()
            .chunks_mut(usize::from(PixelType::Rgba8U.size()))
        {
            let a = pixel[3];
            pixel[0] = premultiply_u8(pixel[0], a);
            pixel[1] = premultiply_u8(pixel[1], a);
            pixel[2] = premultiply_u8(pixel[2], a);
        }

        Pixmap::from_vec(img_data, size)
            .ok_or_else(|| make_custom_png_error("failed to create a pixmap"))
    }

    /// Loads a PNG file into a `Pixmap`.
    ///
    /// Only 8-bit images are supported.
    /// Index PNGs are not supported.
    #[cfg(feature = "png-format")]
    pub fn load_png<P: AsRef<std::path::Path>>(path: P) -> Result<Self, png::DecodingError> {
        // `png::Decoder` is generic over input, which means that it will instance
        // two copies: one for `&[]` and one for `File`. Which will simply bloat the code.
        // Therefore we're using only one type for input.
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

    /// Returns a container that references Pixmap's data.
    pub fn as_ref(&self) -> PixmapRef<'_> {
        let (size, stride, pixel_type) = (self.size, self.stride, self.pixel_type);
        PixmapRef {
            data: self.data(),
            size,
            stride,
            pixel_type,
        }
    }

    /// Returns a container that references Pixmap's data.
    pub fn as_mut(&mut self) -> PixmapMut<'_> {
        let (size, stride, pixel_type) = (self.size, self.stride, self.pixel_type);
        PixmapMut {
            data: self.data_mut(),
            size,
            stride,
            pixel_type,
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

    /// Returns pixmap's pixel type
    #[inline]
    pub fn pixel_type(&self) -> PixelType {
        self.pixel_type
    }

    /// Returns pixmap's stride (spacing in bytes between the starts of rows)
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns pixmap's size.
    #[allow(dead_code)]
    pub(crate) fn size(&self) -> IntSize {
        self.size
    }

    /// Fills the entire pixmap with a specified color.
    pub fn fill(&mut self, color: Color) {
        self.as_mut().fill(color);
    }

    /// Returns the internal data.
    ///
    /// Byteorder: RGBA
    pub fn data(&self) -> &[u8] {
        match &self.data {
            AlignedMemory::Align1(mem) => mem,
        }
    }

    /// Returns the mutable internal data.
    ///
    /// Byteorder: RGBA
    pub fn data_mut(&mut self) -> &mut [u8] {
        match &mut self.data {
            AlignedMemory::Align1(mem) => mem,
        }
    }

    /// Returns a pixel color, converted to `Color`.
    ///
    /// Returns `None` when position is out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> Option<PremultipliedColor> {
        self.as_ref().pixel(x, y)
    }

    /// Consumes the internal data.
    ///
    /// See [PixelType] and [Self::stride()] for the data layout.
    pub fn take(self) -> AlignedMemory {
        self.data
    }

    /// Consumes the pixmap and returns the internal data as demultiplied RGBA bytes.
    ///
    /// See [PixelType] for the data layout.
    pub fn take_demultiplied(mut self) -> AlignedMemory {
        // Demultiply alpha.
        //
        // RasterPipeline is 15% faster here, but produces slightly different results
        // due to rounding. So we stick with this method for now.
        match self.pixel_type {
            PixelType::Rgba8U => {
                let pixels_mut: &mut [PremultipliedColorU8] =
                    bytemuck::cast_slice_mut(self.data_mut());
                for pixel in pixels_mut {
                    let c = pixel.demultiply();
                    *pixel = PremultipliedColorU8::from_rgba_unchecked(
                        c.red(),
                        c.green(),
                        c.blue(),
                        c.alpha(),
                    );
                }
            }
        }
        self.data
    }

    /// Returns a copy of the pixmap that intersects the `rect`.
    ///
    /// Returns `None` when `Pixmap`'s rect doesn't contain `rect`.
    pub fn clone_rect(&self, rect: IntRect) -> Option<Pixmap> {
        self.as_ref().clone_rect(rect)
    }
}

impl core::fmt::Debug for Pixmap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Pixmap")
            .field("data", &"...")
            .field("width", &self.size.width())
            .field("height", &self.size.height())
            .field("stride", &self.stride())
            .field("pixel_type", &self.pixel_type())
            .finish()
    }
}

/// A container that references premultiplied RGBA pixels.
///
/// Can be created from `Pixmap` or from a user provided data.
#[derive(Clone, Copy, PartialEq)]
pub struct PixmapRef<'a> {
    data: &'a [u8],
    size: IntSize,
    stride: usize,
    pixel_type: PixelType,
}

impl<'a> PixmapRef<'a> {
    /// Creates a new `PixmapRef` from bytes.
    ///
    /// The size must be at least `size.width() * size.height() * PixelType::Rgba8U.size()`.
    /// Zero size in an error. Width is limited by i32::MAX/4.
    ///
    /// The `data` is assumed to have premultiplied RGBA pixels (byteorder: RGBA).
    pub fn from_bytes(data: &'a [u8], width: u32, height: u32) -> Option<Self> {
        let size = IntSize::from_wh(width, height)?;
        let stride = min_row_bytes(size, PixelType::Rgba8U)?;
        let data_len = data_len_for_size(size, stride.get())?;
        if data.len() < data_len {
            return None;
        }

        Some(PixmapRef {
            data,
            size,
            stride: stride.get(),
            pixel_type: PixelType::Rgba8U,
        })
    }

    /// Creates a new `PixmapRef` from bytes.
    ///
    /// The size must be at least `size.width() * size.height() * pixel_type.size()`.
    /// Zero size in an error. Width is limited by i32::MAX/pixel_type.size().
    ///
    /// The `data` is assumed to have pixels following the given [PixelType]); it
    /// must have the minimum alignment of the [PixelType]).
    pub fn from_bytes_with_type(
        data: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
        pixel_type: PixelType,
    ) -> Option<Self> {
        let size = IntSize::from_wh(width, height)?;
        let min_stride = min_row_bytes(size, pixel_type)?;
        if min_stride.get() > stride {
            return None;
        }
        let data_len = data_len_for_size(size, stride)?;
        if data.len() != data_len {
            return None;
        }

        if data.as_ptr() as usize % usize::from(pixel_type.alignment()) != 0 {
            return None;
        }

        Some(PixmapRef {
            data,
            size,
            stride,
            pixel_type,
        })
    }

    /// Creates a new `Pixmap` from the current data.
    ///
    /// Clones the underlying data; panics on allocation failure.
    pub fn to_owned(&self) -> Pixmap {
        // Create a tightly packed copy, so that views of larger images don't copy
        // the entire thing
        let new_stride = min_row_bytes(self.size(), self.pixel_type)
            .expect("new stride should be no more than old stride")
            .get();
        let mut new =
            Pixmap::new_with_type(self.width(), self.height(), new_stride, self.pixel_type)
                .unwrap();
        {
            let old_data = self.data();
            let bpp = usize::from(self.pixel_type.size());
            let new_stride = new.stride();
            let mut new_mut = new.as_mut();
            let new_mut = new_mut.data_mut();

            for y in 0..self.height() {
                let old_start = y as usize * self.stride();
                let new_start = y as usize * new_stride;
                let slice_len = (self.width() as usize) * bpp;

                new_mut[new_start..new_start + slice_len]
                    .copy_from_slice(&old_data[old_start..old_start + slice_len]);
            }
        }

        new
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

    /// Returns pixmap's pixel type
    #[inline]
    pub fn pixel_type(&self) -> PixelType {
        self.pixel_type
    }

    /// Returns pixmap's stride (spacing in bytes between the starts of rows)
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns pixmap's size.
    pub(crate) fn size(&self) -> IntSize {
        self.size
    }

    /// Returns the internal data.
    ///
    /// See the output of [Pixmap::pixel_type] for the pixel format.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Returns a pixel color, converted to `Color`.
    ///
    /// Returns `None` when position is out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> Option<PremultipliedColor> {
        if y >= self.height() || x >= self.width() {
            return None;
        }
        let row = &self.data()[y as usize * self.stride..y as usize * self.stride + self.stride];
        match self.pixel_type {
            PixelType::Rgba8U => {
                let row = bytemuck::cast_slice::<_, PremultipliedColorU8>(
                    &row[..self.width() as usize * 4],
                );
                let color_u8 = row[x as usize];
                Some(color_u8.to_color())
            }
        }
    }

    /// Returns a copy of the pixmap that intersects the `rect`.
    ///
    /// Returns `None` when `Pixmap`'s rect doesn't contain `rect`.
    pub fn clone_rect(&self, rect: IntRect) -> Option<Pixmap> {
        Some(self.subpixmap(rect)?.to_owned())
    }

    /// Encodes pixmap into a PNG data.
    #[cfg(feature = "png-format")]
    pub fn encode_png(&self) -> Result<Vec<u8>, png::EncodingError> {
        // Skia uses skcms here, which is somewhat similar to RasterPipeline.

        // Sadly, we have to copy the pixmap here, because of demultiplication.
        // Not sure how to avoid this. (png::Encoder::stream_writer_with_size?)
        // TODO: remove allocation
        let AlignedMemory::Align1(demultiplied_data) = self.to_owned().take_demultiplied();
        let mut data = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut data, self.width(), self.height());
            encoder.set_color(png::ColorType::Rgba);
            match self.pixel_type {
                PixelType::Rgba8U => encoder.set_depth(png::BitDepth::Eight),
            }
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

    /// Returns a reference to the pixmap region that intersects the `rect`.
    ///
    /// Returns `None` when `Pixmap`'s rect doesn't contain `rect`.
    pub fn subpixmap(&self, rect: IntRect) -> Option<PixmapRef<'_>> {
        let rect = self.size.to_int_rect(0, 0).intersect(&rect)?;
        let offset = rect.top() as usize * self.stride
            + rect.left() as usize * usize::from(self.pixel_type.size());

        Some(PixmapRef {
            size: rect.size(),
            stride: self.stride,
            data: &self.data[offset..],
            pixel_type: self.pixel_type,
        })
    }
}

impl core::fmt::Debug for PixmapRef<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PixmapRef")
            .field("data", &"...")
            .field("width", &self.size.width())
            .field("height", &self.size.height())
            .field("stride", &self.stride())
            .field("pixel_type", &self.pixel_type())
            .finish()
    }
}

/// A container that references mutable premultiplied RGBA pixels.
///
/// Can be created from `Pixmap` or from a user provided data.
///
/// This may have any stride >= width * sizeof(one pixel of PixelType)
#[derive(PartialEq)]
pub struct PixmapMut<'a> {
    data: &'a mut [u8],
    size: IntSize,
    stride: usize,
    pixel_type: PixelType,
}

impl<'a> PixmapMut<'a> {
    /// Creates a new `PixmapMut` from bytes.
    ///
    /// The size must be at least `size.width() * size.height() * 4`.
    /// Zero size in an error. Width is limited by i32::MAX/4.
    ///
    /// The `data` is assumed to have premultiplied RGBA pixels (see [PixelType::Rgba8U]).
    pub fn from_bytes(data: &'a mut [u8], width: u32, height: u32) -> Option<Self> {
        let size = IntSize::from_wh(width, height)?;
        let stride = min_row_bytes(size, PixelType::Rgba8U)?;
        let data_len = data_len_for_size(size, stride.get())?;
        if data.len() < data_len {
            return None;
        }

        Some(PixmapMut {
            data,
            size,
            stride: stride.get(),
            pixel_type: PixelType::Rgba8U,
        })
    }

    /// Creates a new `PixmapMut` from bytes.
    ///
    /// The size must be at least `size.width() * size.height() * pixel_type.size()`.
    /// Zero size in an error. Width is limited by i32::MAX/pixel_type.size().
    ///
    /// The `data` is assumed to have pixels following the given [PixelType]); it
    /// must have the minimum alignment of the [PixelType]).
    pub fn from_bytes_with_type(
        data: &'a mut [u8],
        width: u32,
        height: u32,
        stride: usize,
        pixel_type: PixelType,
    ) -> Option<Self> {
        let size = IntSize::from_wh(width, height)?;
        let min_stride = min_row_bytes(size, pixel_type)?;
        if min_stride.get() > stride {
            return None;
        }
        let data_len = data_len_for_size(size, stride)?;
        if data.len() != data_len {
            return None;
        }

        if data.as_ptr() as usize % usize::from(pixel_type.alignment()) != 0 {
            return None;
        }

        Some(PixmapMut {
            data,
            size,
            stride,
            pixel_type,
        })
    }

    /// Creates a new `Pixmap` from the current data.
    ///
    /// Clones the underlying data.
    pub fn to_owned(&self) -> Pixmap {
        self.as_ref().to_owned()
    }

    /// Returns a container that references Pixmap's data.
    pub fn as_ref(&self) -> PixmapRef<'_> {
        PixmapRef {
            data: self.data,
            size: self.size,
            stride: self.stride,
            pixel_type: self.pixel_type,
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

    /// Returns pixmap's pixel type
    #[inline]
    pub fn pixel_type(&self) -> PixelType {
        self.pixel_type
    }

    /// Returns pixmap's stride (spacing in bytes between the starts of rows)
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns pixmap's size.
    pub(crate) fn size(&self) -> IntSize {
        self.size
    }

    /// Fills the entire pixmap with a specified color.
    pub fn fill(&mut self, color: Color) {
        let stride = self.stride;
        let row_len = (self.width() as usize) * usize::from(self.pixel_type.size());

        match self.pixel_type {
            PixelType::Rgba8U => {
                let c = color.premultiply().to_color_u8();
                for y in 0..self.height() {
                    let row = &mut self.data[y as usize * stride..y as usize * stride + row_len];
                    for p in bytemuck::cast_slice_mut(row) {
                        *p = c;
                    }
                }
            }
        }
    }

    /// Returns the mutable internal data.
    ///
    /// See the output of [Pixmap::pixel_type] for the pixel format.
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }

    // /// Returns a mutable slice of pixels.
    // pub fn pixels_mut(&mut self) -> &mut [PremultipliedColorU8] {
    //     bytemuck::cast_slice_mut(self.data_mut())
    // }

    /// Returns a mutable reference to the pixmap region that intersects the `rect`.
    ///
    /// Returns `None` when `Pixmap`'s rect doesn't contain `rect`.
    pub fn subpixmap(&mut self, rect: IntRect) -> Option<PixmapMut<'_>> {
        let rect = self.size.to_int_rect(0, 0).intersect(&rect)?;
        let offset = rect.top() as usize * self.stride
            + rect.left() as usize * usize::from(self.pixel_type.size());

        Some(PixmapMut {
            size: rect.size(),
            stride: self.stride,
            data: &mut self.data[offset..],
            pixel_type: self.pixel_type,
        })
    }
}

impl core::fmt::Debug for PixmapMut<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PixmapMut")
            .field("data", &"...")
            .field("width", &self.size.width())
            .field("height", &self.size.height())
            .field("stride", &self.stride())
            .field("pixel_type", &self.pixel_type())
            .finish()
    }
}

/// Returns minimum bytes per row as usize.
///
/// Pixmap's maximum value for row bytes must fit in 31 bits.
fn min_row_bytes(size: IntSize, pixel_type: PixelType) -> Option<NonZeroUsize> {
    let w = i32::try_from(size.width()).ok()?;
    let w = w.checked_mul(pixel_type.size() as i32)?;
    NonZeroUsize::new(w as usize)
}

/// Returns storage size required by pixel array. Assumes stride is valid.
///
/// The caller must have validated `stride` beforehand.
///
/// For simplicity, this requires the trailing row include the entire padding
/// up to 'stride'.
fn data_len_for_size(size: IntSize, stride: usize) -> Option<usize> {
    (size.height() as usize).checked_mul(stride)
}
