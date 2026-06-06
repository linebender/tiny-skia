// Copyright 2020 Yevhenii Reizner
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// No need to use explicit 256bit AVX2 SIMD.
// `-C target-cpu=native` will autovectorize it better than us.
// Not even sure why explicit instructions are so slow...
//
// On ARM AArch64 we can actually get up to 2x performance boost by using SIMD.
//
// We also have to inline all the methods. They are pretty large,
// but without the inlining the performance is plummeting.

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
use bytemuck::cast;
#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
use core::arch::aarch64::uint16x8_t;

#[cfg(all(feature = "simd", target_feature = "avx2", target_arch = "x86"))]
use core::arch::x86::*;
#[cfg(all(feature = "simd", target_feature = "avx2", target_arch = "x86_64"))]
use core::arch::x86_64::*;

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq, Default, Debug)]
pub struct u16x16(pub [u16; 16]);

macro_rules! impl_u16x16_op {
    ($a:expr, $op:ident, $b:expr) => {
        u16x16([
            $a.0[0].$op($b.0[0]),
            $a.0[1].$op($b.0[1]),
            $a.0[2].$op($b.0[2]),
            $a.0[3].$op($b.0[3]),
            $a.0[4].$op($b.0[4]),
            $a.0[5].$op($b.0[5]),
            $a.0[6].$op($b.0[6]),
            $a.0[7].$op($b.0[7]),
            $a.0[8].$op($b.0[8]),
            $a.0[9].$op($b.0[9]),
            $a.0[10].$op($b.0[10]),
            $a.0[11].$op($b.0[11]),
            $a.0[12].$op($b.0[12]),
            $a.0[13].$op($b.0[13]),
            $a.0[14].$op($b.0[14]),
            $a.0[15].$op($b.0[15]),
        ])
    };
}

#[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
macro_rules! impl_aarch64_call {
    ($f:ident, $a:expr, $b:expr) => {
        let a = $a.split();
        let b = $b.split();
        Self(bytemuck::cast([
            unsafe { core::arch::aarch64::$f(a.0, b.0) },
            unsafe { core::arch::aarch64::$f(a.1, b.1) },
        ]))
    };
}

impl u16x16 {
    #[inline]
    pub fn splat(n: u16) -> Self {
        Self([n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n])
    }

    #[inline]
    pub fn as_slice(&self) -> &[u16; 16] {
        &self.0
    }

    #[inline]
    pub fn min(&self, rhs: &Self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                impl_aarch64_call!(vminq_u16, self, rhs)
            } else {
                impl_u16x16_op!(self, min, rhs)
            }
        }
    }

    #[inline]
    pub fn max(&self, rhs: &Self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                impl_aarch64_call!(vmaxq_u16, self, rhs)
            } else {
                impl_u16x16_op!(self, max, rhs)
            }
        }
    }

    #[inline]
    pub fn cmp_le(&self, rhs: &Self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                impl_aarch64_call!(vcleq_u16, self, rhs)
            } else {
                Self([
                    if self.0[ 0] <= rhs.0[ 0] { !0 } else { 0 },
                    if self.0[ 1] <= rhs.0[ 1] { !0 } else { 0 },
                    if self.0[ 2] <= rhs.0[ 2] { !0 } else { 0 },
                    if self.0[ 3] <= rhs.0[ 3] { !0 } else { 0 },
                    if self.0[ 4] <= rhs.0[ 4] { !0 } else { 0 },
                    if self.0[ 5] <= rhs.0[ 5] { !0 } else { 0 },
                    if self.0[ 6] <= rhs.0[ 6] { !0 } else { 0 },
                    if self.0[ 7] <= rhs.0[ 7] { !0 } else { 0 },
                    if self.0[ 8] <= rhs.0[ 8] { !0 } else { 0 },
                    if self.0[ 9] <= rhs.0[ 9] { !0 } else { 0 },
                    if self.0[10] <= rhs.0[10] { !0 } else { 0 },
                    if self.0[11] <= rhs.0[11] { !0 } else { 0 },
                    if self.0[12] <= rhs.0[12] { !0 } else { 0 },
                    if self.0[13] <= rhs.0[13] { !0 } else { 0 },
                    if self.0[14] <= rhs.0[14] { !0 } else { 0 },
                    if self.0[15] <= rhs.0[15] { !0 } else { 0 },
                ])
            }
        }
    }

    #[inline]
    pub fn blend(self, t: Self, e: Self) -> Self {
        (t & self) | (e & !self)
    }

    #[inline]
    #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))]
    pub fn split(self) -> (uint16x8_t, uint16x8_t) {
        let pair: [uint16x8_t; 2] = cast(self.0);
        (pair[0], pair[1])
    }

    /// Loads 16 8888 RGBA pixels (64 bytes) and unpacks each channel into a u16x16
    #[inline(always)]
    pub fn load_8888(data: &[u8; 64]) -> [Self; 4] {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx2"))] {
                // extract each channel by shift+mask from u32 lanes, then saturate-pack u32x8 + u32x8 -> u16x16.
                // packus_epi32 lane-swaps; permute4x64 with 0xD8 puts the halves back in order
                unsafe {
                    let p_lo = _mm256_loadu_si256(data.as_ptr() as *const __m256i);
                    let p_hi = _mm256_loadu_si256(data.as_ptr().add(32) as *const __m256i);
                    let mask = _mm256_set1_epi32(0xFF);
                    let pack = |lo, hi| _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(lo, hi));

                    let mut out = [Self::default(); 4];
                    let rr = pack(_mm256_and_si256(p_lo, mask), _mm256_and_si256(p_hi, mask));
                    let gg = pack(
                        _mm256_and_si256(_mm256_srli_epi32::<8>(p_lo), mask),
                        _mm256_and_si256(_mm256_srli_epi32::<8>(p_hi), mask),
                    );
                    let bb = pack(
                        _mm256_and_si256(_mm256_srli_epi32::<16>(p_lo), mask),
                        _mm256_and_si256(_mm256_srli_epi32::<16>(p_hi), mask),
                    );
                    let aa = pack(_mm256_srli_epi32::<24>(p_lo), _mm256_srli_epi32::<24>(p_hi));

                    _mm256_storeu_si256(out[0].0.as_mut_ptr() as *mut __m256i, rr);
                    _mm256_storeu_si256(out[1].0.as_mut_ptr() as *mut __m256i, gg);
                    _mm256_storeu_si256(out[2].0.as_mut_ptr() as *mut __m256i, bb);
                    _mm256_storeu_si256(out[3].0.as_mut_ptr() as *mut __m256i, aa);
                    out
                }
            } else {
                let mut out = [Self::default(); 4];
                for i in 0..16 {
                    out[0].0[i] = data[i * 4 + 0] as u16;
                    out[1].0[i] = data[i * 4 + 1] as u16;
                    out[2].0[i] = data[i * 4 + 2] as u16;
                    out[3].0[i] = data[i * 4 + 3] as u16;
                }
                out
            }
        }
    }

    /// Packs 4 u16x16 channels back into 16 8888 RGBA pixels (64 bytes),
    /// (channel values must fit in u8)
    #[inline(always)]
    pub fn store_8888(rgba: &[Self; 4], data: &mut [u8; 64]) {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx2"))] {
                // pack rgba into u32 pixels via (g<<8)|r and (a<<8)|b, then interleave;
                // unpack_lo/hi cross 128-bit lanes, so a final permute2x128 reassembles in order.
                unsafe {
                    let rv = _mm256_loadu_si256(rgba[0].0.as_ptr() as *const __m256i);
                    let gv = _mm256_loadu_si256(rgba[1].0.as_ptr() as *const __m256i);
                    let bv = _mm256_loadu_si256(rgba[2].0.as_ptr() as *const __m256i);
                    let av = _mm256_loadu_si256(rgba[3].0.as_ptr() as *const __m256i);

                    let rg = _mm256_or_si256(rv, _mm256_slli_epi16::<8>(gv));
                    let ba = _mm256_or_si256(bv, _mm256_slli_epi16::<8>(av));

                    let p_lo = _mm256_unpacklo_epi16(rg, ba);
                    let p_hi = _mm256_unpackhi_epi16(rg, ba);

                    let out_lo = _mm256_permute2x128_si256::<0x20>(p_lo, p_hi);
                    let out_hi = _mm256_permute2x128_si256::<0x31>(p_lo, p_hi);

                    _mm256_storeu_si256(data.as_mut_ptr() as *mut __m256i, out_lo);
                    _mm256_storeu_si256(data.as_mut_ptr().add(32) as *mut __m256i, out_hi);
                }
            } else {
                for i in 0..16 {
                    data[i * 4 + 0] = rgba[0].0[i] as u8;
                    data[i * 4 + 1] = rgba[1].0[i] as u8;
                    data[i * 4 + 2] = rgba[2].0[i] as u8;
                    data[i * 4 + 3] = rgba[3].0[i] as u8;
                }
            }
        }
    }

    /// Widens 16 u8 bytes into u16x16
    #[inline(always)]
    pub fn load_u8(data: &[u8; 16]) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx2"))] {
                unsafe {
                    let bytes = _mm_loadu_si128(data.as_ptr() as *const __m128i);
                    let widened = _mm256_cvtepu8_epi16(bytes);
                    let mut out = Self::default();
                    _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, widened);
                    out
                }
            } else {
                Self([
                    data[ 0] as u16, data[ 1] as u16, data[ 2] as u16, data[ 3] as u16,
                    data[ 4] as u16, data[ 5] as u16, data[ 6] as u16, data[ 7] as u16,
                    data[ 8] as u16, data[ 9] as u16, data[10] as u16, data[11] as u16,
                    data[12] as u16, data[13] as u16, data[14] as u16, data[15] as u16,
                ])
            }
        }
    }
}

impl core::ops::Add<u16x16> for u16x16 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                impl_aarch64_call!(vaddq_u16, self, rhs)
            } else {
                impl_u16x16_op!(self, add, rhs)
            }
        }
    }
}

impl core::ops::Sub<u16x16> for u16x16 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                impl_aarch64_call!(vsubq_u16, self, rhs)
            } else {
                impl_u16x16_op!(self, sub, rhs)
            }
        }
    }
}

impl core::ops::Mul<u16x16> for u16x16 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                impl_aarch64_call!(vmulq_u16, self, rhs)
            } else {
                impl_u16x16_op!(self, mul, rhs)
            }
        }
    }
}

impl core::ops::Div<u16x16> for u16x16 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        impl_u16x16_op!(self, div, rhs)
    }
}

impl core::ops::BitAnd<u16x16> for u16x16 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                impl_aarch64_call!(vandq_u16, self, rhs)
            } else {
                impl_u16x16_op!(self, bitand, rhs)
            }
        }
    }
}

impl core::ops::BitOr<u16x16> for u16x16 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                impl_aarch64_call!(vorrq_u16, self, rhs)
            } else {
                impl_u16x16_op!(self, bitor, rhs)
            }
        }
    }
}

impl core::ops::Not for u16x16 {
    type Output = Self;

    #[inline]
    fn not(self) -> Self::Output {
        u16x16([
            !self.0[0],
            !self.0[1],
            !self.0[2],
            !self.0[3],
            !self.0[4],
            !self.0[5],
            !self.0[6],
            !self.0[7],
            !self.0[8],
            !self.0[9],
            !self.0[10],
            !self.0[11],
            !self.0[12],
            !self.0[13],
            !self.0[14],
            !self.0[15],
        ])
    }
}

impl core::ops::Shr for u16x16 {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: Self) -> Self::Output {
        impl_u16x16_op!(self, shr, rhs)
    }
}
