// Copyright 2020 Yevhenii Reizner
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Based on https://github.com/Lokathor/wide (Zlib)

use bytemuck::cast;

use super::{i32x8, u32x8};

cfg_if::cfg_if! {
    if #[cfg(all(feature = "simd", target_feature = "avx"))] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        #[derive(Clone, Copy, Debug)]
        #[repr(C, align(32))]
        pub struct f32x8(__m256);
    } else {
        use super::f32x4;

        #[derive(Clone, Copy, Debug)]
        #[repr(C, align(32))]
        pub struct f32x8(pub f32x4, pub f32x4);
    }
}

unsafe impl bytemuck::Zeroable for f32x8 {}
unsafe impl bytemuck::Pod for f32x8 {}

impl Default for f32x8 {
    fn default() -> Self {
        Self::splat(0.0)
    }
}

impl f32x8 {
    pub fn splat(n: f32) -> Self {
        cast([n, n, n, n, n, n, n, n])
    }

    pub fn floor(self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "simd128"))] {
                Self(self.0.floor(), self.1.floor())
            } else {
                let roundtrip: f32x8 = cast(self.trunc_int().to_f32x8());
                roundtrip
                    - roundtrip
                        .cmp_gt(self)
                        .blend(f32x8::splat(1.0), f32x8::default())
            }
        }
    }

    pub fn fract(self) -> Self {
        self - self.floor()
    }

    pub fn normalize(self) -> Self {
        self.max(f32x8::default()).min(f32x8::splat(1.0))
    }

    pub fn to_i32x8_bitcast(self) -> i32x8 {
        bytemuck::cast(self)
    }

    pub fn to_u32x8_bitcast(self) -> u32x8 {
        bytemuck::cast(self)
    }

    pub fn cmp_eq(self, rhs: Self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_EQ_OQ) })
            } else {
                Self(self.0.cmp_eq(rhs.0), self.1.cmp_eq(rhs.1))
            }
        }
    }

    pub fn cmp_ne(self, rhs: Self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                // Use UQ (unordered quiet) to match SSE _mm_cmpneq_ps behavior,
                // which returns true when either operand is NaN.
                Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_NEQ_UQ) })
            } else {
                Self(self.0.cmp_ne(rhs.0), self.1.cmp_ne(rhs.1))
            }
        }
    }

    pub fn cmp_ge(self, rhs: Self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_GE_OQ) })
            } else {
                Self(self.0.cmp_ge(rhs.0), self.1.cmp_ge(rhs.1))
            }
        }
    }

    pub fn cmp_gt(self, rhs: Self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_GT_OQ) })
            } else {
                Self(self.0.cmp_gt(rhs.0), self.1.cmp_gt(rhs.1))
            }
        }
    }

    pub fn cmp_le(self, rhs: Self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_LE_OQ) })
            } else {
                Self(self.0.cmp_le(rhs.0), self.1.cmp_le(rhs.1))
            }
        }
    }

    pub fn cmp_lt(self, rhs: Self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_LT_OQ) })
            } else {
                Self(self.0.cmp_lt(rhs.0), self.1.cmp_lt(rhs.1))
            }
        }
    }

    #[inline]
    pub fn blend(self, t: Self, f: Self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_blendv_ps(f.0, t.0, self.0) })
            } else {
                Self(self.0.blend(t.0, f.0), self.1.blend(t.1, f.1))
            }
        }
    }

    pub fn abs(self) -> Self {
        let non_sign_bits = f32x8::splat(f32::from_bits(i32::MAX as u32));
        self & non_sign_bits
    }

    pub fn max(self, rhs: Self) -> Self {
        // These technically don't have the same semantics for NaN and 0, but it
        // doesn't seem to matter as Skia does it the same way.
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_max_ps(self.0, rhs.0) })
            } else {
                Self(self.0.max(rhs.0), self.1.max(rhs.1))
            }
        }
    }

    pub fn min(self, rhs: Self) -> Self {
        // These technically don't have the same semantics for NaN and 0, but it
        // doesn't seem to matter as Skia does it the same way.
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_min_ps(self.0, rhs.0) })
            } else {
                Self(self.0.min(rhs.0), self.1.min(rhs.1))
            }
        }
    }

    pub fn is_finite(self) -> Self {
        let shifted_exp_mask = u32x8::splat(0xFF000000);
        let u: u32x8 = cast(self);
        let shift_u = u.shl::<1>();
        let out = !(shift_u & shifted_exp_mask).cmp_eq(shifted_exp_mask);
        cast(out)
    }

    pub fn round(self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_round_ps(self.0, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT) })
            } else {
                Self(self.0.round(), self.1.round())
            }
        }
    }

    pub fn round_int(self) -> i32x8 {
        // These technically don't have the same semantics for NaN and out of
        // range values, but it doesn't seem to matter as Skia does it the same
        // way.
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                cast(unsafe { _mm256_cvtps_epi32(self.0) })
            } else {
                i32x8(self.0.round_int(), self.1.round_int())
            }
        }
    }

    pub fn trunc_int(self) -> i32x8 {
        // These technically don't have the same semantics for NaN and out of
        // range values, but it doesn't seem to matter as Skia does it the same
        // way.
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                cast(unsafe { _mm256_cvttps_epi32(self.0) })
            } else {
                i32x8(self.0.trunc_int(), self.1.trunc_int())
            }
        }
    }

    pub fn recip_fast(self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_rcp_ps(self.0) })
            } else {
                Self(self.0.recip_fast(), self.1.recip_fast())
            }
        }
    }

    pub fn recip_sqrt(self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_rsqrt_ps(self.0) })
            } else {
                Self(self.0.recip_sqrt(), self.1.recip_sqrt())
            }
        }
    }

    pub fn sqrt(self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_sqrt_ps(self.0) })
            } else {
                Self(self.0.sqrt(), self.1.sqrt())
            }
        }
    }

    pub fn powf(self, exp: f32) -> Self {
        let x = self;
        // We assume sign(x) is positive so we can use vectorized i32->f32 conversions
        let e = x.to_i32x8_bitcast().to_f32x8() * f32x8::splat(1.0f32 / ((1 << 23) as f32));
        let m = (x.to_u32x8_bitcast() & u32x8::splat(0x007fffff) | u32x8::splat(0x3f000000))
            .to_f32x8_bitcast();

        let log2_x = e
            - f32x8::splat(124.225514990)
            - f32x8::splat(1.498030302) * m
            - f32x8::splat(1.725879990) / (f32x8::splat(0.3520887068) + m);

        let x = log2_x * f32x8::splat(exp);

        let f = x - x.floor();

        let mut a = x + f32x8::splat(121.274057500);
        a = a - f * f32x8::splat(1.490129070);
        a += f32x8::splat(27.728023300) / (f32x8::splat(4.84252568) - f);
        a *= f32x8::splat((1 << 23) as f32);

        let inf_bits = f32x8::splat(f32::INFINITY.to_bits() as f32);

        let x = a
            .max(f32x8::splat(0.0))
            .min(inf_bits)
            .round_int()
            .to_f32x8_bitcast();

        let skip = self.cmp_eq(f32x8::splat(0.0)) | self.cmp_eq(f32x8::splat(1.0));
        skip.blend(self, x)
    }

    /// Loads 8 8888 RGBA pixels, unpacks each channel into a
    /// normalized f32x8 in [0, 1]
    #[inline(always)]
    pub fn load_8888_unorm(data: &[u8; 32]) -> [Self; 4] {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx2"))] {
                unsafe {
                    let p = _mm256_loadu_si256(data.as_ptr() as *const __m256i);
                    let mask = _mm256_set1_epi32(0xFF);
                    let factor = _mm256_set1_ps(1.0 / 255.0);
                    let to_f = |v| _mm256_mul_ps(_mm256_cvtepi32_ps(v), factor);

                    [
                        Self(to_f(_mm256_and_si256(p, mask))),
                        Self(to_f(_mm256_and_si256(_mm256_srli_epi32::<8>(p), mask))),
                        Self(to_f(_mm256_and_si256(_mm256_srli_epi32::<16>(p), mask))),
                        Self(to_f(_mm256_srli_epi32::<24>(p))),
                    ]
                }
            } else {
                // surprisingly, `f32 * FACTOR` is way faster than `f32x8 * f32x8::splat(FACTOR)`.
                const FACTOR: f32 = 1.0 / 255.0;
                let b = |i: usize, ch: usize| data[i * 4 + ch] as f32 * FACTOR;
                [
                    Self::from([b(0, 0), b(1, 0), b(2, 0), b(3, 0), b(4, 0), b(5, 0), b(6, 0), b(7, 0)]),
                    Self::from([b(0, 1), b(1, 1), b(2, 1), b(3, 1), b(4, 1), b(5, 1), b(6, 1), b(7, 1)]),
                    Self::from([b(0, 2), b(1, 2), b(2, 2), b(3, 2), b(4, 2), b(5, 2), b(6, 2), b(7, 2)]),
                    Self::from([b(0, 3), b(1, 3), b(2, 3), b(3, 3), b(4, 3), b(5, 3), b(6, 3), b(7, 3)]),
                ]
            }
        }
    }

    /// Packs 4 f32x8 channels in [0, 1] back into 8 8888 RGBA pixels (32 bytes).
    /// Matches the scalar `unnorm` semantics: clamp -> *255 -> round-to-nearest.
    #[inline(always)]
    pub fn store_8888_unorm(rgba: &[Self; 4], data: &mut [u8; 32]) {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx2"))] {
                unsafe {
                    let scale = _mm256_set1_ps(255.0);
                    let zero = _mm256_setzero_ps();
                    let one = _mm256_set1_ps(1.0);
                    let to_u32 = |v| {
                        let clamped = _mm256_min_ps(_mm256_max_ps(v, zero), one);
                        _mm256_cvtps_epi32(_mm256_mul_ps(clamped, scale))
                    };

                    let ri = to_u32(rgba[0].0);
                    let gi = to_u32(rgba[1].0);
                    let bi = to_u32(rgba[2].0);
                    let ai = to_u32(rgba[3].0);

                    let packed = _mm256_or_si256(
                        _mm256_or_si256(ri, _mm256_slli_epi32::<8>(gi)),
                        _mm256_or_si256(_mm256_slli_epi32::<16>(bi), _mm256_slli_epi32::<24>(ai)),
                    );
                    _mm256_storeu_si256(data.as_mut_ptr() as *mut __m256i, packed);
                }
            } else {
                let unnorm = |v: Self| -> [i32; 8] {
                    (v.max(Self::default()).min(Self::splat(1.0)) * Self::splat(255.0))
                        .round_int()
                        .into()
                };
                let r = unnorm(rgba[0]);
                let g = unnorm(rgba[1]);
                let b = unnorm(rgba[2]);
                let a = unnorm(rgba[3]);
                for i in 0..8 {
                    data[i * 4 + 0] = r[i] as u8;
                    data[i * 4 + 1] = g[i] as u8;
                    data[i * 4 + 2] = b[i] as u8;
                    data[i * 4 + 3] = a[i] as u8;
                }
            }
        }
    }
}

impl From<[f32; 8]> for f32x8 {
    fn from(v: [f32; 8]) -> Self {
        cast(v)
    }
}

impl From<f32x8> for [f32; 8] {
    fn from(v: f32x8) -> Self {
        cast(v)
    }
}

impl core::ops::Add for f32x8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_add_ps(self.0, rhs.0) })
            } else {
                Self(self.0 + rhs.0, self.1 + rhs.1)
            }
        }
    }
}

impl core::ops::AddAssign for f32x8 {
    fn add_assign(&mut self, rhs: f32x8) {
        *self = *self + rhs;
    }
}

impl core::ops::Sub for f32x8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_sub_ps(self.0, rhs.0) })
            } else {
                Self(self.0 - rhs.0, self.1 - rhs.1)
            }
        }
    }
}

impl core::ops::Mul for f32x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_mul_ps(self.0, rhs.0) })
            } else {
                Self(self.0 * rhs.0, self.1 * rhs.1)
            }
        }
    }
}

impl core::ops::MulAssign for f32x8 {
    fn mul_assign(&mut self, rhs: f32x8) {
        *self = *self * rhs;
    }
}

impl core::ops::Div for f32x8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_div_ps(self.0, rhs.0) })
            } else {
                Self(self.0 / rhs.0, self.1 / rhs.1)
            }
        }
    }
}

impl core::ops::BitAnd for f32x8 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_and_ps(self.0, rhs.0) })
            } else {
                Self(self.0 & rhs.0, self.1 & rhs.1)
            }
        }
    }
}

impl core::ops::BitOr for f32x8 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_or_ps(self.0, rhs.0) })
            } else {
                Self(self.0 | rhs.0, self.1 | rhs.1)
            }
        }
    }
}

impl core::ops::BitXor for f32x8 {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                Self(unsafe { _mm256_xor_ps(self.0, rhs.0) })
            } else {
                Self(self.0 ^ rhs.0, self.1 ^ rhs.1)
            }
        }
    }
}

impl core::ops::Neg for f32x8 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::default() - self
    }
}

impl core::ops::Not for f32x8 {
    type Output = Self;

    fn not(self) -> Self {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                let all_bits = unsafe { _mm256_set1_ps(f32::from_bits(u32::MAX)) };
                Self(unsafe { _mm256_xor_ps(self.0, all_bits) })
            } else {
                Self(!self.0, !self.1)
            }
        }
    }
}

impl core::cmp::PartialEq for f32x8 {
    fn eq(&self, rhs: &Self) -> bool {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_feature = "avx"))] {
                let mask = unsafe { _mm256_cmp_ps(self.0, rhs.0, _CMP_EQ_OQ) };
                unsafe { _mm256_movemask_ps(mask) == 0b1111_1111 }
            } else {
                self.0 == rhs.0 && self.1 == rhs.1
            }
        }
    }
}
