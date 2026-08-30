#![feature(test)]

extern crate test;

pub fn is_16bpc() -> bool {
    std::env::var("TINY_SKIA_BENCH_16BPC").is_ok()
}

#[rustfmt::skip]
#[cfg(test)]
mod blend;
#[rustfmt::skip]
#[cfg(test)]
mod clip;
#[rustfmt::skip]
#[cfg(test)]
mod draw_pixmap;
#[rustfmt::skip]
#[cfg(test)]
mod fill;
#[rustfmt::skip]
#[cfg(test)]
mod gradients;
#[rustfmt::skip]
#[cfg(test)]
mod hairline;
#[rustfmt::skip]
#[cfg(test)]
mod patterns;
#[rustfmt::skip]
#[cfg(test)]
mod png_io;
#[rustfmt::skip]
#[cfg(test)]
mod spiral;

fn main() {}
