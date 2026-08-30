use tiny_skia::*;

#[track_caller]
pub fn assert_pixmap_eq<P: Pixel>(actual: &PixmapGeneric<P>, expected: &Pixmap, tolerance: u8) {
    assert_eq!(actual.width(), expected.width(), "pixmap width mismatch");
    assert_eq!(actual.height(), expected.height(), "pixmap height mismatch");

    for (i, (act, exp)) in actual
        .pixels()
        .iter()
        .zip(expected.pixels().iter())
        .enumerate()
    {
        let act_u8 = act.to_u8();
        let dr = (act_u8.red() as i32 - exp.red() as i32).abs();
        let dg = (act_u8.green() as i32 - exp.green() as i32).abs();
        let db = (act_u8.blue() as i32 - exp.blue() as i32).abs();
        let da = (act_u8.alpha() as i32 - exp.alpha() as i32).abs();
        let max_diff = dr.max(dg).max(db).max(da);
        if max_diff > tolerance as i32 {
            let x = (i as u32) % actual.width();
            let y = (i as u32) / actual.width();
            panic!(
                "Pixel ({}, {}) mismatch (tolerance {}): actual RGBA({}, {}, {}, {}), expected RGBA({}, {}, {}, {}), max diff {}",
                x, y, tolerance,
                act_u8.red(), act_u8.green(), act_u8.blue(), act_u8.alpha(),
                exp.red(), exp.green(), exp.blue(), exp.alpha(),
                max_diff
            );
        }
    }
}

#[track_caller]
pub fn assert_mask_eq(actual: &Mask, expected: &Mask, tolerance: u8) {
    assert_eq!(actual.width(), expected.width(), "mask width mismatch");
    assert_eq!(actual.height(), expected.height(), "mask height mismatch");

    for (i, (act, exp)) in actual.data().iter().zip(expected.data().iter()).enumerate() {
        let diff = (*act as i32 - *exp as i32).abs();
        if diff > tolerance as i32 {
            let x = (i as u32) % actual.width();
            let y = (i as u32) / actual.width();
            panic!(
                "Mask pixel ({}, {}) mismatch (tolerance {}): actual {}, expected {}, diff {}",
                x, y, tolerance, act, exp, diff
            );
        }
    }
}

#[macro_export]
macro_rules! test_raster {
    ($name:ident, $w:expr, $h:expr, $expected_png:expr, tolerance: $tol:expr, |$pixmap:ident| $body:block) => {
        #[test]
        fn $name() {
            let expected = tiny_skia::Pixmap::load_png($expected_png).unwrap();

            // 8bpc run (strict byte-for-byte matching)
            {
                let mut $pixmap = tiny_skia::Pixmap::new($w, $h).unwrap();
                $body;
                $crate::common::assert_pixmap_eq(&$pixmap, &expected, 0);
            }

            // 16bpc run (tolerance for subpixel/gamma rounding)
            #[cfg(feature = "16bpc")]
            {
                let mut $pixmap = tiny_skia::PixmapU16::new($w, $h).unwrap();
                $body;
                $crate::common::assert_pixmap_eq(&$pixmap, &expected, $tol);
            }
        }
    };
    ($name:ident, $w:expr, $h:expr, $expected_png:expr, |$pixmap:ident| $body:block) => {
        test_raster!($name, $w, $h, $expected_png, tolerance: 2, |$pixmap| $body);
    };
}
