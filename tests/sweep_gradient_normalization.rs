use tiny_skia::{
    Color, GradientStop, Paint, Pixmap, Point, Rect, SpreadMode, SweepGradient, Transform,
};

fn sweep(start: f32, end: f32, force_hq_pipeline: bool) -> Pixmap {
    let paint = Paint {
        shader: SweepGradient::new(
            // Cardinal sample positions coincide with pixel centers.
            Point::from_xy(32.5, 32.5),
            start,
            end,
            vec![
                GradientStop::new(0.0, Color::from_rgba8(255, 0, 0, 255)),
                GradientStop::new(1.0, Color::from_rgba8(0, 0, 255, 255)),
            ],
            SpreadMode::Pad,
            Transform::identity(),
        )
        .unwrap(),
        anti_alias: false,
        force_hq_pipeline,
        ..Paint::default()
    };
    let mut pixmap = Pixmap::new(64, 64).unwrap();
    pixmap.fill_rect(
        Rect::from_xywh(0.0, 0.0, 64.0, 64.0).unwrap(),
        &paint,
        Transform::identity(),
        None,
    );
    pixmap
}

fn assert_color(pixmap: &Pixmap, x: u32, y: u32, expected: [u8; 4]) {
    let pixel = pixmap.pixel(x, y).unwrap().demultiply();
    let actual = [pixel.red(), pixel.green(), pixel.blue(), pixel.alpha()];
    // Permit byte rounding in the gradient/raster pipeline.
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(&a, b)| a.abs_diff(b) <= 2),
        "pixel ({x}, {y}): expected {expected:?}, got {actual:?}"
    );
}

#[test]
fn zero_start_partial_sweep_applies_scale_without_bias() {
    for force_hq in [false, true] {
        let pixmap = sweep(0.0, 180.0, force_hq);
        assert_color(&pixmap, 56, 32, [255, 0, 0, 255]); // 0°: start
        assert_color(&pixmap, 32, 56, [128, 0, 128, 255]); // 90°: midpoint
        assert_color(&pixmap, 8, 32, [0, 0, 255, 255]); // 180°: end
        assert_color(&pixmap, 32, 8, [0, 0, 255, 255]); // 270°: padded end
    }
}

#[test]
fn shifted_full_sweep_applies_bias_without_scale() {
    for force_hq in [false, true] {
        let pixmap = sweep(90.0, 450.0, force_hq);
        assert_color(&pixmap, 56, 32, [255, 0, 0, 255]); // 0°: padded start
        assert_color(&pixmap, 32, 56, [255, 0, 0, 255]); // 90°: start
        assert_color(&pixmap, 8, 32, [191, 0, 64, 255]); // 180°: quarter
        assert_color(&pixmap, 32, 8, [128, 0, 128, 255]); // 270°: midpoint
    }
}
