use tiny_skia::*;

#[test]
fn solid_fill_and_draw() {
    let mut pixmap = PixmapU16::new(100, 100).unwrap();
    assert_eq!(pixmap.width(), 100);
    assert_eq!(pixmap.height(), 100);

    // Fill with semi-transparent red
    let color = Color::from_rgba(1.0, 0.0, 0.0, 0.5).unwrap();
    let mut paint = Paint::default();
    paint.set_color(color);

    let rect = Rect::from_xywh(10.0, 10.0, 80.0, 80.0).unwrap();
    pixmap.fill_rect(rect, &paint, Transform::identity(), None);

    // Check center pixel: premultiplied alpha = 32768, red = 32768, green = 0, blue = 0
    let pixel = pixmap.pixel(50, 50).unwrap();
    assert_eq!(pixel.alpha(), 32768);
    assert_eq!(pixel.red(), 32768);
    assert_eq!(pixel.green(), 0);
    assert_eq!(pixel.blue(), 0);

    // Check outside pixel: transparent
    let pixel_out = pixmap.pixel(5, 5).unwrap();
    assert_eq!(pixel_out, PremultipliedColorU16::TRANSPARENT);
}

#[test]
fn continuous_gradient_sub_8bit_steps() {
    let mut pixmap = PixmapU16::new(512, 1).unwrap();

    // Subtle gradient spanning from #000000 (0) to #010101 (257 / 65535 in 16-bit)
    let c0 = Color::from_rgba(0.0, 0.0, 0.0, 1.0).unwrap();
    let c1 = Color::from_rgba(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 1.0).unwrap();

    let shader = LinearGradient::new(
        Point::from_xy(0.0, 0.0),
        Point::from_xy(512.0, 0.0),
        vec![
            GradientStop::new(0.0, c0),
            GradientStop::new(1.0, c1),
        ],
        SpreadMode::Pad,
        Transform::identity(),
    ).unwrap();

    let mut paint = Paint::default();
    paint.shader = shader;

    pixmap.fill_rect(
        Rect::from_xywh(0.0, 0.0, 512.0, 1.0).unwrap(),
        &paint,
        Transform::identity(),
        None,
    );

    // Assert that we have intermediate values strictly between 0 and 257
    let mut found_intermediate = false;
    let mut intermediate_values = Vec::new();

    for x in 0..512 {
        let pixel = pixmap.pixel(x, 0).unwrap();
        let r = pixel.red();
        if r > 0 && r < 257 {
            found_intermediate = true;
            intermediate_values.push(r);
        }
    }

    assert!(found_intermediate, "Pipeline must generate true 16-bit intermediate steps between 0 and 257");
    // Verify that we have multiple distinct intermediate steps
    intermediate_values.dedup();
    assert!(
        intermediate_values.len() > 10,
        "Expected multiple distinct sub-8-bit gradient steps, got {}",
        intermediate_values.len()
    );
}

#[test]
fn path_and_hairline_rendering() {
    let mut pixmap = PixmapU16::new(64, 64).unwrap();

    let mut pb = PathBuilder::new();
    pb.move_to(10.0, 10.0);
    pb.line_to(54.0, 54.0);
    let path = pb.finish().unwrap();

    let mut paint = Paint::default();
    paint.set_color(Color::from_rgba(0.0, 1.0, 0.0, 1.0).unwrap());

    let stroke = Stroke::default();
    pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);

    // Assert stroke was rendered
    let p_mid = pixmap.pixel(32, 32).unwrap();
    assert!(p_mid.green() > 30000, "Green channel should be drawn on stroke line");
    assert_eq!(p_mid.red(), 0);
    assert_eq!(p_mid.blue(), 0);
}
