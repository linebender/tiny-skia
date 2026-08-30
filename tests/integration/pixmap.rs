use tiny_skia::*;

#[test]
fn clone_rect_1() {
    let mut pixmap = Pixmap::new(200, 200).unwrap();

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    pixmap.fill_path(
        &PathBuilder::from_circle(100.0, 100.0, 80.0).unwrap(),
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );

    let part = pixmap.as_ref().clone_rect(IntRect::from_xywh(10, 15, 80, 90).unwrap()).unwrap();

    let expected = Pixmap::load_png("tests/images/pixmap/clone-rect-1.png").unwrap();
    assert_eq!(part, expected);
}

#[test]
fn clone_rect_2() {
    let mut pixmap = Pixmap::new(200, 200).unwrap();

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    pixmap.fill_path(
        &PathBuilder::from_circle(100.0, 100.0, 80.0).unwrap(),
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );

    let part = pixmap.as_ref().clone_rect(IntRect::from_xywh(130, 120, 80, 90).unwrap()).unwrap();

    let expected = Pixmap::load_png("tests/images/pixmap/clone-rect-2.png").unwrap();
    assert_eq!(part, expected);
}


#[test]
fn clone_rect_out_of_bound() {
    let mut pixmap = Pixmap::new(200, 200).unwrap();

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    pixmap.fill_path(
        &PathBuilder::from_circle(100.0, 100.0, 80.0).unwrap(),
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );

    assert!(pixmap.as_ref().clone_rect(IntRect::from_xywh(250, 15, 80, 90).unwrap()).is_none());
    assert!(pixmap.as_ref().clone_rect(IntRect::from_xywh(10, 250, 80, 90).unwrap()).is_none());
    assert!(pixmap.as_ref().clone_rect(IntRect::from_xywh(10, -250, 80, 90).unwrap()).is_none());
}

#[test]
fn fill() {
    let c = Color::from_rgba8(50, 100, 150, 200);
    let mut pixmap = Pixmap::new(10, 10).unwrap();
    pixmap.fill(c);
    let pixels = bytemuck::cast_slice::<_, PremultipliedColorU8>(pixmap.data());
    assert!(pixels.iter().all(|p| { *p == c.premultiply().to_color_u8() }));
}

#[test]
fn draw_pixmap() {
    // Tests that painting algorithm will switch `Bicubic`/`Bilinear` to `Nearest`.
    // Otherwise we will get a blurry image.

    // A pixmap with the bottom half filled with solid color.
    let sub_pixmap = {
        let mut paint = Paint::default();
        paint.set_color_rgba8(50, 127, 150, 200);
        paint.anti_alias = false;

        let rect = Rect::from_xywh(0.0, 50.0, 100.0, 50.0).unwrap();

        let mut pixmap = Pixmap::new(100, 100).unwrap();
        pixmap.fill_rect(rect, &paint, Transform::identity(), None);
        pixmap
    };

    let mut paint = PixmapPaint::default();
    paint.quality = FilterQuality::Bicubic;

    let mut pixmap = Pixmap::new(200, 200).unwrap();
    pixmap.draw_pixmap(20, 20, sub_pixmap.as_ref(), &paint, Transform::identity(), None);

    let expected = Pixmap::load_png("tests/images/canvas/draw-pixmap.png").unwrap();
    assert_eq!(pixmap, expected);
}

#[test]
fn draw_pixmap_ts() {
    let triangle = {
        let mut paint = Paint::default();
        paint.set_color_rgba8(50, 127, 150, 200);
        paint.anti_alias = true;

        let mut pb = PathBuilder::new();
        pb.move_to(0.0, 100.0);
        pb.line_to(100.0, 100.0);
        pb.line_to(50.0, 0.0);
        pb.close();
        let path = pb.finish().unwrap();

        let mut pixmap = Pixmap::new(100, 100).unwrap();
        pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
        pixmap
    };

    let mut paint = PixmapPaint::default();
    paint.quality = FilterQuality::Bicubic;

    let mut pixmap = Pixmap::new(200, 200).unwrap();
    pixmap.draw_pixmap(
        5, 10,
        triangle.as_ref(),
        &paint,
        Transform::from_row(1.2, 0.5, 0.5, 1.2, 0.0, 0.0),
        None,
    );

    let expected = Pixmap::load_png("tests/images/canvas/draw-pixmap-ts.png").unwrap();
    assert_eq!(pixmap, expected);
}

#[test]
fn draw_pixmap_opacity() {
    let triangle = {
        let mut paint = Paint::default();
        paint.set_color_rgba8(50, 127, 150, 200);
        paint.anti_alias = true;

        let mut pb = PathBuilder::new();
        pb.move_to(0.0, 100.0);
        pb.line_to(100.0, 100.0);
        pb.line_to(50.0, 0.0);
        pb.close();
        let path = pb.finish().unwrap();

        let mut pixmap = Pixmap::new(100, 100).unwrap();
        pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
        pixmap
    };

    let mut paint = PixmapPaint::default();
    paint.quality = FilterQuality::Bicubic;
    paint.opacity = 0.5;

    let mut pixmap = Pixmap::new(200, 200).unwrap();
    pixmap.draw_pixmap(
        5, 10,
        triangle.as_ref(),
        &paint,
        Transform::from_row(1.2, 0.5, 0.5, 1.2, 0.0, 0.0),
        None,
    );

    let expected = Pixmap::load_png("tests/images/canvas/draw-pixmap-opacity.png").unwrap();
    assert_eq!(pixmap, expected);
}

#[test]
fn type_overlay() {
    let stacks: [(PixelType, PixelType, &'static str); _] = [
        (PixelType::Rgba8U, PixelType::Rgba8U, "8u-on-8u"),
    ];

    for (base_type, over_type, name) in stacks {
        // Ensure image is not tightly packed, so that indexing calculations are exercised
        let over_stride =
            100 * usize::from(over_type.size()) + 19 * usize::from(over_type.alignment());
        let mut over = Pixmap::new_with_type(100, 100, over_stride, over_type).unwrap();

        let mut paint = Paint::default();
        paint.set_color_rgba8(50, 127, 150, 200);
        paint.anti_alias = true;
        paint.shader = LinearGradient::new(
            Point::from_xy(10.0, 10.0),
            Point::from_xy(90.0, 90.0),
            vec![
                GradientStop::new(0.0, Color::from_rgba8(50, 127, 150, 200)),
                GradientStop::new(1.0, Color::from_rgba8(220, 140, 75, 180)),
            ],
            SpreadMode::Pad,
            Transform::identity(),
        )
        .unwrap();

        let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 90.0, 90.0).unwrap());
        over.fill_path(
            &path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );

        let base_stride = 200 * usize::from(base_type.size()) + usize::from(base_type.alignment());
        let mut pixmap = Pixmap::new_with_type(200, 200, base_stride, base_type).unwrap();

        // Overlay with rotation
        let transform = Transform::from_rotate(45.0)
            .post_translate(50.0, 15.0)
            .post_scale(2.0, 2.0);

        let mut paint = Paint::default();
        paint.shader = Pattern::new(
            over.as_ref(),
            SpreadMode::Pad, // Pad, otherwise we will get weird borders overlap.
            FilterQuality::Bilinear,
            0.9,
            transform,
        );
        paint.blend_mode = BlendMode::Modulate;
        paint.anti_alias = true;
        paint.colorspace = ColorSpace::default();

        pixmap.fill(Color::from_rgba(0.4, 0.5, 0.6, 0.9).unwrap());
        pixmap.fill_path(
            &PathBuilder::from_circle(100.0, 100.0, 80.0).unwrap(),
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );

        let expected =
            Pixmap::load_png(format!("tests/images/pixmap/overlay-{}.png", name)).unwrap();
        let packed = pixmap.as_ref().to_owned();
        assert_eq!(expected, packed);
    }
}
