use tiny_skia::*;

test_raster!(horizontal_line, 100, 100, "tests/images/fill/empty.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(10.0, 10.0);
    pb.line_to(90.0, 10.0);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(vertical_line, 100, 100, "tests/images/fill/empty.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(10.0, 10.0);
    pb.line_to(10.0, 90.0);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(single_line, 100, 100, "tests/images/fill/empty.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(10.0, 10.0);
    pb.line_to(90.0, 90.0);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(int_rect, 100, 100, "tests/images/fill/int-rect.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let rect = Rect::from_xywh(10.0, 15.0, 80.0, 70.0).unwrap();

    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
});

test_raster!(float_rect, 100, 100, "tests/images/fill/float-rect.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let rect = Rect::from_xywh(10.3, 15.4, 80.5, 70.6).unwrap();

    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
});

test_raster!(int_rect_aa, 100, 100, "tests/images/fill/int-rect-aa.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let rect = Rect::from_xywh(10.0, 15.0, 80.0, 70.0).unwrap();

    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
});

test_raster!(float_rect_aa, 100, 100, "tests/images/fill/float-rect-aa.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let rect = Rect::from_xywh(10.3, 15.4, 80.5, 70.6).unwrap();

    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
});

test_raster!(float_rect_aa_highp, 100, 100, "tests/images/fill/float-rect-aa-highp.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;
    paint.force_hq_pipeline = true;

    let rect = Rect::from_xywh(10.3, 15.4, 80.5, 70.6).unwrap();

    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
});

#[test]
fn tiny_float_rect() {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let rect = Rect::from_xywh(1.3, 1.4, 0.5, 0.6).unwrap();
    let mut pixmap = Pixmap::new(3, 3).unwrap();
    pixmap.fill_rect(rect, &paint, Transform::identity(), None);

    let expected = [
        ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
        ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
        ColorU8::from_rgba(0, 0, 0, 0).premultiply(),

        ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
        ColorU8::from_rgba(50, 127, 150, 200).premultiply(),
        ColorU8::from_rgba(0, 0, 0, 0).premultiply(),

        ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
        ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
        ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
    ];
    assert_eq!(pixmap.pixels(), &expected);

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(3, 3).unwrap();
        pixmap16.fill_rect(rect, &paint, Transform::identity(), None);
        let center = pixmap16.pixel(1, 1).unwrap().to_u8();
        let expected = ColorU8::from_rgba(50, 127, 150, 200).premultiply();
        let dr = (center.red() as i32 - expected.red() as i32).abs();
        let dg = (center.green() as i32 - expected.green() as i32).abs();
        let db = (center.blue() as i32 - expected.blue() as i32).abs();
        let da = (center.alpha() as i32 - expected.alpha() as i32).abs();
        assert!(dr <= 1 && dg <= 1 && db <= 1 && da <= 1);
    }
}

#[test]
fn tiny_float_rect_aa() {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let rect = Rect::from_xywh(1.3, 1.4, 0.5, 0.6).unwrap();

    let mut pixmap = Pixmap::new(3, 3).unwrap();
    pixmap.fill_rect(rect, &paint, Transform::identity(), None);

    assert_eq!(
        pixmap.pixels(),
        &[
            ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
            ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
            ColorU8::from_rgba(0, 0, 0, 0).premultiply(),

            ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
            ColorU8::from_rgba(51, 128, 153, 60).premultiply(),
            ColorU8::from_rgba(0, 0, 0, 0).premultiply(),

            ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
            ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
            ColorU8::from_rgba(0, 0, 0, 0).premultiply(),
        ]
    );

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(3, 3).unwrap();
        pixmap16.fill_rect(rect, &paint, Transform::identity(), None);
        let center = pixmap16.pixel(1, 1).unwrap().to_u8();
        let expected = ColorU8::from_rgba(51, 128, 153, 60).premultiply();
        let dr = (center.red() as i32 - expected.red() as i32).abs();
        let dg = (center.green() as i32 - expected.green() as i32).abs();
        let db = (center.blue() as i32 - expected.blue() as i32).abs();
        let da = (center.alpha() as i32 - expected.alpha() as i32).abs();
        assert!(dr <= 2 && dg <= 2 && db <= 2 && da <= 2);
    }
}

#[test]
fn tiny_rect_aa() {
    let mut paint = Paint::default();
    paint.set_color_rgba8(0, 0, 0, 0);
    paint.anti_alias = true;
    let rect = Rect::from_xywh(0.7, 0.0, 1.0, 2.0).unwrap();
    let mut pixmap = Pixmap::new(10, 10).unwrap();
    pixmap.fill_rect(rect, &paint, Transform::identity(), None);

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(10, 10).unwrap();
        pixmap16.fill_rect(rect, &paint, Transform::identity(), None);
    }
}

test_raster!(float_rect_clip_top_left_aa, 100, 100, "tests/images/fill/float-rect-clip-top-left-aa.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let rect = Rect::from_xywh(-10.3, -20.4, 100.5, 70.2).unwrap();

    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
});

test_raster!(float_rect_clip_top_right_aa, 100, 100, "tests/images/fill/float-rect-clip-top-right-aa.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let rect = Rect::from_xywh(60.3, -20.4, 100.5, 70.2).unwrap();

    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
});

test_raster!(float_rect_clip_bottom_right_aa, 100, 100, "tests/images/fill/float-rect-clip-bottom-right-aa.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let rect = Rect::from_xywh(60.3, 40.4, 100.5, 70.2).unwrap();

    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
});

test_raster!(int_rect_with_ts_clip_right, 100, 100, "tests/images/fill/int-rect-with-ts-clip-right.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let rect = Rect::from_xywh(0.0, 0.0, 100.0, 100.0).unwrap();

    pixmap.fill_rect(rect, &paint, Transform::from_row(1.0, 0.0, 0.0, 1.0, 0.5, 0.5), None);
});

test_raster!(open_polygon, 100, 100, "tests/images/fill/polygon.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(75.160671, 88.756136);
    pb.line_to(24.797274, 88.734053);
    pb.line_to( 9.255130, 40.828792);
    pb.line_to(50.012955, 11.243795);
    pb.line_to(90.744819, 40.864522);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(closed_polygon, 100, 100, "tests/images/fill/polygon.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(75.160671, 88.756136);
    pb.line_to(24.797274, 88.734053);
    pb.line_to( 9.255130, 40.828792);
    pb.line_to(50.012955, 11.243795);
    pb.line_to(90.744819, 40.864522);
    pb.close(); // the only difference
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(winding_star, 100, 100, "tests/images/fill/winding-star.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(50.0,  7.5);
    pb.line_to(75.0, 87.5);
    pb.line_to(10.0, 37.5);
    pb.line_to(90.0, 37.5);
    pb.line_to(25.0, 87.5);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(even_odd_star, 100, 100, "tests/images/fill/even-odd-star.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(50.0,  7.5);
    pb.line_to(75.0, 87.5);
    pb.line_to(10.0, 37.5);
    pb.line_to(90.0, 37.5);
    pb.line_to(25.0, 87.5);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::EvenOdd, Transform::identity(), None);
});

test_raster!(quad_curve, 100, 100, "tests/images/fill/quad.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(10.0, 15.0);
    pb.quad_to(95.0, 35.0, 75.0, 90.0);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::EvenOdd, Transform::identity(), None);
});

test_raster!(cubic_curve, 100, 100, "tests/images/fill/cubic.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(10.0, 15.0);
    pb.cubic_to(95.0, 35.0, 0.0, 75.0, 75.0, 90.0);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::EvenOdd, Transform::identity(), None);
});

test_raster!(memset2d, 100, 100, "tests/images/fill/memset2d.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 255); // Must be opaque to trigger memset2d.
    paint.anti_alias = false;

    let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 90.0, 90.0).unwrap());

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(memset2d_out_of_bounds, 100, 100, "tests/images/fill/memset2d-2.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 255); // Must be opaque to trigger memset2d.
    paint.anti_alias = false;

    let path = PathBuilder::from_rect(Rect::from_ltrb(50.0, 50.0, 120.0, 120.0).unwrap());

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(fill_aa, 100, 100, "tests/images/fill/star-aa.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let mut pb = PathBuilder::new();
    pb.move_to(50.0,  7.5);
    pb.line_to(75.0, 87.5);
    pb.line_to(10.0, 37.5);
    pb.line_to(90.0, 37.5);
    pb.line_to(25.0, 87.5);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::EvenOdd, Transform::identity(), None);
});

#[test]
fn overflow_in_walk_edges_1() {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(10.0, 20.0);
    pb.cubic_to(39.0, 163.0, 117.0, 61.0, 130.0, 70.0);
    let path = pb.finish().unwrap();

    // Must not panic.
    let mut pixmap = Pixmap::new(100, 100).unwrap();
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(100, 100).unwrap();
        pixmap16.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    }
}

test_raster!(clip_line_1, 100, 100, "tests/images/fill/clip-line-1.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(50.0, -15.0);
    pb.line_to(-15.0, 50.0);
    pb.line_to(50.0, 115.0);
    pb.line_to(115.0, 50.0);
    pb.close();
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(clip_line_2, 100, 100, "tests/images/fill/clip-line-2.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    // This strange path forces `line_clipper::clip` to return an empty array.
    // And we're checking that this case is handled correctly.
    let mut pb = PathBuilder::new();
    pb.move_to(0.0, -1.0);
    pb.line_to(50.0, 0.0);
    pb.line_to(0.0, 50.0);
    pb.close();
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(clip_quad, 100, 100, "tests/images/fill/clip-quad.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut pb = PathBuilder::new();
    pb.move_to(10.0, 85.0);
    pb.quad_to(150.0, 150.0, 85.0, 15.0);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(clip_cubic_1, 100, 100, "tests/images/fill/clip-cubic-1.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    // `line_clipper::clip` produces 2 points for this path.
    let mut pb = PathBuilder::new();
    pb.move_to(10.0, 50.0);
    pb.cubic_to(0.0, 175.0, 195.0, 70.0, 75.0, 20.0);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(clip_cubic_2, 100, 100, "tests/images/fill/clip-cubic-2.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    // `line_clipper::clip` produces 3 points for this path.
    let mut pb = PathBuilder::new();
    pb.move_to(10.0, 50.0);
    pb.cubic_to(10.0, 40.0, 90.0, 120.0, 125.0, 20.0);
    let path = pb.finish().unwrap();

    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

#[test]
fn aa_endless_loop() {
    let mut paint = Paint::default();
    paint.anti_alias = true;

    // This path was causing an endless loop before.
    let mut pb = PathBuilder::new();
    pb.move_to(2.1537175, 11.560721);
    pb.quad_to(1.9999998, 10.787931, 2.0, 10.0);
    let path = pb.finish().unwrap();

    // Must not loop.
    let mut pixmap = Pixmap::new(100, 100).unwrap();
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(100, 100).unwrap();
        pixmap16.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    }
}

test_raster!(clear_aa, 100, 100, "tests/images/fill/clear-aa.png", |pixmap| {
    // Make sure that Clear with AA doesn't fallback to memset.
    let mut paint = Paint::default();
    paint.anti_alias = true;
    paint.blend_mode = BlendMode::Clear;

    pixmap.fill(Color::from_rgba8(50, 127, 150, 200));
    pixmap.fill_path(
        &PathBuilder::from_circle(50.0, 50.0, 40.0).unwrap(),
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );
});

#[test]
fn line_curve() {
    let mut paint = Paint::default();
    paint.anti_alias = true;

    let path = {
        let mut pb = PathBuilder::new();
        pb.move_to(100.0, 20.0);
        pb.cubic_to(100.0, 40.0, 100.0, 160.0, 100.0, 180.0); // Just a line.
        pb.finish().unwrap()
    };

    let mut pixmap = Pixmap::new(200, 200).unwrap();
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(200, 200).unwrap();
        pixmap16.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    }
}

test_raster!(vertical_lines_merging_bug, 100, 100, "tests/images/fill/vertical-lines-merging-bug.png", |pixmap| {
    // This path must not trigger edge_builder::combine_vertical,
    // otherwise AlphaRuns::add will crash later.
    let mut pb = PathBuilder::new();
    pb.move_to(765.56, 158.56);
    pb.line_to(754.4, 168.28);
    pb.cubic_to(754.4, 168.28, 754.4, 168.24, 754.4, 168.17);
    pb.cubic_to(754.4, 168.09, 754.4, 168.02, 754.4, 167.95);
    pb.line_to(754.4, 168.06);
    let path = pb.finish().unwrap();

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    // Must not panic.
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::from_row(5.4, 0.0, 0.0, 5.4, -4050.0, -840.0), None);
});

test_raster!(fill_rect, 100, 100, "tests/images/canvas/fill-rect.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    pixmap.fill_rect(
        Rect::from_xywh(20.3, 10.4, 50.5, 30.2).unwrap(),
        &paint,
        Transform::from_row(1.2, 0.3, -0.7, 0.8, 12.0, 15.3),
        None,
    );
});

