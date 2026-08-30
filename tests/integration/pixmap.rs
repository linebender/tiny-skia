use tiny_skia::*;

#[test]
fn clone_rect_1() {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let expected = Pixmap::load_png("tests/images/pixmap/clone-rect-1.png").unwrap();

    let mut pixmap = Pixmap::new(200, 200).unwrap();
    pixmap.fill_path(
        &PathBuilder::from_circle(100.0, 100.0, 80.0).unwrap(),
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );

    let part = pixmap.as_ref().clone_rect(IntRect::from_xywh(10, 15, 80, 90).unwrap()).unwrap();
    crate::common::assert_pixmap_eq(&part, &expected, 0);

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(200, 200).unwrap();
        pixmap16.fill_path(
            &PathBuilder::from_circle(100.0, 100.0, 80.0).unwrap(),
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );

        let part16 = pixmap16.as_ref().clone_rect(IntRect::from_xywh(10, 15, 80, 90).unwrap()).unwrap();
        crate::common::assert_pixmap_eq(&part16, &expected, 2);
    }
}

#[test]
fn clone_rect_2() {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let expected = Pixmap::load_png("tests/images/pixmap/clone-rect-2.png").unwrap();

    let mut pixmap = Pixmap::new(200, 200).unwrap();
    pixmap.fill_path(
        &PathBuilder::from_circle(100.0, 100.0, 80.0).unwrap(),
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );

    let part = pixmap.as_ref().clone_rect(IntRect::from_xywh(130, 120, 80, 90).unwrap()).unwrap();
    crate::common::assert_pixmap_eq(&part, &expected, 0);

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(200, 200).unwrap();
        pixmap16.fill_path(
            &PathBuilder::from_circle(100.0, 100.0, 80.0).unwrap(),
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );

        let part16 = pixmap16.as_ref().clone_rect(IntRect::from_xywh(130, 120, 80, 90).unwrap()).unwrap();
        crate::common::assert_pixmap_eq(&part16, &expected, 2);
    }
}

#[test]
fn clone_rect_out_of_bound() {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let mut pixmap = Pixmap::new(200, 200).unwrap();
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

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(200, 200).unwrap();
        pixmap16.fill_path(
            &PathBuilder::from_circle(100.0, 100.0, 80.0).unwrap(),
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );

        assert!(pixmap16.as_ref().clone_rect(IntRect::from_xywh(250, 15, 80, 90).unwrap()).is_none());
        assert!(pixmap16.as_ref().clone_rect(IntRect::from_xywh(10, 250, 80, 90).unwrap()).is_none());
        assert!(pixmap16.as_ref().clone_rect(IntRect::from_xywh(10, -250, 80, 90).unwrap()).is_none());
    }
}

#[test]
fn fill() {
    let c = Color::from_rgba8(50, 100, 150, 200);
    let mut pixmap = Pixmap::new(10, 10).unwrap();
    pixmap.fill(c);
    assert_eq!(pixmap.pixel(1, 1).unwrap(), c.premultiply().to_color_u8());

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(10, 10).unwrap();
        pixmap16.fill(c);
        assert_eq!(pixmap16.pixel(1, 1).unwrap(), c.premultiply().to_color_u16());
    }
}

test_raster!(draw_pixmap, 200, 200, "tests/images/canvas/draw-pixmap.png", |pixmap| {
    // A pixmap with the bottom half filled with solid color.
    let sub_pixmap = {
        let mut paint = Paint::default();
        paint.set_color_rgba8(50, 127, 150, 200);
        paint.anti_alias = false;

        let rect = Rect::from_xywh(0.0, 50.0, 100.0, 50.0).unwrap();

        let mut sub = PixmapGeneric::new(100, 100).unwrap();
        sub.fill_rect(rect, &paint, Transform::identity(), None);
        sub
    };

    let mut paint = PixmapPaint::default();
    paint.quality = FilterQuality::Bicubic;

    pixmap.draw_pixmap(20, 20, sub_pixmap.as_ref(), &paint, Transform::identity(), None);
});

test_raster!(draw_pixmap_ts, 200, 200, "tests/images/canvas/draw-pixmap-ts.png", |pixmap| {
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

        let mut sub = PixmapGeneric::new(100, 100).unwrap();
        sub.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
        sub
    };

    let mut paint = PixmapPaint::default();
    paint.quality = FilterQuality::Bicubic;

    pixmap.draw_pixmap(
        5, 10,
        triangle.as_ref(),
        &paint,
        Transform::from_row(1.2, 0.5, 0.5, 1.2, 0.0, 0.0),
        None,
    );
});

test_raster!(draw_pixmap_opacity, 200, 200, "tests/images/canvas/draw-pixmap-opacity.png", |pixmap| {
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

        let mut sub = PixmapGeneric::new(100, 100).unwrap();
        sub.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
        sub
    };

    let mut paint = PixmapPaint::default();
    paint.quality = FilterQuality::Bicubic;
    paint.opacity = 0.5;

    pixmap.draw_pixmap(
        5, 10,
        triangle.as_ref(),
        &paint,
        Transform::from_row(1.2, 0.5, 0.5, 1.2, 0.0, 0.0),
        None,
    );
});
