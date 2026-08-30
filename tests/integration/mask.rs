use tiny_skia::*;

test_raster!(rect, 100, 100, "tests/images/mask/rect.png", |pixmap| {
    let clip_path = PathBuilder::from_rect(Rect::from_xywh(10.0, 10.0, 80.0, 80.0).unwrap());
    let mut mask = Mask::new(100, 100).unwrap();
    mask.fill_path(&clip_path, FillRule::Winding, false, Transform::default());

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let rect = Rect::from_xywh(0.0, 0.0, 100.0, 100.0).unwrap();
    pixmap.fill_rect(rect, &paint, Transform::identity(), Some(&mask));
});

test_raster!(rect_aa, 100, 100, "tests/images/mask/rect-aa.png", |pixmap| {
    let clip_path = PathBuilder::from_rect(Rect::from_xywh(10.5, 10.0, 80.0, 80.5).unwrap());
    let mut mask = Mask::new(100, 100).unwrap();
    mask.fill_path(&clip_path, FillRule::Winding, true, Transform::default());

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let rect = Rect::from_xywh(0.0, 0.0, 100.0, 100.0).unwrap();
    pixmap.fill_rect(rect, &paint, Transform::identity(), Some(&mask));
});

test_raster!(rect_ts, 100, 100, "tests/images/mask/rect-ts.png", |pixmap| {
    let clip_path = PathBuilder::from_rect(Rect::from_xywh(10.0, 10.0, 80.0, 80.0).unwrap());
    let clip_path = clip_path.transform(Transform::from_row(1.0, -0.3, 0.0, 1.0, 0.0, 15.0)).unwrap();

    let mut mask = Mask::new(100, 100).unwrap();
    mask.fill_path(&clip_path, FillRule::Winding, false, Transform::default());

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let rect = Rect::from_xywh(0.0, 0.0, 100.0, 100.0).unwrap();
    pixmap.fill_rect(rect, &paint, Transform::identity(), Some(&mask));
});

test_raster!(circle_bottom_right_aa, 100, 100, "tests/images/mask/circle-bottom-right-aa.png", |pixmap| {
    let clip_path = PathBuilder::from_circle(100.0, 100.0, 50.0).unwrap();
    let mut mask = Mask::new(100, 100).unwrap();
    mask.fill_path(&clip_path, FillRule::Winding, true, Transform::default());

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let rect = Rect::from_xywh(0.0, 0.0, 100.0, 100.0).unwrap();
    pixmap.fill_rect(rect, &paint, Transform::identity(), Some(&mask));
});

test_raster!(stroke, 100, 100, "tests/images/mask/stroke.png", |pixmap| {
    let clip_path = PathBuilder::from_rect(Rect::from_xywh(10.0, 10.0, 80.0, 80.0).unwrap());
    let mut mask = Mask::new(100, 100).unwrap();
    mask.fill_path(&clip_path, FillRule::Winding, false, Transform::default());

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut stroke = Stroke::default();
    stroke.width = 10.0;

    let path = PathBuilder::from_rect(Rect::from_xywh(10.0, 10.0, 80.0, 80.0).unwrap());
    pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), Some(&mask));
});

// Make sure we're clipping only source and not source and destination
test_raster!(skip_dest, 100, 100, "tests/images/mask/skip-dest.png", |pixmap| {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    pixmap.fill_path(
        &PathBuilder::from_rect(Rect::from_xywh(5.0, 5.0, 60.0, 60.0).unwrap()),
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );

    let mut pixmap2 = PixmapGeneric::new(200, 200).unwrap();
    pixmap2.fill_path(
        &PathBuilder::from_rect(Rect::from_xywh(35.0, 35.0, 60.0, 60.0).unwrap()),
        &paint,
        FillRule::Winding,
        Transform::identity(),
        None,
    );

    let clip_path = PathBuilder::from_rect(Rect::from_xywh(40.0, 40.0, 40.0, 40.0).unwrap());
    let mut mask = Mask::new(100, 100).unwrap();
    mask.fill_path(&clip_path, FillRule::Winding, true, Transform::default());

    pixmap.draw_pixmap(0, 0, pixmap2.as_ref(), &PixmapPaint::default(),
                                Transform::identity(), Some(&mask));
});

test_raster!(intersect_aa, 200, 200, "tests/images/mask/intersect-aa.png", |pixmap| {
    let circle1 = PathBuilder::from_circle(75.0, 75.0, 50.0).unwrap();
    let circle2 = PathBuilder::from_circle(125.0, 125.0, 50.0).unwrap();

    let mut mask = Mask::new(200, 200).unwrap();
    mask.fill_path(&circle1, FillRule::Winding, true, Transform::default());
    mask.intersect_path(&circle2, FillRule::Winding, true, Transform::default());

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    pixmap.fill_rect(
        Rect::from_xywh(0.0, 0.0, 200.0, 200.0).unwrap(),
        &paint,
        Transform::identity(),
        Some(&mask),
    );
});

test_raster!(ignore_memset, 100, 100, "tests/images/mask/ignore-memset.png", |pixmap| {
    let clip_path = PathBuilder::from_rect(Rect::from_xywh(10.0, 10.0, 80.0, 80.0).unwrap());

    let mut mask = Mask::new(100, 100).unwrap();
    mask.fill_path(&clip_path, FillRule::Winding, false, Transform::default());

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 255);
    paint.anti_alias = false;

    pixmap.fill_rect(
        Rect::from_xywh(0.0, 0.0, 100.0, 100.0).unwrap(),
        &paint,
        Transform::identity(),
        Some(&mask),
    );
});

test_raster!(ignore_source, 100, 100, "tests/images/mask/ignore-source.png", |pixmap| {
    let clip_path = PathBuilder::from_rect(Rect::from_xywh(10.0, 10.0, 80.0, 80.0).unwrap());

    let mut mask = Mask::new(100, 100).unwrap();
    mask.fill_path(&clip_path, FillRule::Winding, false, Transform::default());

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 255); // Must be opaque.
    paint.blend_mode = BlendMode::SourceOver;
    paint.anti_alias = false;

    pixmap.fill(Color::WHITE);
    pixmap.fill_rect(
        Rect::from_xywh(0.0, 0.0, 100.0, 100.0).unwrap(),
        &paint,
        Transform::identity(),
        Some(&mask),
    );
});

test_raster!(apply_mask, 100, 100, "tests/images/mask/apply-mask.png", |pixmap| {
    let clip_path = PathBuilder::from_circle(100.0, 100.0, 50.0).unwrap();
    let mut mask = Mask::new(100, 100).unwrap();
    mask.fill_path(&clip_path, FillRule::Winding, true, Transform::default());

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let rect = Rect::from_xywh(0.0, 0.0, 100.0, 100.0).unwrap();
    pixmap.fill_rect(rect, &paint, Transform::identity(), None);
    pixmap.apply_mask(&mask);
});

#[test]
fn mask_from_alpha() {
    let path = PathBuilder::from_circle(100.0, 100.0, 50.0).unwrap();

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let mut pixmap = Pixmap::new(100, 100).unwrap();
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::default(), None);

    let mask = Mask::from_pixmap(pixmap.as_ref(), MaskType::Alpha);

    let expected = Mask::load_png("tests/images/mask/mask-from-alpha.png").unwrap();
    assert_eq!(mask, expected);

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(100, 100).unwrap();
        pixmap16.fill_path(&path, &paint, FillRule::Winding, Transform::default(), None);
        let mask16 = Mask::from_pixmap(pixmap16.as_ref(), MaskType::Alpha);
        crate::common::assert_mask_eq(&mask16, &expected, 1);
    }
}

#[test]
fn mask_from_luma() {
    let path = PathBuilder::from_circle(100.0, 100.0, 50.0).unwrap();

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let mut pixmap = Pixmap::new(100, 100).unwrap();
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::default(), None);

    let mask = Mask::from_pixmap(pixmap.as_ref(), MaskType::Luminance);

    let expected = Mask::load_png("tests/images/mask/mask-from-luma.png").unwrap();
    assert_eq!(mask, expected);

    #[cfg(feature = "16bpc")]
    {
        let mut pixmap16 = PixmapU16::new(100, 100).unwrap();
        pixmap16.fill_path(&path, &paint, FillRule::Winding, Transform::default(), None);
        let mask16 = Mask::from_pixmap(pixmap16.as_ref(), MaskType::Luminance);
        crate::common::assert_mask_eq(&mask16, &expected, 1);
    }
}
