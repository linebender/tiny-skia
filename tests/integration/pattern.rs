use tiny_skia::*;

fn create_triangle<P: HighPixel>(_target: &PixmapGeneric<P>) -> PixmapGeneric<P> {
    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let mut pb = PathBuilder::new();
    pb.move_to(0.0, 20.0);
    pb.line_to(20.0, 20.0);
    pb.line_to(10.0, 0.0);
    pb.close();
    let path = pb.finish().unwrap();

    let mut pixmap = PixmapGeneric::new(20, 20).unwrap();
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    pixmap
}

test_raster!(pad_nearest, 200, 200, "tests/images/pattern/pad-nearest.png", |pixmap| {
    let triangle = create_triangle(&pixmap);

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = Pattern::from_pixmap(
        triangle.as_ref(),
        SpreadMode::Pad,
        FilterQuality::Nearest,
        1.0,
        Transform::identity(),
    );

    let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 190.0, 190.0).unwrap());
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(repeat_nearest, 200, 200, "tests/images/pattern/repeat-nearest.png", |pixmap| {
    let triangle = create_triangle(&pixmap);

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = Pattern::from_pixmap(
        triangle.as_ref(),
        SpreadMode::Repeat,
        FilterQuality::Nearest,
        1.0,
        Transform::identity(),
    );

    let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 190.0, 190.0).unwrap());
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(reflect_nearest, 200, 200, "tests/images/pattern/reflect-nearest.png", |pixmap| {
    let triangle = create_triangle(&pixmap);

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = Pattern::from_pixmap(
        triangle.as_ref(),
        SpreadMode::Reflect,
        FilterQuality::Nearest,
        1.0,
        Transform::identity(),
    );

    let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 190.0, 190.0).unwrap());
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(pad_bicubic, 200, 200, "tests/images/pattern/pad-bicubic.png", |pixmap| {
    let triangle = create_triangle(&pixmap);

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = Pattern::from_pixmap(
        triangle.as_ref(),
        SpreadMode::Pad,
        FilterQuality::Bicubic,
        1.0,
        Transform::from_row(1.1, 0.3, 0.0, 1.4, 0.0, 0.0),
    );

    let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 190.0, 190.0).unwrap());
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(repeat_bicubic, 200, 200, "tests/images/pattern/repeat-bicubic.png", |pixmap| {
    let triangle = create_triangle(&pixmap);

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = Pattern::from_pixmap(
        triangle.as_ref(),
        SpreadMode::Repeat,
        FilterQuality::Bicubic,
        1.0,
        Transform::from_row(1.1, 0.3, 0.0, 1.4, 0.0, 0.0),
    );

    let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 190.0, 190.0).unwrap());
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(reflect_bicubic, 200, 200, "tests/images/pattern/reflect-bicubic.png", |pixmap| {
    let triangle = create_triangle(&pixmap);

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = Pattern::from_pixmap(
        triangle.as_ref(),
        SpreadMode::Reflect,
        FilterQuality::Bicubic,
        1.0,
        Transform::from_row(1.1, 0.3, 0.0, 1.4, 0.0, 0.0),
    );

    let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 190.0, 190.0).unwrap());
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(filter_nearest_no_ts, 200, 200, "tests/images/pattern/filter-nearest-no-ts.png", |pixmap| {
    let triangle = create_triangle(&pixmap);

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = Pattern::from_pixmap(
        triangle.as_ref(),
        SpreadMode::Repeat,
        FilterQuality::Nearest,
        1.0,
        Transform::identity(),
    );

    let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 190.0, 190.0).unwrap());
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(filter_nearest, 200, 200, "tests/images/pattern/filter-nearest.png", |pixmap| {
    let triangle = create_triangle(&pixmap);

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = Pattern::from_pixmap(
        triangle.as_ref(),
        SpreadMode::Repeat,
        FilterQuality::Nearest,
        1.0,
        Transform::from_row(1.5, 0.0, -0.4, -0.8, 5.0, 1.0),
    );

    let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 190.0, 190.0).unwrap());
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(filter_bilinear, 200, 200, "tests/images/pattern/filter-bilinear.png", |pixmap| {
    let triangle = create_triangle(&pixmap);

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = Pattern::from_pixmap(
        triangle.as_ref(),
        SpreadMode::Repeat,
        FilterQuality::Bilinear,
        1.0,
        Transform::from_row(1.5, 0.0, -0.4, -0.8, 5.0, 1.0),
    );

    let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 190.0, 190.0).unwrap());
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});

test_raster!(filter_bicubic, 200, 200, "tests/images/pattern/filter-bicubic.png", |pixmap| {
    let triangle = create_triangle(&pixmap);

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = Pattern::from_pixmap(
        triangle.as_ref(),
        SpreadMode::Repeat,
        FilterQuality::Bicubic,
        1.0,
        Transform::from_row(1.5, 0.0, -0.4, -0.8, 5.0, 1.0),
    );

    let path = PathBuilder::from_rect(Rect::from_ltrb(10.0, 10.0, 190.0, 190.0).unwrap());
    pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
});
