use test::Bencher;

fn pattern_tiny_skia(
    quality: tiny_skia::FilterQuality,
    ts: tiny_skia::Transform,
    bencher: &mut Bencher,
) {
    use tiny_skia::*;

    fn crate_triangle() -> Pixmap {
        let mut paint = Paint::default();
        paint.set_color_rgba8(50, 127, 150, 200);
        paint.anti_alias = true;

        let mut pb = PathBuilder::new();
        pb.move_to(0.0, 20.0);
        pb.line_to(20.0, 20.0);
        pb.line_to(10.0, 0.0);
        pb.close();
        let path = pb.finish().unwrap();

        let mut pixmap = Pixmap::new(20, 20).unwrap();
        pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
        pixmap
    }

    let triangle = crate_triangle();

    let mut paint = Paint::default();
    paint.anti_alias = true;
    paint.shader = Pattern::new(
        triangle.as_ref(),
        SpreadMode::Repeat,
        quality,
        1.0,
        ts,
    );

    let mut pb = PathBuilder::new();
    pb.move_to(60.0, 60.0);
    pb.line_to(160.0, 940.0);
    pb.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
    pb.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
    pb.close();
    let path = pb.finish().unwrap();

    #[cfg(feature = "16bpc")]
    if crate::is_16bpc() {
        let mut pixmap = PixmapU16::new(1000, 1000).unwrap();
        bencher.iter(|| {
            pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
        });
        return;
    }

    let mut pixmap = Pixmap::new(1000, 1000).unwrap();

    bencher.iter(|| {
        pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    });
}

#[bench]
fn plain_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;
    pattern_tiny_skia(
        FilterQuality::Nearest,
        Transform::identity(),
        bencher,
    )
}

#[bench]
fn lq_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;
    pattern_tiny_skia(
        FilterQuality::Bilinear,
        Transform::from_row(1.5, -0.4, 0.0, -0.8, 5.0, 1.0),
        bencher,
    )
}

#[bench]
fn hq_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;
    pattern_tiny_skia(
        FilterQuality::Bicubic,
        Transform::from_row(1.5, -0.4, 0.0, -0.8, 5.0, 1.0),
        bencher,
    )
}

#[cfg(feature = "skia-rs")]
fn pattern_skia(
    quality: skia_rs::FilterQuality,
    ts: skia_rs::Transform,
    bencher: &mut Bencher,
) {
    use skia_rs::*;

    fn crate_triangle() -> Surface {
        let mut surface = Surface::new_rgba_premultiplied(20, 20).unwrap();

        let mut paint = Paint::new();
        paint.set_style(PaintStyle::Fill);
        paint.set_anti_alias(true);
        paint.set_color(50, 127, 150, 200);

        let mut path = Path::new();
        path.move_to(0.0, 20.0);
        path.line_to(20.0, 20.0);
        path.line_to(10.0, 0.0);
        path.close();
        surface.draw_path(&path, &paint);

        surface
    }

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let triangle = crate_triangle();
    let shader = Shader::new_from_surface_image(
        &triangle,
        ts,
        quality,
    ).unwrap();

    let mut paint = Paint::new();
    paint.set_style(PaintStyle::Fill);
    paint.set_anti_alias(true);
    paint.set_shader(&shader);

    let mut path = Path::new();
    path.move_to(60.0, 60.0);
    path.line_to(160.0, 940.0);
    path.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
    path.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
    path.close();

    bencher.iter(|| {
        surface.draw_path(&path, &paint);
    });
}

#[cfg(feature = "skia-rs")]
#[bench]
fn plain_skia(bencher: &mut Bencher) {
    use skia_rs::*;
    pattern_skia(
        FilterQuality::None,
        Transform::default(),
        bencher,
    )
}

#[cfg(feature = "skia-rs")]
#[bench]
fn lq_skia(bencher: &mut Bencher) {
    use skia_rs::*;
    pattern_skia(
        FilterQuality::Low,
        Transform::new(1.5, -0.4, 0.0, -0.8, 5.0, 1.0),
        bencher,
    )
}

#[cfg(feature = "skia-rs")]
#[bench]
fn hq_skia(bencher: &mut Bencher) {
    use skia_rs::*;
    pattern_skia(
        FilterQuality::High,
        Transform::new(1.5, -0.4, 0.0, -0.8, 5.0, 1.0),
        bencher,
    )
}

#[cfg(feature = "cairo-rs")]
fn pattern_cairo(
    quality: cairo::Filter,
    ts: cairo::Matrix,
    bencher: &mut Bencher,
) {
    use cairo::*;

    fn crate_triangle() -> ImageSurface {
        let surface = ImageSurface::create(Format::ARgb32, 20, 20).unwrap();

        let cr = cairo::Context::new(&surface);
        cr.set_source_rgba(50.0 / 255.0, 127.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0);
        cr.set_antialias(cairo::Antialias::Subpixel);

        cr.move_to(0.0, 20.0);
        cr.line_to(20.0, 20.0);
        cr.line_to(10.0, 0.0);
        cr.close_path();

        cr.fill();

        surface
    }

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let triangle = crate_triangle();

    let cr = cairo::Context::new(&surface);

    let patt = cairo::SurfacePattern::create(&triangle);
    patt.set_extend(cairo::Extend::Repeat);
    patt.set_filter(quality);

    let mut m = ts.clone();
    m.invert();
    patt.set_matrix(m);

    cr.set_source(&patt);
    cr.set_antialias(cairo::Antialias::Subpixel);

    cr.move_to(60.0, 60.0);
    cr.line_to(160.0, 940.0);
    cr.curve_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
    cr.curve_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
    cr.close_path();

    bencher.iter(|| {
        cr.fill_preserve();
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn plain_cairo(bencher: &mut Bencher) {
    use cairo::*;
    pattern_cairo(
        Filter::Nearest,
        Matrix::default(),
        bencher,
    )
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn lq_cairo(bencher: &mut Bencher) {
    use cairo::*;
    pattern_cairo(
        Filter::Bilinear,
        Matrix::new(1.5, -0.4, 0.0, -0.8, 5.0, 1.0),
        bencher,
    )
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn hq_cairo(bencher: &mut Bencher) {
    use cairo::*;
    pattern_cairo(
        // Looks like in cairo, the best filter is Gaussian, while in Skia it's Bicubic.
        Filter::Best,
        Matrix::new(1.5, -0.4, 0.0, -0.8, 5.0, 1.0),
        bencher,
    )
}

#[cfg(feature = "raqote")]
fn pattern_raqote(
    quality: raqote::FilterMode,
    ts: raqote::Transform,
    bencher: &mut Bencher,
) {
    use raqote::*;

    fn crate_triangle() -> DrawTarget {
        let mut dt = DrawTarget::new(20, 20);

        let mut pb = PathBuilder::new();
        pb.move_to(0.0, 20.0);
        pb.line_to(20.0, 20.0);
        pb.line_to(10.0, 00.0);
        pb.close();
        let path = pb.finish();

        // raqote uses ARGB order.
        let src = Source::from(Color::new(200, 50, 127, 150));

        let draw_opt = DrawOptions {
            blend_mode: BlendMode::SrcOver,
            alpha: 1.0,
            antialias: AntialiasMode::Gray,
        };

        dt.fill(&path, &src, &draw_opt);

        dt
    }

    let mut dt = DrawTarget::new(1000, 1000);

    let triangle = crate_triangle();

    let mut pb = PathBuilder::new();
    pb.move_to(60.0, 60.0);
    pb.line_to(160.0, 940.0);
    pb.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
    pb.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
    pb.close();
    let path = pb.finish();

    let image = raqote::Image {
        width: triangle.width() as i32,
        height: triangle.height() as i32,
        data: triangle.get_data(),
    };

    let src = raqote::Source::Image(
        image,
        ExtendMode::Repeat,
        quality,
        ts.inverse().unwrap(),
    );

    let draw_opt = DrawOptions {
        blend_mode: BlendMode::SrcOver,
        alpha: 1.0,
        antialias: AntialiasMode::Gray,
    };

    bencher.iter(|| {
        dt.fill(&path, &src, &draw_opt);
    });
}

#[cfg(feature = "raqote")]
#[bench]
fn plain_raqote(bencher: &mut Bencher) {
    use raqote::*;
    pattern_raqote(
        FilterMode::Nearest,
        Transform::default(),
        bencher,
    )
}

#[cfg(feature = "raqote")]
#[bench]
fn lq_raqote(bencher: &mut Bencher) {
    use raqote::*;
    pattern_raqote(
        FilterMode::Bilinear,
        Transform::new(1.5, -0.4, 0.0, -0.8, 5.0, 1.0),
        bencher,
    )
}

#[cfg(feature = "raqote")]
#[bench]
fn hq_raqote(_bencher: &mut Bencher) {
    // unsupported
}
