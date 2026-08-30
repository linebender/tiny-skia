use test::Bencher;

#[bench]
fn rect_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let rect = Rect::from_xywh(50.7, 20.1, 812.4, 777.3).unwrap();

    #[cfg(feature = "16bpc")]
    if crate::is_16bpc() {
        let mut pixmap = PixmapU16::new(1000, 1000).unwrap();
        bencher.iter(|| {
            pixmap.fill_rect(rect, &paint, Transform::identity(), None);
        });
        return;
    }

    let mut pixmap = Pixmap::new(1000, 1000).unwrap();

    bencher.iter(|| {
        pixmap.fill_rect(rect, &paint, Transform::identity(), None);
    });
}

#[cfg(feature = "skia-rs")]
#[bench]
fn rect_skia(bencher: &mut Bencher) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let mut paint = Paint::new();
    paint.set_color(50, 127, 150, 200);

    bencher.iter(|| {
        surface.draw_rect(50.7, 20.1, 812.4, 777.3, &paint)
    });
}

#[cfg(feature = "raqote")]
#[bench]
fn rect_raqote(bencher: &mut Bencher) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);

    // raqote uses ARGB order.
    let src = Source::from(Color::new(200, 50, 127, 150));

    let draw_opt = DrawOptions {
        blend_mode: BlendMode::SrcOver,
        alpha: 1.0,
        antialias: AntialiasMode::None,
    };

    bencher.iter(|| {
        dt.fill_rect(50.7, 20.1, 812.4, 777.3, &src, &draw_opt);
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn rect_cairo(bencher: &mut Bencher) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);
    cr.set_source_rgba(50.0 / 255.0, 127.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0);
    cr.rectangle(50.7, 20.1, 812.4, 777.3);

    bencher.iter(|| {
        cr.fill_preserve();
    });
}

#[bench]
fn rect_aa_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let rect = Rect::from_xywh(50.7, 20.1, 812.4, 777.3).unwrap();

    #[cfg(feature = "16bpc")]
    if crate::is_16bpc() {
        let mut pixmap = PixmapU16::new(1000, 1000).unwrap();
        bencher.iter(|| {
            pixmap.fill_rect(rect, &paint, Transform::identity(), None);
        });
        return;
    }

    #[cfg(feature = "16bpc")]
    if crate::is_16bpc() {
        let mut pixmap = PixmapU16::new(1000, 1000).unwrap();
        bencher.iter(|| {
            pixmap.fill_rect(rect, &paint, Transform::identity(), None);
        });
        return;
    }

    let mut pixmap = Pixmap::new(1000, 1000).unwrap();

    bencher.iter(|| {
        pixmap.fill_rect(rect, &paint, Transform::identity(), None);
    });
}

#[cfg(feature = "skia-rs")]
#[bench]
fn rect_aa_skia(bencher: &mut Bencher) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let mut paint = Paint::new();
    paint.set_color(50, 127, 150, 200);
    paint.set_anti_alias(true);

    bencher.iter(|| {
        surface.draw_rect(50.7, 20.1, 812.4, 777.3, &paint)
    });
}

#[cfg(feature = "raqote")]
#[bench]
fn rect_aa_raqote(bencher: &mut Bencher) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);

    // raqote uses ARGB order.
    let src = Source::from(Color::new(200, 50, 127, 150));

    let draw_opt = DrawOptions {
        blend_mode: BlendMode::SrcOver,
        alpha: 1.0,
        antialias: AntialiasMode::Gray,
    };

    bencher.iter(|| {
        dt.fill_rect(50.7, 20.1, 812.4, 777.3, &src, &draw_opt);
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn rect_aa_cairo(bencher: &mut Bencher) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);
    cr.set_source_rgba(50.0 / 255.0, 127.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0);
    cr.set_antialias(Antialias::Subpixel); // TODO: or Gray?
    cr.rectangle(50.7, 20.1, 812.4, 777.3);

    bencher.iter(|| {
        cr.fill_preserve();
    });
}

#[bench]
fn rect_aa_ts_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let rect = Rect::from_xywh(200.3, 100.4, 500.5, 300.2).unwrap();

    #[cfg(feature = "16bpc")]
    if crate::is_16bpc() {
        let mut pixmap = PixmapU16::new(1000, 1000).unwrap();
        bencher.iter(|| {
            pixmap.fill_rect(rect, &paint, Transform::from_row(1.8, 0.3, -0.7, 0.8, 12.0, 15.3), None);
        });
        return;
    }

    let mut pixmap = Pixmap::new(1000, 1000).unwrap();

    bencher.iter(|| {
        pixmap.fill_rect(rect, &paint, Transform::from_row(1.8, 0.3, -0.7, 0.8, 12.0, 15.3), None);
    });
}

#[cfg(feature = "skia-rs")]
#[bench]
fn rect_aa_ts_skia(bencher: &mut Bencher) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let mut paint = Paint::new();
    paint.set_color(50, 127, 150, 200);
    paint.set_anti_alias(true);

    surface.concat(Transform::new(1.8, 0.3, -0.7, 0.8, 12.0, 15.3));

    bencher.iter(|| {
        surface.draw_rect(200.3, 100.4, 500.5, 300.2, &paint)
    });
}

#[cfg(feature = "raqote")]
#[bench]
fn rect_aa_ts_raqote(bencher: &mut Bencher) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);

    // raqote uses ARGB order.
    let src = Source::from(Color::new(200, 50, 127, 150));

    let draw_opt = DrawOptions {
        blend_mode: BlendMode::SrcOver,
        alpha: 1.0,
        antialias: AntialiasMode::Gray,
    };

    dt.set_transform(&Transform::new(1.8, 0.3, -0.7, 0.8, 12.0, 15.3));

    bencher.iter(|| {
        dt.fill_rect(200.3, 100.4, 500.5, 300.2, &src, &draw_opt);
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn rect_aa_ts_cairo(bencher: &mut Bencher) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);
    cr.set_source_rgba(50.0 / 255.0, 127.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0);
    cr.set_antialias(Antialias::Subpixel); // TODO: or Gray?
    cr.transform(Matrix::new(1.8, 0.3, -0.7, 0.8, 12.0, 15.3));
    cr.rectangle(200.3, 100.4, 500.5, 300.2);

    bencher.iter(|| {
        cr.fill_preserve();
    });
}

#[bench]
fn all_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;

    let c = Color::from_rgba8(50, 100, 150, 200);

    #[cfg(feature = "16bpc")]
    if crate::is_16bpc() {
        let mut pixmap = PixmapU16::new(1000, 1000).unwrap();
        bencher.iter(|| {
            pixmap.fill(c);
        });
        return;
    }

    let mut pixmap = Pixmap::new(1000, 1000).unwrap();
    bencher.iter(|| {
        pixmap.fill(c);
    });
}

#[cfg(feature = "skia-rs")]
#[bench]
fn all_skia(bencher: &mut Bencher) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();
    let r = 50.0 / 255.0;
    let g = 100.0 / 255.0;
    let b = 150.0 / 255.0;
    let a = 200.0 / 255.0;
    bencher.iter(|| {
        surface.draw_color(r, g, b, a); // TODO: is there a faster way?
    });
}

#[cfg(feature = "raqote")]
#[bench]
fn all_raqote(bencher: &mut Bencher) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);
    bencher.iter(|| {
        dt.clear(SolidSource { r: 50, g: 100, b: 150, a: 200 });
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn all_cairo(bencher: &mut Bencher) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);
    cr.set_source_rgba(50.0 / 255.0, 127.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0);

    bencher.iter(|| {
        cr.paint(); // TODO: is there a faster way?
    });
}

#[bench]
fn path_aa_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let mut pb = PathBuilder::new();
    pb.move_to(500.0, 20.0);
    pb.cubic_to(650.0, 320.0, 770.0, 650.0, 800.0, 980.0);
    pb.line_to(20.0, 380.0);
    pb.line_to(200.0, 980.0);
    pb.cubic_to(230.0, 650.0, 350.0, 320.0, 500.0, 20.0);
    pb.close();
    let path = pb.finish().unwrap();

    #[cfg(feature = "16bpc")]
    if crate::is_16bpc() {
        let mut pixmap = PixmapU16::new(1000, 1000).unwrap();
        bencher.iter(|| {
            pixmap.fill_path(&path, &paint, FillRule::EvenOdd, Transform::identity(), None);
        });
        return;
    }

    let mut pixmap = Pixmap::new(1000, 1000).unwrap();

    bencher.iter(|| {
        pixmap.fill_path(&path, &paint, FillRule::EvenOdd, Transform::identity(), None);
    });
}

#[cfg(feature = "skia-rs")]
#[bench]
fn path_aa_skia(bencher: &mut Bencher) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let mut paint = Paint::new();
    paint.set_color(50, 127, 150, 200);
    paint.set_blend_mode(BlendMode::SourceOver);
    paint.set_anti_alias(true);

    let mut path = Path::new();
    path.move_to(500.0, 20.0);
    path.cubic_to(650.0, 320.0, 770.0, 650.0, 800.0, 980.0);
    path.line_to(20.0, 380.0);
    path.line_to(200.0, 980.0);
    path.cubic_to(230.0, 650.0, 350.0, 320.0, 500.0, 20.0);
    path.close();
    path.set_fill_type(FillType::EvenOdd);

    bencher.iter(|| {
        surface.draw_path(&path, &paint);
    });
}

#[cfg(feature = "raqote")]
#[bench]
fn path_aa_raqote(bencher: &mut Bencher) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);

    let mut path = {
        let mut pb = PathBuilder::new();
        pb.move_to(500.0, 20.0);
        pb.cubic_to(650.0, 320.0, 770.0, 650.0, 800.0, 980.0);
        pb.line_to(20.0, 380.0);
        pb.line_to(200.0, 980.0);
        pb.cubic_to(230.0, 650.0, 350.0, 320.0, 500.0, 20.0);
        pb.close();
        pb.finish()
    };
    path.winding = Winding::EvenOdd;

    // raqote uses ARGB order.
    let src = Source::from(Color::new(200, 50, 127, 150));

    let draw_opt = DrawOptions {
        blend_mode: BlendMode::SrcOver,
        alpha: 1.0,
        antialias: AntialiasMode::Gray,
    };

    bencher.iter(|| {
        dt.fill(&path, &src, &draw_opt);
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn path_aa_cairo(bencher: &mut Bencher) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);

    cr.move_to(500.0, 20.0);
    cr.curve_to(650.0, 320.0, 770.0, 650.0, 800.0, 980.0);
    cr.line_to(20.0, 380.0);
    cr.line_to(200.0, 980.0);
    cr.curve_to(230.0, 650.0, 350.0, 320.0, 500.0, 20.0);
    cr.close_path();

    cr.set_source_rgba(50.0 / 255.0, 127.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0);
    cr.set_antialias(Antialias::Subpixel); // TODO: or Gray?
    cr.set_fill_rule(FillRule::EvenOdd);
    cr.set_operator(Operator::Over);

    bencher.iter(|| {
        cr.fill_preserve();
    });
}

// Filling a semi-transparent rectangle path with a Source blending mode.
// By using this blending mode we're forcing a simple pixels overwrite.
#[bench]
fn source_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.blend_mode = BlendMode::Source;
    paint.anti_alias = false;

    let path = PathBuilder::from_rect(Rect::from_ltrb(100.0, 100.0, 900.0, 900.0).unwrap());

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

#[cfg(feature = "skia-rs")]
#[bench]
fn source_skia(bencher: &mut Bencher) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let mut paint = Paint::new();
    paint.set_color(50, 127, 150, 200);
    paint.set_blend_mode(BlendMode::Source);

    let mut path = Path::new();
    path.push_rect(100.0, 100.0, 900.0, 900.0);

    bencher.iter(|| {
        surface.draw_path(&path, &paint);
    });
}

#[cfg(feature = "raqote")]
#[bench]
fn source_raqote(bencher: &mut Bencher) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);

    let path = {
        let mut pb = PathBuilder::new();
        pb.rect(100.0, 100.0, 800.0, 800.0);
        pb.finish()
    };

    // raqote uses ARGB order.
    let src = Source::from(Color::new(200, 50, 127, 150));

    let draw_opt = DrawOptions {
        blend_mode: BlendMode::Src,
        alpha: 1.0,
        antialias: AntialiasMode::None,
    };

    bencher.iter(|| {
        dt.fill(&path, &src, &draw_opt);
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn source_cairo(bencher: &mut Bencher) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);

    cr.rectangle(100.0, 100.0, 800.0, 800.0);

    cr.set_source_rgba(50.0 / 255.0, 127.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0);
    cr.set_antialias(Antialias::None);
    cr.set_fill_rule(FillRule::Winding);
    cr.set_operator(Operator::Source);

    bencher.iter(|| {
        cr.fill_preserve();
    });
}


// Filling an opaque rectangle path.
// A chosen blending mode doesn't really matter in this case,
// since we are simply overwriting the pixels.
#[bench]
fn opaque_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 255);
    paint.anti_alias = false;

    let path = PathBuilder::from_rect(Rect::from_ltrb(100.0, 100.0, 900.0, 900.0).unwrap());

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

#[cfg(feature = "skia-rs")]
#[bench]
fn opaque_skia(bencher: &mut Bencher) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let mut paint = Paint::new();
    paint.set_color(50, 127, 150, 255);
    paint.set_blend_mode(BlendMode::SourceOver);

    let mut path = Path::new();
    path.push_rect(100.0, 100.0, 900.0, 900.0);

    bencher.iter(|| {
        surface.draw_path(&path, &paint);
    });
}

#[cfg(feature = "raqote")]
#[bench]
fn opaque_raqote(bencher: &mut Bencher) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);

    let path = {
        let mut pb = PathBuilder::new();
        pb.rect(100.0, 100.0, 800.0, 800.0);
        pb.finish()
    };

    // raqote uses ARGB order.
    let src = Source::from(Color::new(255, 50, 127, 150));

    let draw_opt = DrawOptions {
        blend_mode: BlendMode::SrcOver,
        alpha: 1.0,
        antialias: AntialiasMode::None,
    };

    bencher.iter(|| {
        dt.fill(&path, &src, &draw_opt);
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn opaque_cairo(bencher: &mut Bencher) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);

    cr.rectangle(100.0, 100.0, 800.0, 800.0);

    cr.set_source_rgba(50.0 / 255.0, 127.0 / 255.0, 150.0 / 255.0, 255.0 / 255.0);
    cr.set_antialias(Antialias::None);
    cr.set_fill_rule(FillRule::Winding);
    cr.set_operator(Operator::Over);

    bencher.iter(|| {
        cr.fill_preserve();
    });
}
