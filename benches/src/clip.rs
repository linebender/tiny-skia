use test::Bencher;

fn do_clip_tiny_skia(aa: bool, bencher: &mut Bencher) {
    use tiny_skia::*;

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let path = PathBuilder::from_rect(Rect::from_xywh(0.0, 0.0, 1000.0, 1000.0).unwrap());

    #[cfg(feature = "16bpc")]
    if crate::is_16bpc() {
        let mut pixmap = PixmapU16::new(1000, 1000).unwrap();
        bencher.iter(|| {
            let clip_path = {
                let mut pb = PathBuilder::new();
                pb.push_rect(Rect::from_xywh(100.0, 100.0, 800.0, 800.0).unwrap());
                pb.push_rect(Rect::from_xywh(300.0, 300.0, 400.0, 400.0).unwrap());
                pb.finish().unwrap()
            };

            let clip_path = clip_path.transform(Transform::from_row(1.0, -0.5, 0.0, 1.0, 0.0, 300.0)).unwrap();

            let mut mask = Mask::new(1000, 1000).unwrap();
            mask.fill_path(&clip_path, FillRule::EvenOdd, aa, Transform::identity());

            pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), Some(&mask));
        });
        return;
    }

    let mut pixmap = Pixmap::new(1000, 1000).unwrap();
    bencher.iter(|| {
        let clip_path = {
            let mut pb = PathBuilder::new();
            pb.push_rect(Rect::from_xywh(100.0, 100.0, 800.0, 800.0).unwrap());
            pb.push_rect(Rect::from_xywh(300.0, 300.0, 400.0, 400.0).unwrap());
            pb.finish().unwrap()
        };

        let clip_path = clip_path.transform(Transform::from_row(1.0, -0.5, 0.0, 1.0, 0.0, 300.0)).unwrap();

        let mut mask = Mask::new(1000, 1000).unwrap();
        mask.fill_path(&clip_path, FillRule::EvenOdd, aa, Transform::identity());

        // Do not use fill_rect, because it is very slow by itself.
        pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), Some(&mask));
    });
}

#[bench]
fn tiny_skia(bencher: &mut Bencher) {
    do_clip_tiny_skia(false, bencher);
}

#[bench]
fn aa_tiny_skia(bencher: &mut Bencher) {
    do_clip_tiny_skia(true, bencher);
}

#[cfg(feature = "skia-rs")]
fn do_clip_path_skia(aa: bool, bencher: &mut Bencher) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let mut paint = Paint::new();
    paint.set_color(50, 127, 150, 200);

    let mut clip_path = Path::new();
    clip_path.push_rect(100.0, 100.0, 800.0, 800.0);
    clip_path.push_rect(300.0, 300.0, 400.0, 400.0);
    clip_path.set_fill_type(FillType::EvenOdd);

    let mut path = Path::new();
    path.push_rect(0.0, 0.0, 1000.0, 1000.0);

    bencher.iter(|| {
        surface.save();
        surface.set_transform(Transform::new(1.0, -0.3, 0.0, 1.0, 0.0, 150.0));
        surface.set_clip_path(&clip_path, aa);
        surface.reset_transform();
        // Do not use draw_rect, because it is very slow by itself.
        surface.draw_path(&path, &paint);
        surface.restore(); // acts as clip reset
    });
}

#[cfg(feature = "skia-rs")]
#[bench]
fn skia(bencher: &mut Bencher) {
    do_clip_path_skia(false, bencher);
}

#[cfg(feature = "skia-rs")]
#[bench]
fn aa_skia(bencher: &mut Bencher) {
    do_clip_path_skia(true, bencher);
}

#[cfg(feature = "raqote")]
fn do_clip_path_raqote(aa: raqote::AntialiasMode, bencher: &mut Bencher) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);

    let mut clip_path = {
        let mut pb = PathBuilder::new();
        pb.rect(100.0, 100.0, 800.0, 800.0);
        pb.rect(300.0, 300.0, 400.0, 400.0);
        pb.finish()
    };
    clip_path.winding = Winding::EvenOdd;

    // raqote uses ARGB order.
    let src = Source::from(Color::new(200, 50, 127, 150));

    let draw_opt = DrawOptions {
        blend_mode: BlendMode::SrcOver,
        alpha: 1.0,
        antialias: aa,
    };

    bencher.iter(|| {
        dt.set_transform(&Transform::new(1.0, -0.3, 0.0, 1.0, 0.0, 150.0));
        dt.push_clip(&clip_path);
        dt.set_transform(&Transform::default());
        dt.fill_rect(0.0, 0.0, 1000.0, 1000.0, &src, &draw_opt);
        dt.pop_clip();
    });
}

#[cfg(feature = "raqote")]
#[bench]
fn raqote(bencher: &mut Bencher) {
    do_clip_path_raqote(raqote::AntialiasMode::None, bencher);
}

#[cfg(feature = "raqote")]
#[bench]
fn aa_raqote(bencher: &mut Bencher) {
    do_clip_path_raqote(raqote::AntialiasMode::Gray, bencher);
}

#[cfg(feature = "cairo-rs")]
fn do_clip_path_cairo(aa: cairo::Antialias, bencher: &mut Bencher) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);
    cr.set_source_rgba(50.0 / 255.0, 127.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0);

    bencher.iter(|| {
        cr.set_matrix(Matrix::new(1.0, -0.3, 0.0, 1.0, 0.0, 150.0));
        cr.set_antialias(aa);
        cr.rectangle(100.0, 100.0, 800.0, 800.0);
        cr.rectangle(300.0, 300.0, 400.0, 400.0);
        cr.set_fill_rule(FillRule::EvenOdd);
        cr.clip();

        cr.identity_matrix();
        cr.rectangle(0.0, 0.0, 1000.0, 1000.0);
        cr.set_antialias(cairo::Antialias::None);
        cr.fill();
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn cairo(bencher: &mut Bencher) {
    do_clip_path_cairo(cairo::Antialias::None, bencher);
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn aa_cairo(bencher: &mut Bencher) {
    do_clip_path_cairo(cairo::Antialias::Subpixel, bencher);
}
