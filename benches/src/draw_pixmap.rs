use test::Bencher;

fn bench_draw_pixmap(blend_mode: tiny_skia::BlendMode, bencher: &mut Bencher) {
    use tiny_skia::*;

    let is_16bpc = crate::is_16bpc();

    #[cfg(feature = "16bpc")]
    if is_16bpc {
        let mut dst = PixmapU16::new(1000, 1000).unwrap();
        let mut src = PixmapU16::new(800, 800).unwrap();
        src.fill(Color::from_rgba8(160, 80, 40, 200));

        let paint = PixmapPaint {
            opacity: 0.85,
            blend_mode,
            quality: FilterQuality::Nearest,
        };

        bencher.iter(|| {
            dst.draw_pixmap(100, 100, src.as_ref(), &paint, Transform::identity(), None);
        });
        return;
    }

    let mut dst = Pixmap::new(1000, 1000).unwrap();
    let mut src = Pixmap::new(800, 800).unwrap();
    src.fill(Color::from_rgba8(160, 80, 40, 200));

    let paint = PixmapPaint {
        opacity: 0.85,
        blend_mode,
        quality: FilterQuality::Nearest,
    };

    bencher.iter(|| {
        dst.draw_pixmap(100, 100, src.as_ref(), &paint, Transform::identity(), None);
    });
}

#[bench] fn draw_pixmap_source_over(bencher: &mut Bencher)  { bench_draw_pixmap(tiny_skia::BlendMode::SourceOver, bencher); }
#[bench] fn draw_pixmap_multiply(bencher: &mut Bencher)     { bench_draw_pixmap(tiny_skia::BlendMode::Multiply, bencher); }
#[bench] fn draw_pixmap_screen(bencher: &mut Bencher)       { bench_draw_pixmap(tiny_skia::BlendMode::Screen, bencher); }
#[bench] fn draw_pixmap_overlay(bencher: &mut Bencher)      { bench_draw_pixmap(tiny_skia::BlendMode::Overlay, bencher); }
#[bench] fn draw_pixmap_color_dodge(bencher: &mut Bencher)  { bench_draw_pixmap(tiny_skia::BlendMode::ColorDodge, bencher); }
#[bench] fn draw_pixmap_color_burn(bencher: &mut Bencher)   { bench_draw_pixmap(tiny_skia::BlendMode::ColorBurn, bencher); }
#[bench] fn draw_pixmap_soft_light(bencher: &mut Bencher)   { bench_draw_pixmap(tiny_skia::BlendMode::SoftLight, bencher); }
#[bench] fn draw_pixmap_saturation(bencher: &mut Bencher)   { bench_draw_pixmap(tiny_skia::BlendMode::Saturation, bencher); }
#[bench] fn draw_pixmap_color(bencher: &mut Bencher)        { bench_draw_pixmap(tiny_skia::BlendMode::Color, bencher); }
