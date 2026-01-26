use test::Bencher;

fn two_stops_linear_tiny_skia(
    hq: bool,
    points: Vec<tiny_skia::GradientStop>,
    mode: tiny_skia::SpreadMode,
    bencher: &mut Bencher,
) {
    use tiny_skia::*;

    let mut paint = Paint::default();
    paint.force_hq_pipeline = hq;
    paint.anti_alias = false;
    paint.shader = LinearGradient::new(
        Point::from_xy(100.0, 100.0),
        Point::from_xy(900.0, 900.0),
        points,
        mode,
        Transform::identity(),
    ).unwrap();

    let mut pb = PathBuilder::new();
    pb.move_to(60.0, 60.0);
    pb.line_to(160.0, 940.0);
    pb.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
    pb.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
    pb.close();
    let path = pb.finish().unwrap();

    let mut pixmap = Pixmap::new(1000, 1000).unwrap();

    bencher.iter(|| {
        pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    });
}

#[bench]
fn two_stops_linear_pad_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;
    two_stops_linear_tiny_skia(
        false,
        vec![
            GradientStop::new(0.0, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(1.0, Color::from_rgba8(220, 140, 75, 180)),
        ],
        SpreadMode::Pad,
        bencher,
    );
}

#[bench]
fn two_stops_linear_reflect_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;
    two_stops_linear_tiny_skia(
        false,
        vec![
            GradientStop::new(0.0, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(1.0, Color::from_rgba8(220, 140, 75, 180)),
        ],
        SpreadMode::Reflect,
        bencher,
    );
}

#[bench]
fn two_stops_linear_repeat_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;
    two_stops_linear_tiny_skia(
        false,
        vec![
            GradientStop::new(0.0, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(1.0, Color::from_rgba8(220, 140, 75, 180)),
        ],
        SpreadMode::Repeat,
        bencher,
    );
}

#[bench]
fn three_stops_linear_uneven_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;
    two_stops_linear_tiny_skia(
        false,
        vec![
            GradientStop::new(0.25, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(0.45, Color::from_rgba8(220, 140, 75, 180)),
            GradientStop::new(0.66, Color::from_rgba8(40, 180, 55, 160)),
        ],
        SpreadMode::Pad,
        bencher,
    );
}

#[bench]
fn three_stops_linear_even_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;
    two_stops_linear_tiny_skia(
        false,
        vec![
            GradientStop::new(0.25, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(0.50, Color::from_rgba8(220, 140, 75, 180)),
            GradientStop::new(0.75, Color::from_rgba8(40, 180, 55, 160)),
        ],
        SpreadMode::Pad,
        bencher,
    );
}

#[bench]
fn two_stops_linear_pad_tiny_skia_hq(bencher: &mut Bencher) {
    use tiny_skia::*;
    two_stops_linear_tiny_skia(
        true,
        vec![
            GradientStop::new(0.0, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(1.0, Color::from_rgba8(220, 140, 75, 180)),
        ],
        SpreadMode::Pad,
        bencher,
    );
}

#[bench]
fn two_stops_linear_reflect_tiny_skia_hq(bencher: &mut Bencher) {
    use tiny_skia::*;
    two_stops_linear_tiny_skia(
        true,
        vec![
            GradientStop::new(0.0, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(1.0, Color::from_rgba8(220, 140, 75, 180)),
        ],
        SpreadMode::Reflect,
        bencher,
    );
}

#[bench]
fn two_stops_linear_repeat_tiny_skia_hq(bencher: &mut Bencher) {
    use tiny_skia::*;
    two_stops_linear_tiny_skia(
        true,
        vec![
            GradientStop::new(0.0, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(1.0, Color::from_rgba8(220, 140, 75, 180)),
        ],
        SpreadMode::Repeat,
        bencher,
    );
}

#[bench]
fn three_stops_linear_uneven_tiny_skia_hq(bencher: &mut Bencher) {
    use tiny_skia::*;
    two_stops_linear_tiny_skia(
        true,
        vec![
            GradientStop::new(0.25, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(0.45, Color::from_rgba8(220, 140, 75, 180)),
            GradientStop::new(0.66, Color::from_rgba8(40, 180, 55, 160)),
        ],
        SpreadMode::Pad,
        bencher,
    );
}

#[bench]
fn three_stops_linear_even_tiny_skia_hq(bencher: &mut Bencher) {
    use tiny_skia::*;
    two_stops_linear_tiny_skia(
        true,
        vec![
            GradientStop::new(0.25, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(0.50, Color::from_rgba8(220, 140, 75, 180)),
            GradientStop::new(0.75, Color::from_rgba8(40, 180, 55, 160)),
        ],
        SpreadMode::Pad,
        bencher,
    );
}

#[cfg(feature = "skia-rs")]
fn two_stops_linear_skia(
    colors: Vec<skia_rs::Color>,
    positions: Vec<f32>,
    tile_mode: skia_rs::TileMode,
    bencher: &mut Bencher,
) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let mut paint = Paint::new();
    paint.set_style(PaintStyle::Fill);
    paint.set_shader(&Shader::new_linear_gradient(&LinearGradient {
        start_point: (100.0, 100.0),
        end_point: (900.0, 900.0),
        base: Gradient {
            colors,
            positions,
            tile_mode,
            transform: Transform::default(),
        },
    }).unwrap());

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
fn two_stops_linear_pad_skia(bencher: &mut Bencher) {
    use skia_rs::*;
    two_stops_linear_skia(
        vec![Color::from_rgba(50, 127, 150, 200), Color::from_rgba(220, 140, 75, 180)],
        vec![0.0, 1.0],
        skia_rs::TileMode::Clamp,
        bencher,
    );
}

#[cfg(feature = "skia-rs")]
#[bench]
fn two_stops_linear_reflect_skia(bencher: &mut Bencher) {
    use skia_rs::*;
    two_stops_linear_skia(
        vec![Color::from_rgba(50, 127, 150, 200), Color::from_rgba(220, 140, 75, 180)],
        vec![0.0, 1.0],
        skia_rs::TileMode::Mirror,
        bencher,
    );
}

#[cfg(feature = "skia-rs")]
#[bench]
fn two_stops_linear_repeat_skia(bencher: &mut Bencher) {
    use skia_rs::*;
    two_stops_linear_skia(
        vec![Color::from_rgba(50, 127, 150, 200), Color::from_rgba(220, 140, 75, 180)],
        vec![0.0, 1.0],
        skia_rs::TileMode::Repeat,
        bencher,
    );
}

#[cfg(feature = "skia-rs")]
#[bench]
fn three_stops_linear_uneven_skia(bencher: &mut Bencher) {
    use skia_rs::*;
    two_stops_linear_skia(
        vec![Color::from_rgba(50, 127, 150, 200),
             Color::from_rgba(220, 140, 75, 180),
             Color::from_rgba(40, 180, 55, 160)],
        vec![0.25, 0.45, 0.66],
        skia_rs::TileMode::Clamp,
        bencher,
    );
}

#[cfg(feature = "skia-rs")]
#[bench]
fn three_stops_linear_even_skia(bencher: &mut Bencher) {
    use skia_rs::*;
    two_stops_linear_skia(
        vec![Color::from_rgba(50, 127, 150, 200),
             Color::from_rgba(220, 140, 75, 180),
             Color::from_rgba(40, 180, 55, 160)],
        vec![0.25, 0.50, 0.75],
        TileMode::Clamp,
        bencher,
    );
}

#[cfg(feature = "raqote")]
fn two_stops_linear_raqote(
    stops: Vec<raqote::GradientStop>,
    mode: raqote::Spread,
    bencher: &mut Bencher,
) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);

    let path = {
        let mut pb = PathBuilder::new();
        pb.move_to(60.0, 60.0);
        pb.line_to(160.0, 940.0);
        pb.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
        pb.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
        pb.close();
        pb.finish()
    };

    let grad = Source::new_linear_gradient(
        Gradient { stops },
        Point::new(100.0, 100.0),
        Point::new(900.0, 900.0),
        mode
    );

    let draw_opt = DrawOptions {
        blend_mode: BlendMode::SrcOver,
        alpha: 1.0,
        antialias: AntialiasMode::None,
    };

    bencher.iter(|| {
        dt.fill(&path, &grad, &draw_opt);
    });
}

#[cfg(feature = "raqote")]
#[bench]
fn two_stops_linear_pad_raqote(bencher: &mut Bencher) {
    use raqote::*;
    two_stops_linear_raqote(
        vec![
            GradientStop {
                position: 0.0,
                color: Color::new(200, 50, 127, 150),
            },
            GradientStop {
                position: 1.0,
                color: Color::new(180, 220, 140, 75),
            }
        ],
        raqote::Spread::Pad,
        bencher,
    );
}

#[cfg(feature = "raqote")]
#[bench]
fn two_stops_linear_reflect_raqote(bencher: &mut Bencher) {
    use raqote::*;
    two_stops_linear_raqote(
        vec![
            GradientStop {
                position: 0.0,
                color: Color::new(200, 50, 127, 150),
            },
            GradientStop {
                position: 1.0,
                color: Color::new(180, 220, 140, 75),
            }
        ],
        raqote::Spread::Reflect,
        bencher,
    );
}

#[cfg(feature = "raqote")]
#[bench]
fn two_stops_linear_repeat_raqote(bencher: &mut Bencher) {
    use raqote::*;
    two_stops_linear_raqote(
        vec![
            GradientStop {
                position: 0.0,
                color: Color::new(200, 50, 127, 150),
            },
            GradientStop {
                position: 1.0,
                color: Color::new(180, 220, 140, 75),
            }
        ],
        raqote::Spread::Repeat,
        bencher,
    );
}

#[cfg(feature = "raqote")]
#[bench]
fn three_stops_linear_uneven_raqote(bencher: &mut Bencher) {
    use raqote::*;
    two_stops_linear_raqote(
        vec![
            GradientStop {
                position: 0.25,
                color: Color::new(200, 50, 127, 150),
            },
            GradientStop {
                position: 0.45,
                color: Color::new(180, 220, 140, 75),
            },
            GradientStop {
                position: 0.6,
                color: Color::new(160, 40, 180, 55),
            },
        ],
        raqote::Spread::Pad,
        bencher,
    );
}

#[cfg(feature = "raqote")]
#[bench]
fn three_stops_linear_even_raqote(bencher: &mut Bencher) {
    use raqote::*;
    two_stops_linear_raqote(
        vec![
            GradientStop {
                position: 0.25,
                color: Color::new(200, 50, 127, 150),
            },
            GradientStop {
                position: 0.50,
                color: Color::new(180, 220, 140, 75),
            },
            GradientStop {
                position: 0.75,
                color: Color::new(160, 40, 180, 55),
            },
        ],
        raqote::Spread::Pad,
        bencher,
    );
}

#[cfg(feature = "cairo-rs")]
fn two_stops_linear_cairo(
    stops: Vec<(f64, u8, u8, u8, u8)>,
    mode: cairo::Extend,
    bencher: &mut Bencher,
) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);

    cr.move_to(60.0, 60.0);
    cr.line_to(160.0, 940.0);
    cr.curve_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
    cr.curve_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
    cr.close_path();

    let grad = LinearGradient::new(100.0, 100.0, 900.0, 900.0);
    grad.set_extend(mode);
    for stop in stops {
        grad.add_color_stop_rgba(
            stop.0,
            stop.1 as f64 / 255.0,
            stop.2 as f64 / 255.0,
            stop.3 as f64 / 255.0,
            stop.4 as f64 / 255.0,
        );
    }

    cr.set_source(&grad);
    cr.set_antialias(Antialias::None);
    cr.set_operator(Operator::Over);

    bencher.iter(|| {
        cr.fill_preserve();
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn two_stops_linear_pad_cairo(bencher: &mut Bencher) {
    two_stops_linear_cairo(
        vec![
            (0.0, 50, 127, 150, 200),
            (1.0, 220, 140, 75, 180),
        ],
        cairo::Extend::Pad,
        bencher,
    );
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn two_stops_linear_reflect_cairo(bencher: &mut Bencher) {
    two_stops_linear_cairo(
        vec![
            (0.0, 50, 127, 150, 200),
            (1.0, 220, 140, 75, 180),
        ],
        cairo::Extend::Reflect,
        bencher,
    );
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn two_stops_linear_repeat_cairo(bencher: &mut Bencher) {
    two_stops_linear_cairo(
        vec![
            (0.0, 50, 127, 150, 200),
            (1.0, 220, 140, 75, 180),
        ],
        cairo::Extend::Repeat,
        bencher,
    );
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn three_stops_linear_uneven_cairo(bencher: &mut Bencher) {
    two_stops_linear_cairo(
        vec![
            (0.25, 50, 127, 150, 200),
            (0.45, 220, 140, 75, 180),
            (0.66, 40, 180, 55, 160),
        ],
        cairo::Extend::Pad,
        bencher,
    );
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn three_stops_linear_even_cairo(bencher: &mut Bencher) {
    two_stops_linear_cairo(
        vec![
            (0.25, 50, 127, 150, 200),
            (0.50, 220, 140, 75, 180),
            (0.75, 40, 180, 55, 160),
        ],
        cairo::Extend::Pad,
        bencher,
    );
}

#[bench]
fn simple_radial_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = RadialGradient::new(
        Point::from_xy(500.0, 500.0),
        0.0,
        Point::from_xy(500.0, 500.0),
        500.0,
        vec![
            GradientStop::new(0.25, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(1.00, Color::from_rgba8(220, 140, 75, 180)),
        ],
        SpreadMode::Pad,
        Transform::identity(),
    ).unwrap();

    let mut pb = PathBuilder::new();
    pb.move_to(60.0, 60.0);
    pb.line_to(160.0, 940.0);
    pb.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
    pb.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
    pb.close();
    let path = pb.finish().unwrap();

    let mut pixmap = Pixmap::new(1000, 1000).unwrap();

    bencher.iter(|| {
        pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    });
}

#[cfg(feature = "skia-rs")]
#[bench]
fn simple_radial_skia(bencher: &mut Bencher) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let mut paint = Paint::new();
    paint.set_style(PaintStyle::Fill);
    paint.set_shader(&Shader::new_two_point_conical_gradient(&TwoPointConicalGradient {
        start: (500.0, 500.0),
        start_radius: 0.0,
        end: (500.0, 500.0),
        end_radius: 500.0,
        base: Gradient {
            colors: vec![Color::from_rgba(50, 127, 150, 200), Color::from_rgba(220, 140, 75, 180)],
            positions: vec![0.25, 1.0],
            tile_mode: TileMode::Clamp,
            transform: Transform::default(),
        },
    }).unwrap());

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

#[cfg(feature = "raqote")]
#[bench]
fn simple_radial_raqote(bencher: &mut Bencher) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);

    let path = {
        let mut pb = PathBuilder::new();
        pb.move_to(60.0, 60.0);
        pb.line_to(160.0, 940.0);
        pb.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
        pb.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
        pb.close();
        pb.finish()
    };

    let grad = Source::new_radial_gradient(
        Gradient { stops: vec![
            GradientStop {
                position: 0.25,
                color: Color::new(200, 50, 127, 150),
            },
            GradientStop {
                position: 1.0,
                color: Color::new(180, 220, 140, 75),
            }
        ], },
        Point::new(500.0, 500.0),
        500.0,
        Spread::Pad,
    );

    let draw_opt = DrawOptions {
        blend_mode: BlendMode::SrcOver,
        alpha: 1.0,
        antialias: AntialiasMode::None,
    };

    bencher.iter(|| {
        dt.fill(&path, &grad, &draw_opt);
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn simple_radial_cairo(bencher: &mut Bencher) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);

    cr.move_to(60.0, 60.0);
    cr.line_to(160.0, 940.0);
    cr.curve_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
    cr.curve_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
    cr.close_path();

    let grad = RadialGradient::new(500.0, 500.0, 0.0, 500.0, 500.0, 500.0);
    grad.set_extend(Extend::Pad);

    grad.add_color_stop_rgba(
        0.25,
        50.0 / 255.0,
        127.0 / 255.0,
        150.0 / 255.0,
        200.0 / 255.0,
    );

    grad.add_color_stop_rgba(
        1.0,
        220.0 / 255.0,
        140.0 / 255.0,
        75.0 / 255.0,
        180.0 / 255.0,
    );

    cr.set_source(&grad);
    cr.set_antialias(Antialias::None);
    cr.set_operator(Operator::Over);

    bencher.iter(|| {
        cr.fill_preserve();
    });
}

#[bench]
fn two_point_radial_tiny_skia(bencher: &mut Bencher) {
    use tiny_skia::*;

    let mut paint = Paint::default();
    paint.anti_alias = false;
    paint.shader = RadialGradient::new(
        Point::from_xy(400.0, 400.0),
        0.0,
        Point::from_xy(500.0, 500.0),
        500.0,
        vec![
            GradientStop::new(0.25, Color::from_rgba8(50, 127, 150, 200)),
            GradientStop::new(1.00, Color::from_rgba8(220, 140, 75, 180)),
        ],
        SpreadMode::Pad,
        Transform::identity(),
    ).unwrap();

    let mut pb = PathBuilder::new();
    pb.move_to(60.0, 60.0);
    pb.line_to(160.0, 940.0);
    pb.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
    pb.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
    pb.close();
    let path = pb.finish().unwrap();

    let mut pixmap = Pixmap::new(1000, 1000).unwrap();

    bencher.iter(|| {
        pixmap.fill_path(&path, &paint, FillRule::Winding, Transform::identity(), None);
    });
}

#[cfg(feature = "skia-rs")]
#[bench]
fn two_point_radial_skia(bencher: &mut Bencher) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let mut paint = Paint::new();
    paint.set_style(PaintStyle::Fill);
    paint.set_shader(&Shader::new_two_point_conical_gradient(&TwoPointConicalGradient {
        start: (400.0, 400.0),
        start_radius: 0.0,
        end: (500.0, 500.0),
        end_radius: 500.0,
        base: Gradient {
            colors: vec![Color::from_rgba(50, 127, 150, 200), Color::from_rgba(220, 140, 75, 180)],
            positions: vec![0.25, 1.0],
            tile_mode: TileMode::Clamp,
            transform: Transform::default(),
        },
    }).unwrap());

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

#[cfg(feature = "raqote")]
#[bench]
fn two_point_radial_raqote(bencher: &mut Bencher) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);

    let path = {
        let mut pb = PathBuilder::new();
        pb.move_to(60.0, 60.0);
        pb.line_to(160.0, 940.0);
        pb.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
        pb.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
        pb.close();
        pb.finish()
    };

    let grad = Source::new_two_circle_radial_gradient(
        Gradient { stops: vec![
            GradientStop {
                position: 0.25,
                color: Color::new(200, 50, 127, 150),
            },
            GradientStop {
                position: 1.0,
                color: Color::new(180, 220, 140, 75),
            }
        ], },
        Point::new(400.0, 400.0),
        0.0,
        Point::new(500.0, 500.0),
        500.0,
        Spread::Pad,
    );

    let draw_opt = DrawOptions {
        blend_mode: BlendMode::SrcOver,
        alpha: 1.0,
        antialias: AntialiasMode::None,
    };

    bencher.iter(|| {
        dt.fill(&path, &grad, &draw_opt);
    });
}

#[cfg(feature = "cairo-rs")]
#[bench]
fn two_point_radial_cairo(bencher: &mut Bencher) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);

    cr.move_to(60.0, 60.0);
    cr.line_to(160.0, 940.0);
    cr.curve_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
    cr.curve_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
    cr.close_path();

    let grad = RadialGradient::new(400.0, 400.0, 0.0, 500.0, 500.0, 500.0);
    grad.set_extend(Extend::Pad);

    grad.add_color_stop_rgba(
        0.25,
        50.0 / 255.0,
        127.0 / 255.0,
        150.0 / 255.0,
        200.0 / 255.0,
    );

    grad.add_color_stop_rgba(
        1.0,
        220.0 / 255.0,
        140.0 / 255.0,
        75.0 / 255.0,
        180.0 / 255.0,
    );

    cr.set_source(&grad);
    cr.set_antialias(Antialias::None);
    cr.set_operator(Operator::Over);

    bencher.iter(|| {
        cr.fill_preserve();
    });
}
