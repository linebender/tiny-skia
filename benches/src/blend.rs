use test::Bencher;

fn fill_tiny_skia(blend_mode: tiny_skia::BlendMode, bencher: &mut Bencher) {
    use tiny_skia::*;

    let mut paint1 = Paint::default();
    paint1.set_color_rgba8(50, 127, 150, 200);
    paint1.blend_mode = BlendMode::SourceOver;
    paint1.anti_alias = false;

    let mut paint2 = Paint::default();
    paint2.set_color_rgba8(220, 140, 75, 180);
    paint2.blend_mode = blend_mode; // <-- variable
    paint2.anti_alias = false;

    let path1 = {
        let mut pb = PathBuilder::new();
        pb.move_to(60.0, 60.0);
        pb.line_to(160.0, 940.0);
        pb.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
        pb.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
        pb.close();
        pb.finish().unwrap()
    };

    let path2 = {
        let mut pb = PathBuilder::new();
        pb.move_to(940.0, 60.0);
        pb.line_to(840.0, 940.0);
        pb.cubic_to(620.0, 840.0, 340.0, 800.0, 60.0, 800.0);
        pb.cubic_to(260.0, 460.0, 560.0, 160.0, 940.0, 60.0);
        pb.close();
        pb.finish().unwrap()
    };

    #[cfg(feature = "16bpc")]
    if crate::is_16bpc() {
        let mut pixmap = PixmapU16::new(1000, 1000).unwrap();
        pixmap.fill_path(&path1, &paint1, FillRule::Winding, Transform::identity(), None);

        bencher.iter(|| {
            pixmap.fill_path(&path2, &paint2, FillRule::Winding, Transform::identity(), None);
        });
        return;
    }

    let mut pixmap = Pixmap::new(1000, 1000).unwrap();
    pixmap.fill_path(&path1, &paint1, FillRule::Winding, Transform::identity(), None);

    bencher.iter(|| {
        pixmap.fill_path(&path2, &paint2, FillRule::Winding, Transform::identity(), None);
    });
}

#[bench] fn clear_tiny_skia(bencher: &mut Bencher)               { fill_tiny_skia(tiny_skia::BlendMode::Clear, bencher); }
#[bench] fn source_tiny_skia(bencher: &mut Bencher)              { fill_tiny_skia(tiny_skia::BlendMode::Source, bencher); }
#[bench] fn destination_tiny_skia(bencher: &mut Bencher)         { fill_tiny_skia(tiny_skia::BlendMode::Destination, bencher); }
#[bench] fn source_over_tiny_skia(bencher: &mut Bencher)         { fill_tiny_skia(tiny_skia::BlendMode::SourceOver, bencher); }
#[bench] fn destination_over_tiny_skia(bencher: &mut Bencher)    { fill_tiny_skia(tiny_skia::BlendMode::DestinationOver, bencher); }
#[bench] fn source_in_tiny_skia(bencher: &mut Bencher)           { fill_tiny_skia(tiny_skia::BlendMode::SourceIn, bencher); }
#[bench] fn destination_in_tiny_skia(bencher: &mut Bencher)      { fill_tiny_skia(tiny_skia::BlendMode::DestinationIn, bencher); }
#[bench] fn source_out_tiny_skia(bencher: &mut Bencher)          { fill_tiny_skia(tiny_skia::BlendMode::SourceOut, bencher); }
#[bench] fn destination_out_tiny_skia(bencher: &mut Bencher)     { fill_tiny_skia(tiny_skia::BlendMode::DestinationOut, bencher); }
#[bench] fn source_atop_tiny_skia(bencher: &mut Bencher)         { fill_tiny_skia(tiny_skia::BlendMode::SourceAtop, bencher); }
#[bench] fn destination_atop_tiny_skia(bencher: &mut Bencher)    { fill_tiny_skia(tiny_skia::BlendMode::DestinationAtop, bencher); }
#[bench] fn xor_tiny_skia(bencher: &mut Bencher)                 { fill_tiny_skia(tiny_skia::BlendMode::Xor, bencher); }
#[bench] fn plus_tiny_skia(bencher: &mut Bencher)                { fill_tiny_skia(tiny_skia::BlendMode::Plus, bencher); }
#[bench] fn modulate_tiny_skia(bencher: &mut Bencher)            { fill_tiny_skia(tiny_skia::BlendMode::Modulate, bencher); }
#[bench] fn screen_tiny_skia(bencher: &mut Bencher)              { fill_tiny_skia(tiny_skia::BlendMode::Screen, bencher); }
#[bench] fn overlay_tiny_skia(bencher: &mut Bencher)             { fill_tiny_skia(tiny_skia::BlendMode::Overlay, bencher); }
#[bench] fn darken_tiny_skia(bencher: &mut Bencher)              { fill_tiny_skia(tiny_skia::BlendMode::Darken, bencher); }
#[bench] fn lighten_tiny_skia(bencher: &mut Bencher)             { fill_tiny_skia(tiny_skia::BlendMode::Lighten, bencher); }
#[bench] fn color_dodge_tiny_skia(bencher: &mut Bencher)         { fill_tiny_skia(tiny_skia::BlendMode::ColorDodge, bencher); }
#[bench] fn color_burn_tiny_skia(bencher: &mut Bencher)          { fill_tiny_skia(tiny_skia::BlendMode::ColorBurn, bencher); }
#[bench] fn hard_light_tiny_skia(bencher: &mut Bencher)          { fill_tiny_skia(tiny_skia::BlendMode::HardLight, bencher); }
#[bench] fn soft_light_tiny_skia(bencher: &mut Bencher)          { fill_tiny_skia(tiny_skia::BlendMode::SoftLight, bencher); }
#[bench] fn difference_tiny_skia(bencher: &mut Bencher)          { fill_tiny_skia(tiny_skia::BlendMode::Difference, bencher); }
#[bench] fn exclusion_tiny_skia(bencher: &mut Bencher)           { fill_tiny_skia(tiny_skia::BlendMode::Exclusion, bencher); }
#[bench] fn multiply_tiny_skia(bencher: &mut Bencher)            { fill_tiny_skia(tiny_skia::BlendMode::Multiply, bencher); }
#[bench] fn hue_tiny_skia(bencher: &mut Bencher)                 { fill_tiny_skia(tiny_skia::BlendMode::Hue, bencher); }
#[bench] fn saturation_tiny_skia(bencher: &mut Bencher)          { fill_tiny_skia(tiny_skia::BlendMode::Saturation, bencher); }
#[bench] fn color_tiny_skia(bencher: &mut Bencher)               { fill_tiny_skia(tiny_skia::BlendMode::Color, bencher); }
#[bench] fn luminosity_tiny_skia(bencher: &mut Bencher)          { fill_tiny_skia(tiny_skia::BlendMode::Luminosity, bencher); }

#[cfg(feature = "skia-rs")]
fn fill_skia(mode: skia_rs::BlendMode, bencher: &mut Bencher) {
    use skia_rs::*;

    let mut surface = Surface::new_rgba_premultiplied(1000, 1000).unwrap();

    let mut paint1 = Paint::new();
    paint1.set_color(50, 127, 150, 200);
    paint1.set_blend_mode(BlendMode::SourceOver);

    let mut paint2 = Paint::new();
    paint2.set_color(220, 140, 75, 180);
    paint2.set_blend_mode(mode); // <-- variable

    let path1 = {
        let mut path = Path::new();
        path.move_to(60.0, 60.0);
        path.line_to(160.0, 940.0);
        path.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
        path.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
        path.close();
        path
    };

    let path2 = {
        let mut path = Path::new();
        path.move_to(940.0, 60.0);
        path.line_to(840.0, 940.0);
        path.cubic_to(620.0, 840.0, 340.0, 800.0, 60.0, 800.0);
        path.cubic_to(260.0, 460.0, 560.0, 160.0, 940.0, 60.0);
        path.close();
        path
    };

    surface.draw_path(&path1, &paint1);

    bencher.iter(|| {
        surface.draw_path(&path2, &paint2);
    });
}

#[cfg(feature = "skia-rs")] #[bench] fn clear_skia(bencher: &mut Bencher)               { fill_skia(skia_rs::BlendMode::Clear, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn source_skia(bencher: &mut Bencher)              { fill_skia(skia_rs::BlendMode::Source, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn destination_skia(bencher: &mut Bencher)         { fill_skia(skia_rs::BlendMode::Destination, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn source_over_skia(bencher: &mut Bencher)         { fill_skia(skia_rs::BlendMode::SourceOver, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn destination_over_skia(bencher: &mut Bencher)    { fill_skia(skia_rs::BlendMode::DestinationOver, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn source_in_skia(bencher: &mut Bencher)           { fill_skia(skia_rs::BlendMode::SourceIn, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn destination_in_skia(bencher: &mut Bencher)      { fill_skia(skia_rs::BlendMode::DestinationIn, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn source_out_skia(bencher: &mut Bencher)          { fill_skia(skia_rs::BlendMode::SourceOut, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn destination_out_skia(bencher: &mut Bencher)     { fill_skia(skia_rs::BlendMode::DestinationOut, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn source_atop_skia(bencher: &mut Bencher)         { fill_skia(skia_rs::BlendMode::SourceATop, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn destination_atop_skia(bencher: &mut Bencher)    { fill_skia(skia_rs::BlendMode::DestinationATop, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn xor_skia(bencher: &mut Bencher)                 { fill_skia(skia_rs::BlendMode::Xor, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn plus_skia(bencher: &mut Bencher)                { fill_skia(skia_rs::BlendMode::Plus, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn modulate_skia(bencher: &mut Bencher)            { fill_skia(skia_rs::BlendMode::Modulate, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn screen_skia(bencher: &mut Bencher)              { fill_skia(skia_rs::BlendMode::Screen, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn overlay_skia(bencher: &mut Bencher)             { fill_skia(skia_rs::BlendMode::Overlay, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn darken_skia(bencher: &mut Bencher)              { fill_skia(skia_rs::BlendMode::Darken, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn lighten_skia(bencher: &mut Bencher)             { fill_skia(skia_rs::BlendMode::Lighten, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn color_dodge_skia(bencher: &mut Bencher)         { fill_skia(skia_rs::BlendMode::ColorDodge, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn color_burn_skia(bencher: &mut Bencher)          { fill_skia(skia_rs::BlendMode::ColorBurn, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn hard_light_skia(bencher: &mut Bencher)          { fill_skia(skia_rs::BlendMode::HardLight, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn soft_light_skia(bencher: &mut Bencher)          { fill_skia(skia_rs::BlendMode::SoftLight, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn difference_skia(bencher: &mut Bencher)          { fill_skia(skia_rs::BlendMode::Difference, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn exclusion_skia(bencher: &mut Bencher)           { fill_skia(skia_rs::BlendMode::Exclusion, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn multiply_skia(bencher: &mut Bencher)            { fill_skia(skia_rs::BlendMode::Multiply, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn hue_skia(bencher: &mut Bencher)                 { fill_skia(skia_rs::BlendMode::Hue, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn saturation_skia(bencher: &mut Bencher)          { fill_skia(skia_rs::BlendMode::Saturation, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn color_skia(bencher: &mut Bencher)               { fill_skia(skia_rs::BlendMode::Color, bencher); }
#[cfg(feature = "skia-rs")] #[bench] fn luminosity_skia(bencher: &mut Bencher)          { fill_skia(skia_rs::BlendMode::Luminosity, bencher); }

#[cfg(feature = "raqote")]
fn fill_raqote(blend_mode: raqote::BlendMode, bencher: &mut Bencher) {
    use raqote::*;

    let mut dt = DrawTarget::new(1000, 1000);

    let path1 = {
        let mut pb = PathBuilder::new();
        pb.move_to(60.0, 60.0);
        pb.line_to(160.0, 940.0);
        pb.cubic_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
        pb.cubic_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
        pb.close();
        pb.finish()
    };

    let path2 = {
        let mut pb = PathBuilder::new();
        pb.move_to(940.0, 60.0);
        pb.line_to(840.0, 940.0);
        pb.cubic_to(620.0, 840.0, 340.0, 800.0, 60.0, 800.0);
        pb.cubic_to(260.0, 460.0, 560.0, 160.0, 940.0, 60.0);
        pb.close();
        pb.finish()
    };

    // raqote uses ARGB order.
    let src1 = Source::from(Color::new(200, 50, 127, 150));
    let src2 = Source::from(Color::new(180, 220, 140, 75));

    let draw_opt1 = DrawOptions {
        blend_mode: BlendMode::SrcOver,
        alpha: 1.0,
        antialias: AntialiasMode::None,
    };

    let draw_opt2 = DrawOptions {
        blend_mode, // <-- variable
        alpha: 1.0,
        antialias: AntialiasMode::None,
    };

    dt.fill(&path1, &src1, &draw_opt1);

    bencher.iter(|| {
        dt.fill(&path2, &src2, &draw_opt2);
    });
}

#[cfg(feature = "raqote")] #[bench] fn clear_raqote(bencher: &mut Bencher)               { fill_raqote(raqote::BlendMode::Clear, bencher); }
#[cfg(feature = "raqote")] #[bench] fn source_raqote(bencher: &mut Bencher)              { fill_raqote(raqote::BlendMode::Src, bencher); }
#[cfg(feature = "raqote")] #[bench] fn destination_raqote(bencher: &mut Bencher)         { fill_raqote(raqote::BlendMode::Dst, bencher); }
#[cfg(feature = "raqote")] #[bench] fn source_over_raqote(bencher: &mut Bencher)         { fill_raqote(raqote::BlendMode::SrcOver, bencher); }
#[cfg(feature = "raqote")] #[bench] fn destination_over_raqote(bencher: &mut Bencher)    { fill_raqote(raqote::BlendMode::DstOver, bencher); }
#[cfg(feature = "raqote")] #[bench] fn source_in_raqote(bencher: &mut Bencher)           { fill_raqote(raqote::BlendMode::SrcIn, bencher); }
#[cfg(feature = "raqote")] #[bench] fn destination_in_raqote(bencher: &mut Bencher)      { fill_raqote(raqote::BlendMode::DstIn, bencher); }
#[cfg(feature = "raqote")] #[bench] fn source_out_raqote(bencher: &mut Bencher)          { fill_raqote(raqote::BlendMode::SrcOut, bencher); }
#[cfg(feature = "raqote")] #[bench] fn destination_out_raqote(bencher: &mut Bencher)     { fill_raqote(raqote::BlendMode::DstOut, bencher); }
#[cfg(feature = "raqote")] #[bench] fn source_atop_raqote(bencher: &mut Bencher)         { fill_raqote(raqote::BlendMode::SrcAtop, bencher); }
#[cfg(feature = "raqote")] #[bench] fn destination_atop_raqote(bencher: &mut Bencher)    { fill_raqote(raqote::BlendMode::DstAtop, bencher); }
#[cfg(feature = "raqote")] #[bench] fn xor_raqote(bencher: &mut Bencher)                 { fill_raqote(raqote::BlendMode::Xor, bencher); }
#[cfg(feature = "raqote")] #[bench] fn plus_raqote(bencher: &mut Bencher)                { fill_raqote(raqote::BlendMode::Add, bencher); }
#[cfg(feature = "raqote")] #[bench] fn screen_raqote(bencher: &mut Bencher)              { fill_raqote(raqote::BlendMode::Screen, bencher); }
#[cfg(feature = "raqote")] #[bench] fn overlay_raqote(bencher: &mut Bencher)             { fill_raqote(raqote::BlendMode::Overlay, bencher); }
#[cfg(feature = "raqote")] #[bench] fn darken_raqote(bencher: &mut Bencher)              { fill_raqote(raqote::BlendMode::Darken, bencher); }
#[cfg(feature = "raqote")] #[bench] fn lighten_raqote(bencher: &mut Bencher)             { fill_raqote(raqote::BlendMode::Lighten, bencher); }
#[cfg(feature = "raqote")] #[bench] fn color_dodge_raqote(bencher: &mut Bencher)         { fill_raqote(raqote::BlendMode::ColorDodge, bencher); }
#[cfg(feature = "raqote")] #[bench] fn color_burn_raqote(bencher: &mut Bencher)          { fill_raqote(raqote::BlendMode::ColorBurn, bencher); }
#[cfg(feature = "raqote")] #[bench] fn hard_light_raqote(bencher: &mut Bencher)          { fill_raqote(raqote::BlendMode::HardLight, bencher); }
#[cfg(feature = "raqote")] #[bench] fn soft_light_raqote(bencher: &mut Bencher)          { fill_raqote(raqote::BlendMode::SoftLight, bencher); }
#[cfg(feature = "raqote")] #[bench] fn difference_raqote(bencher: &mut Bencher)          { fill_raqote(raqote::BlendMode::Difference, bencher); }
#[cfg(feature = "raqote")] #[bench] fn exclusion_raqote(bencher: &mut Bencher)           { fill_raqote(raqote::BlendMode::Exclusion, bencher); }
#[cfg(feature = "raqote")] #[bench] fn multiply_raqote(bencher: &mut Bencher)            { fill_raqote(raqote::BlendMode::Multiply, bencher); }
#[cfg(feature = "raqote")] #[bench] fn hue_raqote(bencher: &mut Bencher)                 { fill_raqote(raqote::BlendMode::Hue, bencher); }
#[cfg(feature = "raqote")] #[bench] fn saturation_raqote(bencher: &mut Bencher)          { fill_raqote(raqote::BlendMode::Saturation, bencher); }
#[cfg(feature = "raqote")] #[bench] fn color_raqote(bencher: &mut Bencher)               { fill_raqote(raqote::BlendMode::Color, bencher); }
#[cfg(feature = "raqote")] #[bench] fn luminosity_raqote(bencher: &mut Bencher)          { fill_raqote(raqote::BlendMode::Luminosity, bencher); }

#[cfg(feature = "cairo-rs")]
fn fill_cairo(op: cairo::Operator, bencher: &mut Bencher) {
    use cairo::*;

    let surface = ImageSurface::create(Format::ARgb32, 1000, 1000).unwrap();

    let cr = Context::new(&surface);

    cr.move_to(60.0, 60.0);
    cr.line_to(160.0, 940.0);
    cr.curve_to(380.0, 840.0, 660.0, 800.0, 940.0, 800.0);
    cr.curve_to(740.0, 460.0, 440.0, 160.0, 60.0, 60.0);
    cr.close_path();

    cr.set_source_rgba(50.0 / 255.0, 127.0 / 255.0, 150.0 / 255.0, 200.0 / 255.0);
    cr.set_antialias(Antialias::None);
    cr.set_fill_rule(FillRule::Winding);
    cr.set_operator(Operator::Over);
    cr.fill();

    cr.move_to(940.0, 60.0);
    cr.line_to(840.0, 940.0);
    cr.curve_to(620.0, 840.0, 340.0, 800.0, 60.0, 800.0);
    cr.curve_to(260.0, 460.0, 560.0, 160.0, 940.0, 60.0);
    cr.close_path();

    cr.set_source_rgba(220.0 / 255.0, 140.0 / 255.0, 75.0 / 255.0, 180.0 / 255.0);
    cr.set_antialias(Antialias::None);
    cr.set_fill_rule(FillRule::Winding);
    cr.set_operator(op); // <-- variable

    bencher.iter(|| {
        cr.fill_preserve();
    });
}

#[cfg(feature = "cairo-rs")] #[bench] fn clear_cairo(bencher: &mut Bencher)               { fill_cairo(cairo::Operator::Clear, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn source_cairo(bencher: &mut Bencher)              { fill_cairo(cairo::Operator::Source, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn destination_cairo(bencher: &mut Bencher)         { fill_cairo(cairo::Operator::Dest, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn source_over_cairo(bencher: &mut Bencher)         { fill_cairo(cairo::Operator::Over, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn destination_over_cairo(bencher: &mut Bencher)    { fill_cairo(cairo::Operator::DestOver, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn source_in_cairo(bencher: &mut Bencher)           { fill_cairo(cairo::Operator::In, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn destination_in_cairo(bencher: &mut Bencher)      { fill_cairo(cairo::Operator::DestIn, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn source_out_cairo(bencher: &mut Bencher)          { fill_cairo(cairo::Operator::Out, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn destination_out_cairo(bencher: &mut Bencher)     { fill_cairo(cairo::Operator::DestOut, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn source_atop_cairo(bencher: &mut Bencher)         { fill_cairo(cairo::Operator::Atop, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn destination_atop_cairo(bencher: &mut Bencher)    { fill_cairo(cairo::Operator::DestAtop, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn xor_cairo(bencher: &mut Bencher)                 { fill_cairo(cairo::Operator::Xor, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn plus_cairo(bencher: &mut Bencher)                { fill_cairo(cairo::Operator::Add, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn screen_cairo(bencher: &mut Bencher)              { fill_cairo(cairo::Operator::Screen, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn overlay_cairo(bencher: &mut Bencher)             { fill_cairo(cairo::Operator::Overlay, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn darken_cairo(bencher: &mut Bencher)              { fill_cairo(cairo::Operator::Darken, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn lighten_cairo(bencher: &mut Bencher)             { fill_cairo(cairo::Operator::Lighten, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn color_dodge_cairo(bencher: &mut Bencher)         { fill_cairo(cairo::Operator::ColorDodge, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn color_burn_cairo(bencher: &mut Bencher)          { fill_cairo(cairo::Operator::ColorBurn, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn hard_light_cairo(bencher: &mut Bencher)          { fill_cairo(cairo::Operator::HardLight, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn soft_light_cairo(bencher: &mut Bencher)          { fill_cairo(cairo::Operator::SoftLight, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn difference_cairo(bencher: &mut Bencher)          { fill_cairo(cairo::Operator::Difference, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn exclusion_cairo(bencher: &mut Bencher)           { fill_cairo(cairo::Operator::Exclusion, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn multiply_cairo(bencher: &mut Bencher)            { fill_cairo(cairo::Operator::Multiply, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn hue_cairo(bencher: &mut Bencher)                 { fill_cairo(cairo::Operator::HslHue, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn saturation_cairo(bencher: &mut Bencher)          { fill_cairo(cairo::Operator::HslSaturation, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn color_cairo(bencher: &mut Bencher)               { fill_cairo(cairo::Operator::HslColor, bencher); }
#[cfg(feature = "cairo-rs")] #[bench] fn luminosity_cairo(bencher: &mut Bencher)          { fill_cairo(cairo::Operator::HslLuminosity, bencher); }
