use tiny_skia::*;

fn draw_line(x0: f32, y0: f32, x1: f32, y1: f32, anti_alias: bool, width: f32, line_cap: LineCap) -> Pixmap {
    let mut pixmap = Pixmap::new(100, 100).unwrap();

    let mut pb = PathBuilder::new();
    pb.move_to(x0, y0);
    pb.line_to(x1, y1);
    let path = pb.finish().unwrap();

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = anti_alias;

    let mut stroke = Stroke::default();
    stroke.width = width;
    stroke.line_cap = line_cap;
    pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);

    pixmap
}

#[test]
fn hline_05() {
    let expected = Pixmap::load_png("tests/images/hairline/hline-05.png").unwrap();
    assert_eq!(draw_line(10.0, 10.0, 90.0, 10.0, false, 0.5, LineCap::Butt), expected);
}

#[test]
fn hline_05_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/hline-05-aa.png").unwrap();
    assert_eq!(draw_line(10.0, 10.0, 90.0, 10.0, true, 0.5, LineCap::Butt), expected);
}

#[test]
fn hline_05_aa_round() {
    let expected = Pixmap::load_png("tests/images/hairline/hline-05-aa-round.png").unwrap();
    assert_eq!(draw_line(10.0, 10.0, 90.0, 10.0, true, 0.5, LineCap::Round), expected);
}

#[test]
fn vline_05() {
    let expected = Pixmap::load_png("tests/images/hairline/vline-05.png").unwrap();
    assert_eq!(draw_line(10.0, 10.0, 10.0, 90.0, false, 0.5, LineCap::Butt), expected);
}

#[test]
fn vline_05_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/vline-05-aa.png").unwrap();
    assert_eq!(draw_line(10.0, 10.0, 10.0, 90.0, true, 0.5, LineCap::Butt), expected);
}

#[test]
fn vline_05_aa_round() {
    let expected = Pixmap::load_png("tests/images/hairline/vline-05-aa-round.png").unwrap();
    assert_eq!(draw_line(10.0, 10.0, 10.0, 90.0, true, 0.5, LineCap::Round), expected);
}

#[test]
fn horish_05_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/horish-05-aa.png").unwrap();
    assert_eq!(draw_line(10.0, 10.0, 90.0, 70.0, true, 0.5, LineCap::Butt), expected);
}

#[test]
fn vertish_05_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/vertish-05-aa.png").unwrap();
    assert_eq!(draw_line(10.0, 10.0, 70.0, 90.0, true, 0.5, LineCap::Butt), expected);
}

#[test]
fn clip_line_05_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/clip-line-05-aa.png").unwrap();
    assert_eq!(draw_line(-10.0, 10.0, 110.0, 70.0, true, 0.5, LineCap::Butt), expected);
}

#[test]
fn clip_line_00() {
    let expected = Pixmap::load_png("tests/images/hairline/clip-line-00.png").unwrap();
    assert_eq!(draw_line(-10.0, 10.0, 110.0, 70.0, false, 0.0, LineCap::Butt), expected);
}

#[test]
fn clip_line_00_v2() {
    let mut pixmap = Pixmap::new(512, 512).unwrap();

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = false;

    let mut stroke = Stroke::default();
    stroke.width = 0.0;

    let mut builder = PathBuilder::default();
    builder.move_to(369.26462, 577.8069);
    builder.line_to(488.0846, 471.04388);
    let path = builder.finish().unwrap();
    pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);

    let expected = Pixmap::load_png("tests/images/hairline/clip-line-00-v2.png").unwrap();
    assert_eq!(pixmap, expected);
}

#[test]
fn clip_hline_top_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/clip-hline-top-aa.png").unwrap();
    assert_eq!(draw_line(-1.0, 0.0, 101.0, 0.0, true, 1.0, LineCap::Butt), expected);
}

#[test]
fn clip_hline_bottom_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/clip-hline-bottom-aa.png").unwrap();
    assert_eq!(draw_line(-1.0, 100.0, 101.0, 100.0, true, 1.0, LineCap::Butt), expected);
}

#[test]
fn clip_vline_left_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/clip-vline-left-aa.png").unwrap();
    assert_eq!(draw_line(0.0, -1.0, 0.0, 101.0, true, 1.0, LineCap::Butt), expected);
}

#[test]
fn clip_vline_right_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/clip-vline-right-aa.png").unwrap();
    assert_eq!(draw_line(100.0, -1.0, 100.0, 101.0, true, 1.0, LineCap::Butt), expected);
}

fn draw_quad(anti_alias: bool, width: f32, line_cap: LineCap) -> Pixmap {
    let mut pixmap = Pixmap::new(200, 100).unwrap();

    let mut pb = PathBuilder::new();
    pb.move_to(25.0, 80.0);
    pb.quad_to(155.0, 75.0, 175.0, 20.0);
    let path = pb.finish().unwrap();

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = anti_alias;

    let mut stroke = Stroke::default();
    stroke.width = width;
    stroke.line_cap = line_cap;
    pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);

    pixmap
}

#[test]
fn quad_width_05_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/quad-width-05-aa.png").unwrap();
    assert_eq!(draw_quad(true, 0.5, LineCap::Butt), expected);
}

#[test]
fn quad_width_05_aa_round() {
    let expected = Pixmap::load_png("tests/images/hairline/quad-width-05-aa-round.png").unwrap();
    assert_eq!(draw_quad(true, 0.5, LineCap::Round), expected);
}

#[test]
fn quad_width_00() {
    let expected = Pixmap::load_png("tests/images/hairline/quad-width-00.png").unwrap();
    assert_eq!(draw_quad(false, 0.0, LineCap::Butt), expected);
}

fn draw_cubic(points: &[f32; 8], anti_alias: bool, width: f32, line_cap: LineCap) -> Pixmap {
    let mut pixmap = Pixmap::new(200, 100).unwrap();

    let mut pb = PathBuilder::new();
    pb.move_to(points[0], points[1]);
    pb.cubic_to(points[2], points[3], points[4], points[5], points[6], points[7]);
    let path = pb.finish().unwrap();

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = anti_alias;

    let mut stroke = Stroke::default();
    stroke.width = width;
    stroke.line_cap = line_cap;
    pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);

    pixmap
}

#[test]
fn cubic_width_10_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/cubic-width-10-aa.png").unwrap();
    assert_eq!(draw_cubic(&[25.0, 80.0, 55.0, 25.0, 155.0, 75.0, 175.0, 20.0], true, 1.0, LineCap::Butt), expected);
}

#[test]
fn cubic_width_05_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/cubic-width-05-aa.png").unwrap();
    assert_eq!(draw_cubic(&[25.0, 80.0, 55.0, 25.0, 155.0, 75.0, 175.0, 20.0], true, 0.5, LineCap::Butt), expected);
}

#[test]
fn cubic_width_00_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/cubic-width-00-aa.png").unwrap();
    assert_eq!(draw_cubic(&[25.0, 80.0, 55.0, 25.0, 155.0, 75.0, 175.0, 20.0], true, 0.0, LineCap::Butt), expected);
}

#[test]
fn cubic_width_00() {
    let expected = Pixmap::load_png("tests/images/hairline/cubic-width-00.png").unwrap();
    assert_eq!(draw_cubic(&[25.0, 80.0, 55.0, 25.0, 155.0, 75.0, 175.0, 20.0], false, 0.0, LineCap::Butt), expected);
}

#[test]
fn cubic_width_05_aa_round() {
    let expected = Pixmap::load_png("tests/images/hairline/cubic-width-05-aa-round.png").unwrap();
    assert_eq!(draw_cubic(&[25.0, 80.0, 55.0, 25.0, 155.0, 75.0, 175.0, 20.0], true, 0.5, LineCap::Round), expected);
}

#[test]
fn cubic_width_00_round() {
    let expected = Pixmap::load_png("tests/images/hairline/cubic-width-00-round.png").unwrap();
    assert_eq!(draw_cubic(&[25.0, 80.0, 55.0, 25.0, 155.0, 75.0, 175.0, 20.0], false, 0.0, LineCap::Round), expected);
}

#[test]
fn chop_cubic_01() {
    let expected = Pixmap::load_png("tests/images/hairline/chop-cubic-01.png").unwrap();
    // This curve will invoke `path_geometry::chop_cubic_at_max_curvature` branch of `hair_cubic`.
    assert_eq!(draw_cubic(&[57.0, 13.0, 17.0, 15.0, 55.0, 97.0, 89.0, 62.0], true, 0.5, LineCap::Butt), expected);
}

#[test]
fn clip_cubic_05_aa() {
    let expected = Pixmap::load_png("tests/images/hairline/clip-cubic-05-aa.png").unwrap();
    assert_eq!(draw_cubic(&[-25.0, 80.0, 55.0, 25.0, 155.0, 75.0, 175.0, 20.0], true, 0.5, LineCap::Butt), expected);
}

#[test]
fn clip_cubic_00() {
    let expected = Pixmap::load_png("tests/images/hairline/clip-cubic-00.png").unwrap();
    assert_eq!(draw_cubic(&[-25.0, 80.0, 55.0, 25.0, 155.0, 75.0, 175.0, 20.0], false, 0.0, LineCap::Butt), expected);
}

#[test]
fn clipped_circle_aa() {
    let mut pixmap = Pixmap::new(100, 100).unwrap();

    let mut paint = Paint::default();
    paint.set_color_rgba8(50, 127, 150, 200);
    paint.anti_alias = true;

    let mut stroke = Stroke::default();
    stroke.width = 0.5;

    let path = PathBuilder::from_circle(50.0, 50.0, 55.0).unwrap();
    pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);

    let expected = Pixmap::load_png("tests/images/hairline/clipped-circle-aa.png").unwrap();
    assert_eq!(pixmap, expected);
}

#[test]
fn one_pixel_high_stroke() {
    let mut pixmap = Pixmap::new(64, 1).unwrap();
    let mut builder = PathBuilder::new();
    builder.move_to(0.0, 0.5);
    builder.line_to(64.0, 0.5);
    let path = builder.finish().unwrap();
    let mut paint = Paint::default();
    paint.set_color_rgba8(255, 0, 0, 255);
    pixmap.stroke_path(&path, &paint, &Stroke::default(), Transform::identity(), None);

    let mut expected = Pixmap::new(64, 1).unwrap();
    expected.fill(Color::from_rgba8(255, 0, 0, 255));
    assert_eq!(pixmap, expected);
}

#[test]
fn tiny_canvas_strokes_match_cropped_canvas() {
    // A small destination must clip a stroke, not discard it. A larger canvas
    // supplies the same rasterization with room for the hairline culling inset.
    // Keep endpoints outside both canvases so cap clipping does not affect it.
    for (width, height) in [(64, 1), (64, 2), (1, 64), (2, 64), (1, 1), (2, 2), (3, 3)] {
        for anti_alias in [false, true] {
            for stroke_width in [0.5, 1.0] {
                for line_cap in [LineCap::Butt, LineCap::Round, LineCap::Square] {
                    for curve in 0..3 {
                        let mut builder = PathBuilder::new();
                        let x = width as f32;
                        let y = height as f32;
                        if width < height {
                            builder.move_to(x * 0.5, -8.0);
                            match curve {
                                0 => builder.line_to(x * 0.5, y + 8.0),
                                1 => builder.quad_to(x * 0.5, y * 0.5, x * 0.5, y + 8.0),
                                _ => builder.cubic_to(x * 0.5, 0.0, x * 0.5, y, x * 0.5, y + 8.0),
                            }
                        } else {
                            builder.move_to(-8.0, y * 0.5);
                            match curve {
                                0 => builder.line_to(x + 8.0, y * 0.5),
                                1 => builder.quad_to(x * 0.5, y * 0.5, x + 8.0, y * 0.5),
                                _ => builder.cubic_to(0.0, y * 0.5, x, y * 0.5, x + 8.0, y * 0.5),
                            }
                        }
                        let path = builder.finish().unwrap();
                        let mut paint = Paint::default();
                        paint.set_color_rgba8(50, 127, 150, 200);
                        paint.anti_alias = anti_alias;
                        let stroke = Stroke {
                            width: stroke_width,
                            line_cap,
                            ..Stroke::default()
                        };
                        let mut actual = Pixmap::new(width, height).unwrap();
                        actual.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
                        let mut padded = Pixmap::new(width + 8, height + 8).unwrap();
                        padded.stroke_path(&path, &paint, &stroke, Transform::from_translate(4.0, 4.0), None);
                        let expected = padded.clone_rect(IntRect::from_xywh(4, 4, width, height).unwrap()).unwrap();
                        assert_eq!(actual, expected, "{width}x{height}, AA={anti_alias}, width={stroke_width}, cap={line_cap:?}, curve={curve}");
                    }
                }
            }
        }
    }
}
