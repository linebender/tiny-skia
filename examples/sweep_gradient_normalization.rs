use tiny_skia::{
    Color, GradientStop, Paint, Pixmap, Point, Rect, SpreadMode, SweepGradient, Transform,
};

fn main() {
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sweep-gradient.png".into());
    let paint = Paint {
        shader: SweepGradient::new(
            Point::from_xy(128.5, 128.5),
            0.0,
            180.0,
            vec![
                GradientStop::new(0.0, Color::from_rgba8(255, 0, 0, 255)),
                GradientStop::new(1.0, Color::from_rgba8(0, 0, 255, 255)),
            ],
            SpreadMode::Pad,
            Transform::identity(),
        )
        .unwrap(),
        anti_alias: false,
        ..Paint::default()
    };
    let mut pixmap = Pixmap::new(256, 256).unwrap();
    pixmap.fill_rect(
        Rect::from_xywh(0.0, 0.0, 256.0, 256.0).unwrap(),
        &paint,
        Transform::identity(),
        None,
    );
    pixmap.save_png(&output).unwrap();

    // Clockwise screen angles: down is 90 degrees, left is 180 degrees.
    println!(
        "90° midpoint (expected approximately R128 G0 B128): {:?}",
        pixmap.pixel(128, 224).unwrap().demultiply()
    );
    println!(
        "180° endpoint (expected R0 G0 B255): {:?}",
        pixmap.pixel(32, 128).unwrap().demultiply()
    );
    println!(
        "270° padded region (expected R0 G0 B255): {:?}",
        pixmap.pixel(128, 32).unwrap().demultiply()
    );
    println!("Wrote {output}");
}
