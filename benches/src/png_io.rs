use test::Bencher;

#[bench]
fn decode_rgb(bencher: &mut Bencher) {
    let filename = "tests/images/pngs/rgb.png";
    let data = std::fs::read(filename).unwrap_or_else(|err| {
        panic!(
            "{err}\nFile: {filename}\nCurrent Working Directory: {wd:?}",
            wd = std::env::current_dir()
        );
    });
    bencher.iter(|| {
        let _ = tiny_skia::Pixmap::decode_png(&data).unwrap();
    });
}

#[bench]
fn decode_rgba(bencher: &mut Bencher) {
    let data = std::fs::read("tests/images/pngs/rgba.png").unwrap();
    bencher.iter(|| {
        let _ = tiny_skia::Pixmap::decode_png(&data).unwrap();
    });
}

// Just a PNG decoding without preprocessing
// to see how much overhead our code has.
#[bench]
fn decode_raw_rgb(bencher: &mut Bencher) {
    let data = std::fs::read("tests/images/pngs/rgb.png").unwrap();
    let mut img_data = vec![0; 30000];
    bencher.iter(|| {
        let reader = std::io::BufReader::new(std::io::Cursor::new(&data));
        let decoder = png::Decoder::new(reader);
        let mut reader = decoder.read_info().unwrap();
        let _ = reader.next_frame(&mut img_data).unwrap();
    });
}

#[bench]
fn decode_raw_rgba(bencher: &mut Bencher) {
    let data = std::fs::read("tests/images/pngs/rgba.png").unwrap();
    let mut img_data = vec![0; 40000];
    bencher.iter(|| {
        let reader = std::io::BufReader::new(std::io::Cursor::new(&data));
        let decoder = png::Decoder::new(reader);
        let mut reader = decoder.read_info().unwrap();
        let _ = reader.next_frame(&mut img_data).unwrap();
    });
}

#[bench]
fn encode_rgba(bencher: &mut Bencher) {
    let pixmap = tiny_skia::Pixmap::load_png("tests/images/pngs/rgba.png").unwrap();
    bencher.iter(|| {
        let _ = pixmap.encode_png().unwrap();
    });
}

#[bench]
fn encode_raw_rgba(bencher: &mut Bencher) {
    let pixmap = tiny_skia::Pixmap::load_png("tests/images/pngs/rgba.png").unwrap();
    bencher.iter(|| {
        let mut data = Vec::new();

        let mut encoder = png::Encoder::new(&mut data, pixmap.width(), pixmap.height());
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();

        writer.write_image_data(pixmap.data()).unwrap();
    });
}
