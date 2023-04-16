use image::{DynamicImage, ImageBuffer, Rgb};

// 画像ファイルの読み込み
pub fn load_image(file_path: &str) -> DynamicImage {
    image::open(file_path).expect("Failed to open image")
}

// 画像サイズの変更
pub fn resize_image(image: DynamicImage, width: u32, height: u32) -> DynamicImage {
    image.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
}

pub fn dynamic_image_to_image_buffer(dynamic_image: &DynamicImage) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let rgb_image = dynamic_image.to_rgb8();
    let (width, height) = rgb_image.dimensions();
    let pixels: Vec<_> = rgb_image.into_raw();
    ImageBuffer::from_vec(width, height, pixels).unwrap()
}

pub fn save_dynamic_image_to_png(dynamic_image: &DynamicImage, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = std::fs::File::create(file_path)?;
    dynamic_image.write_to(&mut file, image::ImageOutputFormat::Png)?;
    Ok(())
}
