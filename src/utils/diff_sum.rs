use image::{ImageBuffer, Rgb};
use rayon::prelude::*;

pub fn calculate_diff_sum(img1: &ImageBuffer<Rgb<u8>, Vec<u8>>, img2: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> f64 {
    let (width, height) = img1.dimensions();

    let diff_sum = img1
        .pixels()
        .zip(img2.pixels())
        .collect::<Vec<_>>() // Zipされたイテレータをベクタに変換する
        .par_iter() // パラレルにイテレーションする
        .fold(
            || 0u32,
            |mut sum, (p1, p2)| {
                let r_diff = p1[0] as i32 - p2[0] as i32;
                let g_diff = p1[1] as i32 - p2[1] as i32;
                let b_diff = p1[2] as i32 - p2[2] as i32;
                sum += (r_diff.abs() + g_diff.abs() + b_diff.abs()) as u32;
                sum
            },
        )
        .sum::<u32>() as f64;

    diff_sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::processing_image;

    #[test]
    fn eva_calculate_diff_sum() {
        let file_path = "./data/target_image.jpeg";
        let target_image = processing_image::load_image(file_path);
        let target_image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = target_image.to_rgb8();;
        let fitness = calculate_diff_sum(&target_image_buffer, &target_image_buffer);
        assert_eq!(fitness, 0.0);
    }
}
