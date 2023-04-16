use image::GrayImage;

// 画像をグレースケールに変換する関数
fn convert_to_grayscale(image: &image::DynamicImage) -> GrayImage {
    image.to_luma8()
}

// 画像の輝度の平均値を計算する関数
fn calculate_luma_mean(image: &GrayImage) -> f64 {
    let sum: f64 = image.iter().map(|&p| f64::from(p)).sum();
    let count = image.width() * image.height() as u32;
    sum / f64::from(count)
}

// 画像の輝度の標準偏差を計算する関数
fn calculate_luma_std_dev(image: &GrayImage, luma_mean: f64) -> f64 {
    let sum_squares: f64 = image
        .iter()
        .map(|&p| f64::from(p).powi(2))
        .sum();
    let count = image.width() * image.height() as u32;
    let variance = (sum_squares / f64::from(count)) - luma_mean.powi(2);
    variance.sqrt()
}

// 画像の構造の平均値を計算する関数
fn calculate_structure_mean(image: &GrayImage, luma_mean: f64) -> f64 {
    let sobel_x = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    let sobel_y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

    let sum_x: f64 = image
        .pixels()
        .zip(sobel_x.iter())
        .map(|(p, k)| f64::from(p[0]) * f64::from(*k))
        .sum();

    let sum_y: f64 = image
        .pixels()
        .zip(sobel_y.iter())
        .map(|(p, k)| f64::from(p[0]) * f64::from(*k))
        .sum();

    let count = image.width() * image.height() as u32;
    let structure_mean = (sum_x.powi(2) + sum_y.powi(2)).sqrt() / f64::from(count) / luma_mean;
    structure_mean
}


// 画像の輝度の相互関係を計算する関数
fn calculate_luma_similarity(luma_mean1: f64, luma_mean2: f64) -> f64 {
    let luma_mean_similarity = 2.0 * luma_mean1 * luma_mean2 / (luma_mean1.powi(2) + luma_mean2.powi(2));
    luma_mean_similarity
}

// 画像のコントラストの相互関係を計算する関数
fn calculate_contrast_similarity(luma_std_dev1: f64, luma_std_dev2: f64) -> f64 {
    let contrast_similarity = 2.0 * luma_std_dev1 * luma_std_dev2 / (luma_std_dev1.powi(2) + luma_std_dev2.powi(2));
    contrast_similarity
}

// 画像の構造の相互関係を計算する関数
fn calculate_structure_similarity(structure_mean1: f64, structure_mean2: f64) -> f64 {
    let structure_similarity = 2.0 * structure_mean1 * structure_mean2 / (structure_mean1.powi(2) + structure_mean2.powi(2));
    structure_similarity
}

// SSIMを計算する関数
pub fn calculate_ssim(image1: &image::DynamicImage, image2: &image::DynamicImage) -> f64 {
    let gray_image1 = convert_to_grayscale(image1);
    let gray_image2 = convert_to_grayscale(image2);

    let luma_mean1 = calculate_luma_mean(&gray_image1);
    let luma_mean2 = calculate_luma_mean(&gray_image2);

    let luma_std_dev1 = calculate_luma_std_dev(&gray_image1, luma_mean1);
    let luma_std_dev2 = calculate_luma_std_dev(&gray_image2, luma_mean2);

    let structure_mean1 = calculate_structure_mean(&gray_image1, luma_mean1);
    let structure_mean2 = calculate_structure_mean(&gray_image2, luma_mean2);

    let luma_similarity = calculate_luma_similarity(luma_mean1, luma_mean2);
    let contrast_similarity = calculate_contrast_similarity(luma_std_dev1, luma_std_dev2);
    let structure_similarity = calculate_structure_similarity(structure_mean1, structure_mean2);

    let ssim = luma_similarity * contrast_similarity * structure_similarity;
    ssim
}


#[cfg(test)]
mod tests {
    use super::*;

    use crate::utils::processing_image;

    #[test]
    fn eva_calculate_ssim() {
        let file_path = "./data/target_image.jpeg";
        let target_image = processing_image::load_image(file_path);
        let ssim = calculate_ssim(&target_image, &target_image);
        assert_eq!(ssim, 1 as f64)
    }
}
