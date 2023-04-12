use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;

const POPULATIONS: usize = 10;
const SEED: u64 = 0;
const IMAGE_SIZE: (u32, u32) = (400, 400);

// 各個体の構造体を定義
struct Individual {
    genom_buffer: ImageBuffer<Rgb<u8>, Vec<u8>>,
    fitness: i32,
}

impl Individual {
    fn new(genom_buffer: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Self {
        Individual {
            genom_buffer,
            fitness: 0,
        }
    }

    // 目標画像とのSSIMを算出
    fn set_fitness(&mut self) {
        unimplemented!()
    }
}

// 画像ファイルの読み込み
fn load_image(file_path: &str) -> DynamicImage {
    image::open(file_path).expect("Failed to open image")
}

// 画像サイズの変更
fn resize_image(image: DynamicImage, width: u32, height: u32) -> DynamicImage {
    image.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
}

// 第一世代を生成
fn initialize_generation(populations: usize, image_size: (u32, u32), seed: u64) -> Vec<Individual> {
    let rng = StdRng::seed_from_u64(seed);
    let uniform = Uniform::new(0, 255);
    let generation: Vec<Individual> = (0..populations)
        .into_par_iter()
        .map(|_| {
            let mut rng_clone = rng.clone();
            let genom_buffer = ImageBuffer::from_fn(image_size.0, image_size.1, |_x, _y| {
                Rgb([
                    rng_clone.sample(uniform),
                    rng_clone.sample(uniform),
                    rng_clone.sample(uniform),
                ])
            });
            Individual::new(genom_buffer)
        })
        .collect();
    generation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_the_first_generation() {
        let generation: Vec<Individual> = initialize_generation(POPULATIONS, IMAGE_SIZE, SEED);

        assert_eq!(generation.len(), POPULATIONS);
        assert_eq!(generation[0].genom_buffer.dimensions(), IMAGE_SIZE);
        assert_eq!(generation[0].fitness, 0);
    }

    #[test]
    fn load_resize_image() {
        let file_path = "./data/target_image.jpeg";
        let target_image = load_image(file_path);
        let resized_target_image = resize_image(target_image, IMAGE_SIZE.0, IMAGE_SIZE.1);
        assert_eq!(resized_target_image.dimensions(), IMAGE_SIZE);
    }
}

fn main() {
    unimplemented!();
}
