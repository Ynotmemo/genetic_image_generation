use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use rand::distributions::Uniform;
use rand::prelude::*;
use rayon::prelude::*;

mod utils;

const POPULATIONS: usize = 10;
const SEED: u64 = 0;
const IMAGE_SIZE: (u32, u32) = (400, 400);

// 各個体の構造体を定義
#[derive(Clone)]
struct Individual {
    genom_image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>>,
    genom_dynamic_image: DynamicImage,
    fitness: f64,
}

impl Individual {
    fn new(genom_image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Self {
        let genom_dynamic_image = DynamicImage::ImageRgb8(genom_image_buffer.clone());
        Individual {
            genom_image_buffer,
            genom_dynamic_image,
            fitness: 0.0,
        }
    }

    // 目標画像とのSSIMを算出
    fn calc_fitness(&mut self, target_image: &DynamicImage) {
        self.fitness = utils::ssim::calculate_ssim(target_image, &self.genom_dynamic_image)
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
            let genom_image_buffer = ImageBuffer::from_fn(image_size.0, image_size.1, |_x, _y| {
                Rgb([
                    rng_clone.sample(uniform),
                    rng_clone.sample(uniform),
                    rng_clone.sample(uniform),
                ])
            });
            Individual::new(genom_image_buffer)
        })
        .collect();
    generation
}

// 世代ごとに適応度の高い２個体を選択
fn get_largest_two_fitness(generation: &[Individual]) -> (Individual, Individual) {
    // 世代の長さを取得
    let len = generation.len();

    // 世代が空または長さが1の場合はエラーを返す
    if len == 0 {
        panic!("No individuals in the generation.");
    } else if len == 1 {
        panic!("At least two individuals are required in the generation.");
    }

    // 適応度の降順でソート
    let mut sorted_generation = generation.to_vec();
    sorted_generation.sort_unstable_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

    // 適応度が最も大きい２つの個体を選択
    let largest_individual = sorted_generation[0].clone();
    let second_largest_individual = sorted_generation[1].clone();

    (largest_individual, second_largest_individual)
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_the_first_generation() {
        let generation: Vec<Individual> = initialize_generation(POPULATIONS, IMAGE_SIZE, SEED);

        assert_eq!(generation.len(), POPULATIONS);
        assert_eq!(generation[0].genom_image_buffer.dimensions(), IMAGE_SIZE);
        assert_eq!(generation[0].fitness, 0.0);
    }

    #[test]
    fn load_resize_image() {
        let file_path = "./data/target_image.jpeg";
        let target_image = load_image(file_path);
        let resized_target_image = resize_image(target_image, IMAGE_SIZE.0, IMAGE_SIZE.1);
        assert_eq!(resized_target_image.dimensions(), IMAGE_SIZE);
    }

    #[test]
    fn eva_calculate_ssim() {
        let file_path = "./data/target_image.jpeg";
        let target_image = load_image(file_path);
        let ssim = utils::ssim::calculate_ssim(&target_image, &target_image);
        assert_eq!(ssim, 1 as f64)
    }

    #[test]
    fn eva_calc_fitness() {
        let file_path = "./data/target_image.jpeg";
        let target_image = load_image(file_path);
        let resized_target_image = resize_image(target_image, IMAGE_SIZE.0, IMAGE_SIZE.1);
        let mut generation: Vec<Individual> = initialize_generation(POPULATIONS, IMAGE_SIZE, SEED);

        generation.par_iter_mut()
            .for_each(|individual| {
                individual.calc_fitness(&resized_target_image);
                assert!((0.0 <= individual.fitness) && (individual.fitness <= 1.0));
            });
    }

    #[test]
    fn choice_large_two_fitness() {
        let mut generation: Vec<Individual> = initialize_generation(POPULATIONS, IMAGE_SIZE, SEED);

        generation[2].fitness = 0.88;
        generation[5].fitness = 1.0;
        generation[7].fitness = 0.91;
        generation[9].fitness = 0.82;

        let (largest_individual, second_largest_individual) =  get_largest_two_fitness(&generation);
        assert_eq!(largest_individual.fitness, generation[5].fitness);
        assert_eq!(second_largest_individual.fitness, generation[7].fitness);
    }
}

fn main() { unimplemented!(); }
