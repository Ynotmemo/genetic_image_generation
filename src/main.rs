use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use std::path::Path;

mod utils;

const POPULATIONS: usize = 10;
const SEED: u64 = 0;
const IMAGE_SIZE: (u32, u32) = (400, 400);
const CROSS_RATE: f64 = 0.5;

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

// 2個体を交叉させて新しい個体を生成する関数
fn crossover_individuals(individual1: &Individual, individual2: &Individual, image_size: (u32, u32), cross_rate: f64) -> Individual {

    let (width, height) = image_size;
    let pixels: Vec<_> = (0..height)
        .into_par_iter()
        .flat_map(|h| {
            (0..width)
                .into_par_iter()
                .map(move |w| {
                    let mut rng = thread_rng();
                    let choice = rng.gen_bool(cross_rate);
                    let pixel = if choice {
                        *individual1.genom_image_buffer.get_pixel(w, h)
                    } else {
                        *individual2.genom_image_buffer.get_pixel(w, h)
                    };
                    pixel
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let pixels_u8: Vec<u8> = pixels.into_iter().map(|p| p.0).flatten().collect();
    let genom_image_buffer = ImageBuffer::from_vec(width, height, pixels_u8).unwrap();
    Individual::new(genom_image_buffer)
}

// 次の世代を生成
fn generate_next_generation(individual1: &Individual, individual2: &Individual, image_size: (u32, u32), cross_rate: f64, populations: usize, ) -> Vec<Individual> {
    let crossed_images: Vec<Individual> = (0..populations)
        .into_par_iter()
        .map(|_| crossover_individuals(individual1, individual2, image_size, cross_rate))
        .collect();
    crossed_images
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
    fn eva_calc_fitness() {
        let file_path = "./data/target_image.jpeg";
        let target_image = utils::processing_image::load_image(file_path);
        let resized_target_image = utils::processing_image::resize_image(target_image, IMAGE_SIZE.0, IMAGE_SIZE.1);
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


fn main() -> Result<(), Box<dyn std::error::Error>> {


    // unimplemented!();
    let dad_image = utils::processing_image::load_image("./data/target_image.jpeg");
    let mom_image = utils::processing_image::load_image("./data/temp_image.jpg");
    let resized_dad = utils::processing_image::resize_image(dad_image, IMAGE_SIZE.0, IMAGE_SIZE.1);
    let resized_mom = utils::processing_image::resize_image(mom_image, IMAGE_SIZE.0, IMAGE_SIZE.1);
    let dad = Individual::new(utils::processing_image::dynamic_image_to_image_buffer(&resized_dad));
    let mom = Individual::new(utils::processing_image::dynamic_image_to_image_buffer(&resized_mom));
    let mut child = crossover_individuals(&dad, &mom, IMAGE_SIZE, CROSS_RATE);
    child.calc_fitness(&resized_dad);
    println!("similarity to dad: {}", child.fitness);
    child.calc_fitness(&resized_mom);
    println!("similarity to mom: {}", child.fitness);

    utils::processing_image::save_dynamic_image_to_png(&child.genom_dynamic_image, "./results/child.png")?;

    Ok(())
}
