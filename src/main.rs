use chrono::Local;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::Rng;
use rand_isaac::isaac64::Isaac64Rng;
use rayon::prelude::*;
use std::path::Path;

mod utils;
use utils::ssim::calculate_ssim;
use utils::processing_image::{dynamic_image_to_image_buffer, load_image, resize_image, save_dynamic_image_to_png};

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
    let _seeds: Vec<u64> = (0..populations).map(|i| seed.wrapping_add(i as u64)).collect();
    let uniform: Uniform<u8> = Uniform::new(0, 255);

    let generation: Vec<Individual> = _seeds
        .into_par_iter()
        .map(|s| {
            let mut rng = Isaac64Rng::seed_from_u64(s);
            let genom_image_buffer = ImageBuffer::from_fn(image_size.0, image_size.1, |_x, _y| {
                Rgb([
                    rng.sample(uniform),
                    rng.sample(uniform),
                    rng.sample(uniform),
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
    let mut pixels = Vec::new();
    let mut rng = rand::thread_rng();
    for h in 0..height {
        for w in 0..width {
            let choice = rng.gen_bool(cross_rate);
            let pixel = if choice {
                *individual1.genom_image_buffer.get_pixel(w, h)
            } else {
                *individual2.genom_image_buffer.get_pixel(w, h)
            };
            pixels.push(pixel);
        }
    }

    let pixels_u8: Vec<u8> = pixels.into_iter().map(|p| p.0).flatten().collect();
    let genom_image_buffer = ImageBuffer::from_vec(width, height, pixels_u8).unwrap();
    Individual::new(genom_image_buffer)
}


// 次の世代を生成
fn generate_next_generation(individual1: &Individual, individual2: &Individual, image_size: (u32, u32), cross_rate: f64, populations: usize) -> Vec<Individual> {
    let mut crossed_images = Vec::with_capacity(populations);
    for _ in 0..populations {
        let crossed_image = crossover_individuals(individual1, individual2, image_size, cross_rate);
        crossed_images.push(crossed_image);
    }
    crossed_images
}


fn main() {
    let population: usize = 10;
    let max_generations: usize = 1000;
    let cross_rate: f64 = 0.2;
    let image_size: (u32, u32) = (3, 3);
    let file_path: &str = "./data/target_image.jpeg";

    let target_image: DynamicImage = load_image(file_path);

    let resized_target_image: DynamicImage = resize_image(target_image, image_size.0, image_size.1);
    save_dynamic_image_to_png(&resized_target_image, "./results/resized_target_image.png").expect("save fig error");
    // let image_size: (u32, u32) = (target_image.width(), target_image.height());

    // 現在の年月日をシード値に設定する
    let seed: u64 = Local::now().format("%Y%m%d").to_string().parse::<u64>().unwrap();

    // 第一世代の生成
    let mut current_generation: Vec<Individual> = initialize_generation(population, image_size, seed);


    // 世代ごとのループ
    for generation_count in 1..=(max_generations+1) {
        println!("Generation: {}", generation_count);
        current_generation
            .iter_mut()
            .for_each(|individual| {
                individual.calc_fitness(&resized_target_image);
            });
        let (largest_individual, second_largest_individual) = get_largest_two_fitness(&current_generation);
        // イテレーション10回毎に
        if generation_count % 10 == 1{
            // 最高適合率の出力
            println!("the best similarity to target image: {}", largest_individual.fitness);
        // 最善画像の保存
            save_dynamic_image_to_png(&largest_individual.genom_dynamic_image, &format!("./results/iteration{}.png", generation_count)).expect("save fig error"); // 修正
        }
        // 次世代の生成
        let mut next_generation: Vec<Individual> = generate_next_generation(&largest_individual, &second_largest_individual, image_size, cross_rate, population);

        //世代交代
        current_generation = next_generation;
        }
}



#[cfg(test)]
mod tests {
    use super::*;
    const POPULATIONS: usize = 10;
    const SEED: u64 = 0;
    const IMAGE_SIZE: (u32, u32) = (400, 400);
    const CROSS_RATE: f64 = 0.5;

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
