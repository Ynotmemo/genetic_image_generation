use image::{DynamicImage, GenericImageView};
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use rand_isaac::isaac64::Isaac64Rng;
use rayon::prelude::*;


const POPULATIONS: usize = 10;
const SEED: u64 = 0;
const IMAGE_SIZE: [usize; 2] = [400; 2];

// 各個体の構造体を定義
struct Individual {
    genom_buffer: Vec<Array2<i32>>,
    fitness: i32,
}

impl Individual {
    fn new(genom_buffer: Vec<Array2<i32>>) -> Self {
        let individual = Individual {
            genom_buffer,
            fitness: 0,
        };
        // individual.set_fitness();
        individual
    }

    // 目標画像とのSSIMを算出
    fn set_fitness(&mut self) {
        unimplemented!()
    }
}

// 画像ファイルの読み込み
fn load_image(file_path: &str) -> DynamicImage {
    image::open(file_path).unwrap()
}

//画像サイズの変更
fn resize_image(image: DynamicImage, width: u32, height: u32) -> DynamicImage {
    image.resize_exact(width, height, image::imageops::FilterType::Lanczos3)
}

// 第一世代を生成
fn initialize_generation(populations: usize, image_size: [usize; 2]) -> Vec<Individual> {
    let seeds: Vec<u64> = (0..populations).map(|i| SEED.wrapping_add(i as u64)).collect();

    let generation: Vec<Individual> = seeds.into_par_iter().map(|seed| {
        let mut rng = Isaac64Rng::seed_from_u64(seed);
        let genom_buffer= [
            Array::random_using(image_size, Uniform::new(0, 256), &mut rng),
            Array::random_using(image_size, Uniform::new(0, 256), &mut rng),
            Array::random_using(image_size, Uniform::new(0, 256), &mut rng),
        ].to_vec();
        Individual::new(genom_buffer)
    }).collect();

    generation
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn create_the_first_generation() {
        let mut generation: Vec<Individual> =  initialize_generation(POPULATIONS, IMAGE_SIZE);

        assert_eq!(generation.len(), POPULATIONS);
        assert_eq!(generation[0].genom_buffer.len(), 3);
        assert_eq!(generation[0].genom_buffer[0].shape(), IMAGE_SIZE);
        assert_eq!(generation[0].genom_buffer[1].shape(), IMAGE_SIZE);
        assert_eq!(generation[0].genom_buffer[2].shape(), IMAGE_SIZE);
        assert_eq!(generation[0].fitness, 0);
    }

    #[test]
    fn load_resize_image() {
        let file_path = "./data/target_image.jpeg";
        let mut target_image = load_image(file_path);

        target_image = resize_image(target_image, IMAGE_SIZE[0] as u32, IMAGE_SIZE[1] as u32);
        let (width, height) = target_image.dimensions();
        assert_eq!([width as usize, height as usize], IMAGE_SIZE);
    }
}

fn main() { unimplemented!(); }
