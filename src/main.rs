use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use rand_isaac::isaac64::Isaac64Rng;

const POPULATIONS: usize = 100;
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

    // 目標画像との差分を算出
    fn set_fitness(&mut self) {
        unimplemented!()
    }
}

// 第一世代を生成
fn initialize_generation(populations: usize, image_size: [usize; 2]) -> Vec<Individual> {
    let mut rng = Isaac64Rng::seed_from_u64(SEED);
    let mut generation = Vec::new();

    for _ in 0..populations {
        let mut genom_buffer = Vec::new();
        genom_buffer.extend([
            Array::random_using(image_size, Uniform::new(0, 256), &mut rng),
            Array::random_using(image_size, Uniform::new(0, 256), &mut rng),
            Array::random_using(image_size, Uniform::new(0, 256), &mut rng),
        ].iter().cloned());
        let individual = Individual::new(genom_buffer);
        generation.push(individual);
    }
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
}

fn main() {
    unimplemented!();
}
