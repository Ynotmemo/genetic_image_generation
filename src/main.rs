use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use rand_isaac::isaac64::Isaac64Rng;

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

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn create_instance_of_individual() {
        let mut rng = Isaac64Rng::seed_from_u64(SEED);
        let mut genom_buffer = Vec::new();

        genom_buffer.extend([
            Array::random_using(IMAGE_SIZE, Uniform::new(0, 256), &mut rng),
            Array::random_using(IMAGE_SIZE, Uniform::new(0, 256), &mut rng),
            Array::random_using(IMAGE_SIZE, Uniform::new(0, 256), &mut rng),
        ].iter().cloned());

        let individual = Individual::new(genom_buffer);

        println!("{:?}", individual.genom_buffer[0].shape());
        assert_eq!(individual.genom_buffer.len(), 3);
        assert_eq!(individual.genom_buffer[0].shape(), IMAGE_SIZE);
        assert_eq!(individual.genom_buffer[1].shape(), IMAGE_SIZE);
        assert_eq!(individual.genom_buffer[2].shape(), IMAGE_SIZE);
        assert_eq!(individual.fitness, 0);

    }
}

fn main() {
    unimplemented!();
}
