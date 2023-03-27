mod rec_sys;
use rec_sys::{RecEngine, InitStrategy};
mod rating;
// use chrono::prelude::*;
use ndarray::prelude::*;


fn main() {
    let ratings = array![
        [5.,3.,0.,4.,2.],
        [4.,0.,2.,1.,3.],
        [1.,1.,0.,1.,0.],
        [1.,0.,3.,4.,1.],
        [0.,3.,3.,4.,0.],
        [5.,3.,0.,3.,3.],
        [5.,3.,2.,3.,4.],
        [4.,2.,2.,4.,0.],
        [1.,3.,3.,5.,0.],
        [4.,2.,2.,5.,0.],
        [1.,2.,1.,5.,1.],
    ];

    let r = RecEngine::new(ratings, 0., 5);
    // println!("Similar to User 1 and have rated movie 3:\n {:?}", r.get_top_k_sim_with_rating(0, 3, 2));
    println!("Prediction (Cosine): {:?}", r.get_prediction_cosine());

    let (p,q):(Array2<f32>, Array2<f32>) = r.get_prediction_mf(4, 0.1, 0.002,5000, 0.1
        , InitStrategy::MinMax((0., 1.)));
    println!("Prediction (MF) {}", p.dot(&q.t()));



}
