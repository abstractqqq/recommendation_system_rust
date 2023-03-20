mod rec_sys;
use rec_sys::matrix_factorization;
use ndarray::prelude::*;


fn main() {
    let ratings = array![
        [5.,3.,0.,4.],
        [4.,0.,4.,4.],
        [1.,1.,0.,1.],
        [1.,0.,3.,4.],
        [0.,3.,3.,4.],
        [2.,1.,3.,0.]
    ];

    let result = matrix_factorization(&ratings, 3, 0.1, 0.002,5000, 0.1, 0.);
    match result {
        Some(res) => {
            let (p, q) = res;
            println!("P is {}", p);
            println!("Q is {}", q);
            println!("Their product is {}", p.dot(&q.t()));
        },
        _ => {
            print!("Failed to find factorization");
        }
    }

}
