use ndarray::{Axis, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// Regularized l2 cost
fn l2_cost(u:&Array2<f64>, v:&Array2<f64>, y:&Array2<f64>, reg_l2:f64, mask:&Array2<f64>, count:&f64) -> (Array2<f64>, f64) {
    let pt_diff:Array2<f64> = (y - u.dot(&v.t())) * mask; // pointwise multiply with mask
    // squared sum of entries in pt_diff, normalized by count
    let err = (pt_diff.map_axis(Axis(1), |v| v.dot(&v))).sum() / count;
    let reg_term:f64 = reg_l2 * (
        u.map_axis(Axis(1), |v| v.dot(&v)).sum() 
        + v.map_axis(Axis(1), |v| v.dot(&v)).sum() 
    );// reg_l2 * (squared sum of entries in u + squared sum of entries in v)
    (pt_diff, 0.5*(err + reg_term)) // add in the 1/2 scaling factor
}

pub fn matrix_factorization(y:&Array2<f64>, K:usize, alpha:f64, reg_l2:f64
    , num_iters:usize, threshold:f64, mask_value:f64
) -> Option<(Array2<f64>, Array2<f64>)> {
    
    let mask = y.map(|v| (((v != &mask_value) as u8) * 1) as f64);
    let count = mask.sum();
    if count < 0. {
        // error. There is not a single rating.
        return None
    }

    let (U, M) = y.dim(); // U: user count, M: movie count / Item count
    let mut u:Array2<f64> = Array2::random((U, K),Uniform::new(0., 1.));
    let mut v:Array2<f64> = Array2::random((M, K),Uniform::new(0., 1.));
    let scale_factor:f64 = alpha / count;
    let scale_reg:f64 = alpha * reg_l2 * 0.5;

    let mut exit_mode:bool = false; // end of iteration. No convergence.
    for i in 0..num_iters {

        let (delta, err) = l2_cost(&u, &v, y, reg_l2, &mask, &count);
        //println!("Step {}, error: {}", i+1, err);
        if err < threshold { // error needs to be normalized, err/count is the true average error.
            exit_mode = true; // Converges. Break to exit.
            println!("Converged at iteration {}, with error: {}", i+1, err);
            break
        }
        // Can try multithreading here. But maybe not worth it.
        let scaled_delta:Array2<f64> = scale_factor * delta;
        let u_change:Array2<f64> = scaled_delta.dot(&v) - scale_reg * &u;
        let v_change:Array2<f64> = scaled_delta.t().dot(&u) - scale_reg * &v;
        u = u + u_change;
        v = v + v_change;

    }

    if exit_mode {
        Some((u, v))
    } else {// Print
        println!("Failed to reach threshold {} in {} iterations.", threshold, num_iters );
        None
    }

    


}