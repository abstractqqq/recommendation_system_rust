use ndarray::{Axis, Array1, Array2, ArrayView1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// Regularized l2 cost/error
/// u: user matrix in training
/// v: movie matrix in training
/// y: target
/// reg_l2: l2 regularization factor
/// mask: mask of non-missing values in y
/// count: sum(mask)
/// returns (pt_diff * mask, L2 regularized error)
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

/// Get mask matrix from mask value
/// y: target
/// mask_value: the value used to mask
/// returns: the mask matrix
#[inline]
fn get_mask(y:&Array2<f64>, mask_value:f64) -> Array2<f64> {
    y.map(|v| (((v != &mask_value) as u8) * 1) as f64)
}

/// Performs Matrix Factorization using gradient descent. (Not stochastic gradient descent.)
/// y: target
/// K: num of latent features
/// alpha: learning rate
/// reg_l2: l2 regularization parameter
/// num_iters: number of iterations
/// threshold: convergence threshold
/// mask_value: the value you used for indicating missing values in y
pub fn matrix_factorization(y:&Array2<f64>, K:usize, alpha:f64, reg_l2:f64
    , num_iters:usize, threshold:f64, mask_value:f64
) -> Option<(Array2<f64>, Array2<f64>)> {
    
    let mask = get_mask(y, mask_value);
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

pub struct RecEngine {
    rating:Array2<f64>,
    mask:Array2<f64>,
    centered_rating:Array2<f64>,
    k: usize // the default value for the k in top_k similarities.
}

impl RecEngine {
    pub fn new(v:Array2<f64>, mask_value:f64, boundary:usize) -> RecEngine {
        let (mask, centered):(Array2<f64>, Array2<f64>) = Self::mask_and_center(&v, mask_value);
        RecEngine { rating: v, mask: mask, centered_rating: centered, k: boundary}
    }

    fn mask_and_center(y:&Array2<f64>, mask_value:f64) -> (Array2<f64>, Array2<f64>) {
        // let row_sum:Array1<f64> = y.sum_axis(Axis(1));
        // let row_count:Array1<f64> = mask.sum_axis(Axis(1));
        // add 0.01 to avoid dividing by 0
        let mask:Array2<f64> = get_mask(y, mask_value);
        let row_true_avg:Array1<f64> = y.sum_axis(Axis(1)) / (0.001 + mask.sum_axis(Axis(1)));
        let centered:Array2<f64> = (y - row_true_avg.insert_axis(Axis(1))) * &mask;
        (mask, centered)
    }

    // i, j should start from 0
    pub fn sim(&self, i:usize, j:usize) -> f64 {
        let vi:ArrayView1<f64> = self.centered_rating.index_axis(Axis(0), i); 
        let vj:ArrayView1<f64> = self.centered_rating.index_axis(Axis(0), j);         
        vi.dot(&vj) / (vi.dot(&vi).sqrt() * vj.dot(&vj).sqrt())
    }

    /// Get top k similar users to i
    pub fn get_top_k_sim(&self, i:usize, k:usize) -> Vec<(f64, usize)> {
        let mut output:Vec<(f64, usize)> = Vec::new();
        for row in 0..self.centered_rating.dim().0 {
            if row != i {
                output.push((self.sim(i, row), row));
            }
        }

        output.sort_by(|a, b| b.partial_cmp(a).unwrap());
        output.truncate(k);       
        output
    }

    #[inline]
    pub fn get_top_k_sim_users(&self, i:usize, k:usize) -> Vec<usize> {
        self.get_top_k_sim(i, k).into_iter().map(|t| t.1).collect()
    }

    /// Get prediction based on centered cosine similarity
    pub fn get_prediction_cosine(&self) -> Array2<f64> {
        let mut pred:Array2<f64> = self.rating.clone();
        let (row_count, col_count):(usize, usize) = self.rating.dim();
        for i in 0..row_count {
            for j in 0..col_count {
                if self.mask.get((i,j)).unwrap() == &0. {
                    // This is the i-th user. Need to rate the j-th movie.
                    pred[[i,j]] = 1.; // do this later
                    todo!()
                }
            }
        }


        pred  
    }





}