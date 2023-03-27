use ndarray::{Axis, Array1, Array2, ArrayView2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

// WILL SOON BE UNDER RECONSTRUCTION

pub enum InitStrategy {
    MinMax((f32, f32)), // a uniform dist from min to max
    Constant(f32)
}

pub struct RecEngine {
    rating:Array2<f32>,
    mask:Array2<f32>, // ideally, should be sparse
    mask_value:f32,
    sim_table:Array2<f32>,
    k: usize // the default value for the k in top_k similarities.
}

impl RecEngine {
    pub fn new(v:Array2<f32>, mask_value:f32, top_k:usize) -> RecEngine {
        let (mask, centered):(Array2<f32>, Array2<f32>) = Self::mask_and_center(&v, mask_value);
        let sim_table:Array2<f32> = Self::get_sim_table(centered);
        RecEngine { rating: v, mask: mask, mask_value: mask_value, sim_table: sim_table, k: top_k}
    }

    fn mask_and_center(y:&Array2<f32>, mask_value:f32) -> (Array2<f32>, Array2<f32>) {
        // let row_sum:Array1<f32> = y.sum_axis(Axis(1));
        // let row_count:Array1<f32> = mask.sum_axis(Axis(1));
        // add 0.01 to avoid dividing by 0
        let mask:Array2<f32> = y.map(|v| (( v != &mask_value) as u8) as f32);
        let row_true_avg:Array1<f32> = y.sum_axis(Axis(1)) / (0.001 + mask.sum_axis(Axis(1)));
        let centered:Array2<f32> = (y - row_true_avg.insert_axis(Axis(1))) * &mask;
        (mask, centered)
    }

    #[inline]
    pub fn sim(&self, i:usize, j:usize) -> Option<f32> {
        self.sim_table.get((i,j)).copied()
    }

    #[inline]
    pub fn get_sim(&self) -> ArrayView2<f32> {
        self.sim_table.view()
    }

    #[inline]
    pub fn get_mask_value(&self) -> f32 {
        self.mask_value.clone()
    }

    fn get_sim_table(centered:Array2<f32>) -> Array2<f32> {
        let normalized = Self::normalize(centered);
        normalized.dot(&normalized.t())
    }

    #[inline]
    fn normalize(mat:Array2<f32>) -> Array2<f32>{
        // Divide mat row-wise by the row norms
        &mat / (// compute L2 norm of each axis
            mat.map_axis(Axis(1), |view| (view.dot(&view)).sqrt())
            .insert_axis(Axis(1)) // turn 1D into 2D
        )
    }

    /// Get top k similar users to i
    /// Potentially I can use some data structure to speed this up? Similarity
    /// table can definitely be precomputed hmm... Then I just need to query from
    /// the table.
    pub fn get_top_k_sim(&self, i:usize, k:usize) -> Vec<(f32, usize)> {
        let sim:Vec<f32> = self.sim_table.index_axis(Axis(1), i).to_vec();
        let n:usize = sim.len();
        let mut output:Vec<(f32, usize)> = (sim.into_iter().zip(0..n)).filter(|t| t.1 != i).collect();
        output.sort_by(|a, b| b.partial_cmp(a).unwrap());
        output.truncate(k);  
        output
    }

    /// Get top k similar users to i, but each user must have rated movie j
    pub fn get_top_k_sim_with_rating(&self, i:usize, k:usize, j:usize) -> Vec<(f32, usize)> {
        let sim:Vec<f32> = self.sim_table.index_axis(Axis(1), i).to_vec();
        let n:usize = sim.len();
        let mut output:Vec<(f32, usize)> = (sim.into_iter().zip(0..n)).filter(|t|
            !(self.mask[[t.1, j]] == 0. || t.1 == i)
        ).collect();

        output.sort_by(|a, b| b.partial_cmp(a).unwrap());
        output.truncate(k);       
        output
    }

    #[inline]
    pub fn get_top_k_sim_users(&self, i:usize, k:usize) -> Vec<usize> {
        self.get_top_k_sim(i, k).into_iter().map(|t| t.1).collect()
    }

    /// Get prediction based on centered cosine similarity
    pub fn get_prediction_cosine(&self) -> Array2<f32> {
        let mut pred:Array2<f32> = self.rating.clone();
        let (row_count, col_count):(usize, usize) = self.rating.dim();
        // How to make this run faster? 
        for i in 0..row_count {
            for j in 0..col_count {
                if self.mask[[i,j]] == 0. {
                    // This is the i-th user. Need to rate the j-th movie.
                    // Computed weighted average
                    // This returns a tuple (Sum weight * rating, Sum of all weights as a normalizing factor)
                    let weighted_avg_temp = self.get_top_k_sim_with_rating(i, self.k, j)
                        .into_iter().fold((0.,0.), |acc, s| {
                            (acc.0 + s.0 * self.rating[[s.1,j]], acc.1 + s.0)
                        });
                    // If somehow the weighted_rating is < 0, then keep mask_value
                    pred[[i,j]] = (weighted_avg_temp.0 / weighted_avg_temp.1).max(self.mask_value);
                }
            }
        }
        pred
    }

    /// Performs Matrix Factorization using gradient descent. (Not stochastic gradient descent.)
    /// y: target
    /// K: num of latent features
    /// alpha: learning rate
    /// reg_l2: l2 regularization parameter
    /// num_iters: number of iterations
    /// threshold: convergence threshold
    /// mask_value: the value you used for indicating missing values in y
    /// returns: will always return something
    pub fn get_prediction_mf(&self, K:usize, alpha:f32, reg_l2:f32, num_iters:usize, threshold:f32, init:InitStrategy
    ) -> (Array2<f32>, Array2<f32>) {
        let count = self.mask.sum();
        let epsilon:f32 = 0.001;
        let mut current_err:f32 = 0.;
        let (U, M) = self.rating.dim(); // U: user count, M: movie count / Item count
        
        let mut u:Array2<f32>;
        let mut v:Array2<f32>;
        match init {
            InitStrategy::Constant(f) => {
                u = Array2::from_elem((U,K), f);
                v = Array2::from_elem((M,K), f);
            }
            InitStrategy::MinMax((min, max)) => {
                u = Array2::random((U, K),Uniform::new(min, max));
                v = Array2::random((M, K),Uniform::new(min, max));
            }
        }
        let scale_factor:f32 = alpha / count;
        let scale_reg:f32 = alpha * reg_l2 * 0.5;
        if count < 0. {
            // error. There is not a single filled rating. return initial values
            return (u,v)
        }
        for i in 0..num_iters {
    
            let (delta, err) = self.l2_cost(&u, &v, reg_l2, &count);
            let change_in_err:f32 = (current_err - err).abs();
            current_err = err;
            //println!("Step {}, error: {}", i+1, err);
            if err < threshold { // error needs to be normalized, err/count is the true average error.
                println!("Converged at iteration {}, with error: {}", i+1, err);
                return (u , v)
            } else if change_in_err < epsilon {
                println!("Error improvement is {}, which is smaller than {}, at iteration {}. ", change_in_err, epsilon, i+1,);
                return (u , v)       
            }
            let scaled_delta:Array2<f32> = scale_factor * delta;
            let u_change:Array2<f32> = scaled_delta.dot(&v) - scale_reg * &u;
            let v_change:Array2<f32> = scaled_delta.t().dot(&u) - scale_reg * &v;
            u = u + u_change;
            v = v + v_change;
    
        }
    
        println!("Finished running {} iterations. Current error is {}, which is > the given threshold {}.", num_iters, current_err, threshold);
        println!("The algorithm is still improving more than {} per step. It is recommended that you add more iterations or lower the error threshold.", epsilon);
        (u,v)
    
    }

    /// Regularized l2 cost/error
    /// u: user matrix in training
    /// v: movie matrix in training
    /// y: target
    /// reg_l2: l2 regularization factor
    /// mask: mask of non-missing values in y
    /// count: sum(mask)
    /// returns (pt_diff * mask, L2 regularized error)
    fn l2_cost(&self, u:&Array2<f32>, v:&Array2<f32>, reg_l2:f32, count:&f32) -> (Array2<f32>, f32) {
        let pt_diff:Array2<f32> = (&self.rating - u.dot(&v.t())) * &self.mask; // pointwise multiply with mask
        // squared sum of entries in pt_diff, normalized by count
        let err = (pt_diff.map_axis(Axis(1), |v| v.dot(&v))).sum() / count;
        let reg_term:f32 = reg_l2 * (
            u.map_axis(Axis(1), |v| v.dot(&v)).sum() 
            + v.map_axis(Axis(1), |v| v.dot(&v)).sum() 
        );// reg_l2 * (squared sum of entries in u + squared sum of entries in v)
        (pt_diff, 0.5*(err + reg_term)) // add in the 1/2 scaling factor
    }


}