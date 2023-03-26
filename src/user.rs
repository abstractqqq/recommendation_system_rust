use chrono::prelude::*;

/// A user struct for recommendation enginee.
/// Not a user struct for a general e-commerce website. 
/// By default, this user is assumed to be active. 
/// Things like username, password are omitted because they do not have anything to do with recommendation.
///  
/// 
/// RecUser might be a mutable user for better "online update ability"

pub struct RecUser {
    user_id: usize,
    ratings: Vec<(usize, Rating)>,
    rec_queue: Vec<usize>, // current recommendation queue for this user
    last_updated: DataTime<Utc> // last time rec_queue was updated for this user
}

impl RecUser {
    pub fn new() {
        todo!();
    }

} 