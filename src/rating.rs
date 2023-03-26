use chrono::prelude::*;
struct Rating {
    user_id: usize,
    movie_id: usize,
    rating_id: usize, // not necessary. But would be nice if a user can change ratings.
    stars: f32, // let's assume stars can only be from 0 to 10. 
    utc: DateTime<Utc>,
    // For simplicity, non-essential elements are not modelled.
    // user_id, movie_id, rating_id AND utc will be a unique identifier for a rating, if we assume that
    // users can change ratings. This will also allow use to see if a user changes a rating for a movie, 
    // does the user usually change upwards or downwards? 
}

impl Rating {
    pub fn new(user_id:usize, movie_id:usize, stars:f32) -> Rating {
        Rating {
            user_id: user_id, 
            movie_id: movie_id,
            rating_id: 1, // Change this to read from a site meta, find rating count from the meta data and + 1. 
            stars: stars,
            utc: Utc::now()
        }
    }
}