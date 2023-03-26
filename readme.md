# Recommendation Systems in Rust

This repo is just for my own learning on the subject of recommendation systems and algorithms. Nothing is garuanteed to be production quality and error-tolerant and may not be the most performant. 

## Some Goals:

1. Matrix factorization technique with regularization. (Done)
2. Centered cosine similarity (Done).
3. Adding users, build this into a full simulation. (Ideation stage. Bullet points not in order.)
    1. Refactor RecommendationEngine so it can handle matrices of size: million by thousand. Refactor so that recommendation engine can create itself by querying from list of users (database of users) and list of movies (database of movies). 
    2. Create user "class".
    3. Create movie "class".
    4. Create user simulation that makes sense. 
    5. Check the result of the each recommendation technique against simulated user behavior.
    6. See if we can combine MF and Centered cosine similarity to create better recommendation.
    7. Build a frontend of a fake site. 

### References:

Perfect lecture for this: https://www.youtube.com/watch?v=ypZdwetUhCs&t=484s

Informative, but bad for actual code: https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b

My video on (1.) Matrix Factorization: https://www.youtube.com/watch?v=o26ZOtzO-SM

Recommended video on (2.) Centered cosine similarity: https://www.youtube.com/watch?v=h9gpufJFF-0&t=820s