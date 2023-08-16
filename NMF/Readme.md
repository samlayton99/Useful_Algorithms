# Non-Negative Matrix Factorization Project Description

Non-negative Matrix Factorization (NMF) is a potent algorithm that factorizes a non-negative matrix into two other non-negative matrices. Widely employed in image processing, text mining, and bioinformatics, this project dives deep into the understanding and application of NMF, with a focus on both theory and practical implementations.

## Mathematical Background and Overview of Functions

NMF aims to decompose a non-negative matrix `V` of dimensions `m x n` into two matrices `W` of size `m x r` and `H` of size `r x n`. The primary objective during this factorization is to minimize the Frobenius norm of the difference between `V` and the product of `W` and `H`. Numerous algorithms exist to achieve this, such as multiplicative update rules, alternating least squares, and gradient descent.

1. **Matrix Decomposition Using Sklearn**
    - `sklearn.decomposition.NMF`: A handy class for performing NMF. Given the desired number of components, it returns matrices `W` and `H` and provides methods for further matrix transformation based on the learned factorization.

2. **Optimization with CVXPY**
    - `cvxpy`: A Python utility for convex optimization, which can effectively tackle the NMF optimization challenge using sophisticated convex optimization techniques.

3. **Reconstruction Error Calculation**
    - `mean_squared_error`: Used to ascertain the error between two matrices, typically between the original and its reconstruction.

4. **NMF-based Recommendation System**
    - `NMFRecommender`: A versatile class that harnesses the power of NMF to provide recommendations. With parameters like random state, rank, maximum iterations, and tolerance, it employs CVXPY for optimization. The `fit()` method updates the weight matrices `W` and `H` until a set tolerance or maximum iterations is reached. The `reconstruct()` method allows the reconstruction of the matrix `V` using the factorized matrices for comparison purposes.

5. **Applications in Image Processing**
    - `getfaces()`: A function that procures a dataset of faces and applies NMF using the `sklearn.decomposition.NMF` class. It returns the matrices `W` and `H`.
    - `prob4()`: A reconstruction-centric function, it uses matrices `W` and `H` to regenerate a face, employing the formula `face = W.dot(H)`.
    - `prob5()`: Analogous to `prob4()`, this function recreates an image from its decomposed matrices, with the resulting image acquired through `image = W.dot(H)`.

## Project Flow

1. **Data Loading**: Using the `imageio` library, an image is sourced for the project.
2. **Data Preprocessing**: Conversion of the image to grayscale and reshaping it into a vector form.
3. **Implementing NMF**: The `sklearn.decomposition.NMF` class is put to use to carry out NMF on the vectorized image.
4. **Image Reconstruction**: Using the matrices `W` and `H`, the original image is reconstructed.
5. **Evaluation**: The deviation between the original and the reconstructed images is quantified using the mean squared error.

## Dependencies

- numpy (`import numpy as np`)
- cvxpy (`import cvxpy as cp`)
- matplotlib (`from matplotlib import pyplot as plt`)
- os (`import os`)
- imageio (`from imageio import imread`)
- warnings (`import warnings`)
- sklearn (`from sklearn.decomposition import NMF`, `from sklearn.metrics import mean_squared_error as mse`)
