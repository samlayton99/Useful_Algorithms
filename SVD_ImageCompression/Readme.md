# SVD Image Compression Project Description

The Singular Value Decomposition (SVD) is an incredibly useful matrix factor- ization that is widely used in both theoretical and applied mathematics. The SVD is structured in a way that makes it easy to construct low-rank approximations of matrices, and it is therefore the basis of several data compression algorithms. In this project I compute the SVD and use it to implement a simple image compression routine.

## Mathematical Background and Overview of Functions

Singular Value Decomposition (SVD) dissects an \( m \times n \) matrix \( A \) into its constituent matrices, represented by:

\[ A = U \Sigma V^H \]

In this equation:
- \( U \) stands as an \( m \times m \) orthogonal matrix.
- \( \Sigma \) is presented as an \( m \times n \) diagonal matrix. Its diagonal entities, known as singular values, are all non-negative and traditionally arranged in a descending sequence.
- \( V^H \) acts as the conjugate transpose of an \( n \times n \) orthogonal matrix \( V \).

When approximating \( A \), utilization of the top \( s \) singular values leads to a rank \( s \) representation.

1. **Matrix Decomposition**
    - `compact_svd(A, tol=1e-6)`: Processes the truncated SVD of matrix \( A \) in accordance with a specified tolerance. Eigenvalues and eigenvectors evolve from \( A^H A \). Singular values below the tolerance are removed.

2. **SVD Visualization**
    - `visualize_svd(A)`: Visually delineates the influence of SVD on the unit circle and standard basis vectors. The function sequentially integrates \( V^H \), \( \Sigma \), and \( U \) to a unit circle and basis vectors, subsequently charting the transformation.

3. **Matrix Rank Approximation**
    - `svd_approx(A, s)`: Returns the best rank \( s \) approximation of \( A \) and computes the byte requirements for storing the approximation. A ValueError emerges if \( s \) overshadows the rank of \( A \).

4. **Optimal Rank Approximation**
    - `lowest_rank_approx(A, err)`: Evaluates the optimal rank approximation of \( A \) to ensure the approximation error (considering the 2-norm) remains below a given error using the `compact_svd` and `svd_approx` utilities.

5. **Image Compression through SVD**
    - `compress_image(filename, s)`: Employs SVD to condense an image to rank \( s \). It is adept at processing both grayscale and RGB images. The function showcases the original and condensed images adjacently, highlighting the disparity in storage demands.

## Project Flow

1. **Matrix Decomposition**: The project kicks off with the `compact_svd` function delineating a truncated SVD of a matrix.

2. **Visualization**: Through the `visualize_svd` function, the transformation of the unit circle and basis vectors by SVD is vividly depicted, offering a tangible grasp of the matrix's operation.

3. **Matrix Approximation**: The `svd_approx` function is employed to approximate the matrix to a distinct rank. For those unsure about the necessary rank for a specific error, the `lowest_rank_approx` tool can determine the minimal rank meeting a particular error threshold.

4. **Image Compression**: The pinnacle application illuminated is image compression. The `compress_image` mechanism harnesses SVD to condense images, contrasting the original with the compressed visuals, emphasizing storage savings.

## Dependencies

```python
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from imageio import imread
```