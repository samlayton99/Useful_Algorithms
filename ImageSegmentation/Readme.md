# Image Segmentation Using Graph Theory

Graph theory has a variety of applications. A graph (or network) can be repre- sented in many ways on a computer. In this project I explore common matrix representation for graphs and show how certain properties of the matrix representation correspond to inherent properties of the original graph. We also introduce tools for working with images in Python, and conclude with an application of using graphs and linear algebra to segment images.

## Mathematical Concepts:

1. **Laplacian Matrix (L)**: This is the difference between the degree matrix (D) and the adjacency matrix (A). It's used in graph theory to determine properties of a graph. Mathematically, \( L = D - A \).
2. **Adjacency Matrix (A)**: Represents a graph, indicating which vertices are adjacent to which others.
3. **Eigenvalues & Eigenvectors**: Used here to extract information about the segmentation boundaries. The second smallest eigenvalue (also known as Fiedler value) and its associated eigenvector play a pivotal role.

## Functions:

1. `laplacian(A)`: Computes the Laplacian matrix for a graph with adjacency matrix A.
2. `connectivity(A, tol=1e-8)`: Determines the number of connected components in the graph and its algebraic connectivity.
3. `get_neighbors(index, radius, height, width)`: Helper function to find neighboring pixels within a specified radius from a central pixel in an image.

## Classes:

Stores and segments images using the brightness values of its pixels and methods built on graph theory.

- **Methods**:
  1. `__init__(filename)`: Initializes the image segmenter, reading the image and storing its brightness values.
  2. `show_original()`: Displays the original image.
  3. `adjacency(r=5., sigma_B2=.02, sigma_X2=3.)`: Computes the Adjacency and Degree matrices for the image graph based on given parameters.
  4. `cut(A, D)`: Computes a boolean mask that segments the image using the Laplacian matrix.
  5. `segment(r=5., sigma_B=.02, sigma_X=3.)`: Displays the original image alongside its positive and negative segments.

## Project Flow:

1. **Initialization**: The ImageSegmenter class reads an image and stores its brightness values in a flattened array.
2. **Visualization**: The original image can be displayed using the `show_original` method.
3. **Graph Representation**: Using the `adjacency` method, the adjacency and degree matrices for the image graph are computed.
4. **Image Segmentation**: 
   - First, the Laplacian matrix is computed.
   - Eigenvalues and eigenvectors of the normalized Laplacian are computed to identify boundaries.
   - The second smallest eigenvector is reshaped to match the image and then used to generate a boolean mask.
   - This mask segments the image into positive and negative sections.
5. **Display**: Using the `segment` method, the original image and its segments (positive and negative) are displayed side by side.

## Useful Applications:

- **Medical Imaging**: Enables differentiation between tissues or regions in medical scans.
- **Object Detection**: Helps identify distinct objects or features within an image.
- **Image Editing**: Can be used for selective editing or to apply different filters to segmented portions of an image.
- **Computer Vision**: Enhances machine learning models by providing segmented data which may improve accuracy in tasks like object recognition.

## Dependencies
```python
import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg
from imageio import imread
from matplotlib import pyplot as plt
```

To use this project, uncomment the bottom lines and provide your images for segmentation.