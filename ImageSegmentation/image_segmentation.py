# image_segmentation.py
"""Volume 1: Image Segmentation.
<Sam Layton>
<Section 003>
<11/1/22>
"""

import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg
from imageio import imread
from matplotlib import pyplot as plt

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    # Make sure A is a numpy array and sum the columns.
    A = np.array(A)
    x = np.sum(A, axis=1)

    # Create the diagonal matrix and subtract A from it.
    return np.diag(x) - A


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    # Calculate the Laplacian matrix and its eigen values.
    L = laplacian(A)
    w, v = la.eig(L)

    # Sort the eigenvalues
    w = np.sort(w)

    # Initialize the counter and loop through the eigenvalues.
    j = 0
    for i in range(len(w)):
        # If the eigenvalue is less than the tolerance, increment the counter.
        if w[i] < tol:
            j += 1

    # return the number of connected components and the algebraic connectivity.
    return j, w[1].real

# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        # Read the image and store it as an attribute
        self.image = imread(filename) / 255

        # Flatten the brightness matrix and store it as an attribute
        if len(self.image.shape) == 3:
            average_color = self.image.mean(axis=2)
            self.brightness = np.ravel(average_color)
            self.type = 'c'
        
        # Do the same for grayscale images
        else:
            self.brightness = np.ravel(self.image)
            self.type = 'g'
        
        # Store the height and width of the image as attributes.
        self.height = len(self.image)
        self.width = len(self.image[0])

    # Problem 3
    def show_original(self):
        """Display the original image."""
        # Display the image if it is grayscale
        if len(self.image.shape) == 2:
            plt.imshow(self.image, cmap='gray')

        # Display the image if it is RGB
        else:
            plt.imshow(self.image)
        plt.axis('off')

        # Show it neatly
        plt.tight_layout()
        plt.show()
        

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        # Get the number of pixels in the image, and the height and width.
        mn = len(self.brightness)
        m = len(self.image)
        n = len(self.image[0])

        # Initialize the adjacency matrix and the degree matrix.
        A = sparse.lil_matrix((mn,mn))
        D = np.zeros(mn)
        
        # Loop through the pixels in the image.
        for i in range(mn):
            # Get the neighbors of the current pixel.
            neighbors, distances = get_neighbors(i, r, m, n)
            
            # Calculate the weights of the edges.
            weights = np.exp((-abs(self.brightness[i] - self.brightness[neighbors]) / sigma_B2) - distances / sigma_X2)

            # Add the weights to the adjacency matrix.
            A[i, neighbors] = weights

            D[i] = np.sum(weights)

        # convert A to a csr matrix and return it.
        A = A.tocsc()
        return A, D


    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        # Compute the Laplacian matrix.
        L = csgraph.laplacian(A)
        
        # Form the diagonal matrix from degree matrix.
        D = D ** (-1/2)
        D = sparse.diags(D)

        # Compute the normalized Laplacian matrix.
        L = D @ L @ D

        # Compute the eigenvalues and eigenvectors of the normalized Laplacian.
        w, v = linalg.eigsh(L, k=2, which='SM')
        eigen = v[:,1]

        # Reshape the eigenvector to the original image shape and make a boolean mask.
        eigen = eigen.reshape(self.height, self.width)
        mask = eigen > 0

        # Return the boolean mask.
        return mask


    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        # Get the mask for segmentation.
        mask = self.cut(*self.adjacency(r, sigma_B, sigma_X))
        
        # specify color display type
        if self.type == 'g':
            map = 'gray'

        # if the image is RGB, set map to None and make the mask 3D.
        else:
            map = None
            mask = np.stack((mask, mask, mask), axis=2)

        # Get the postive and negative segments
        positive = mask
        negative = ~positive

        # Plot the original image
        plt.subplot(1,3,1)
        plt.title('Original')
        plt.imshow(self.image, cmap = map)
        plt.axis('off')
        
        # Plot the positive segment
        plt.subplot(1,3,2)
        plt.title('Positive')
        plt.imshow(self.image * positive, cmap = map)
        plt.axis('off')

        # Plot the negative segment
        plt.subplot(1,3,3)
        plt.title('Negative')
        plt.imshow(self.image * negative, cmap = map)
        plt.axis('off')

        # Show it neatly
        plt.tight_layout()
        plt.suptitle("Image Segmentation")
        plt.show()


# if __name__ == '__main__':
#     ImageSegmenter("input_files/dream_gray.png").segment()
#     ImageSegmenter("input_files/dream.png").segment()
#     ImageSegmenter("input_files/monument_gray.png").segment()
#     ImageSegmenter("input_files/monument.png").segment()