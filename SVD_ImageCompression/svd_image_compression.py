"""Volume 1: The SVD and Image Compression."""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from imageio import imread

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    # Calculate AhA and get its eigenvalues and eigenvectors.
    matrix = A.conj().T @ A
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Calculate the singular values and make a sorting dictionary.
    sigma = abs(eigenvalues)**(.5)
    sortingdictionary = {}

    # Make a dictionary of the singular values and their corresponding eigenvectors.
    for i in range(len(sigma)):
        sortingdictionary[sigma[i]] = eigenvectors[:,i]
    
    # Sort the singular values and eigenvectors in descending order.
    sigma = np.sort(sigma)[::-1]
    V = np.array([sortingdictionary[sigma[i]] for i in range(len(sigma))]).T

    # Count the number of singular values that are greater than the tolerance.
    i = 0
    while i < len(sigma) and sigma[i] > tol:
        i += 1
    
    # Truncate the singular values and eigenvectors, and calculate U.
    sigma1 = sigma[:i]
    V1 = V[:,:i]
    U1 = np.array(A @ V1 / sigma1)

    # return U1, sigma1, V1h
    return U1, sigma1, V1.conj().T
    

# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    # Generate a 2 by 200 matrix of points on the unit circle, as well as the two standard basis vectors.
    theta = np.linspace(0, 2*np.pi, 200)
    S = np.array([np.cos(theta), np.sin(theta)])
    E = np.array([[1,0,0],[0,0,1]])

    # Calculate the SVD of A.
    U, sigma, Vh = la.svd(A)

    # Calculate the different stages of the transformation for S
    S1 = Vh @ S
    S2 = np.diag(sigma) @ S1
    S3 = U @ S2

    # Calculate the different stages of the transformation for E
    E1 = Vh @ E
    E2 = np.diag(sigma) @ E1
    E3 = U @ E2

    # Plot the unit circle and the standard basis vectors.
    plt.subplot(2,2,1)
    plt.title('S - Original')
    plt.axis('equal')

    # Plot the actual data points
    plt.plot(S[0], S[1], 'b', label='Unit Circle')
    plt.plot(E[0], E[1], 'r',label='Standard Basis')
    
    # Plot the fist stage of the transformation
    plt.subplot(2,2,2)
    plt.title('S - After Vh')
    plt.axis('equal')

    # Plot the actual data points
    plt.plot(S1[0], S1[1], 'b', label='Unit Circle')
    plt.plot(E1[0], E1[1], 'r',label='Standard Basis')

    # Plot the second stage of the transformation
    plt.subplot(2,2,3)
    plt.title('S - After Vh and Sigma')
    plt.axis('equal')
    
    # Plot the actual data points
    plt.plot(S2[0], S2[1], 'b', label='Unit Circle')
    plt.plot(E2[0], E2[1], 'r',label='Standard Basis')

    # Plot the third stage of the transformation
    plt.subplot(2,2,4)
    plt.title('S - After Vh, Sigma, and U')
    plt.axis('equal')

    # Plot the actual data points
    plt.plot(S3[0], S3[1], 'b', label='Unit Circle')
    plt.plot(E3[0], E3[1], 'r',label='Standard Basis')

    # Show the plot neatly
    plt.suptitle('Linear Transformation of A on the Unit Circle')
    plt.tight_layout()
    plt.show()


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    # Calculate the SVD of A.
    U, sigma, Vh = la.svd(A)

    # Raise a value error if s larger than the number of non zero singular values of A.
    if s > np.count_nonzero(sigma):
        raise ValueError("s must be less than the Rank of A")

    # Calculate the restricted SVD of A to rank s.
    Us = U[:,:s]
    sigmas = sigma[:s]
    Vhs = Vh[:s,:]

    # Calculate the best rank s approximation to A.
    As = Us @ np.diag(sigmas) @ Vhs
    
    # Calculate the number of bytes needed to store the truncated SVD.
    entries = Us.size + sigmas.size + Vhs.size

    # Return the best rank s approximation and the number of bytes needed to store the truncated SVD.
    return As, entries


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    # Calculate the SVD of A.
    U, sigma, Vh = compact_svd(A)
    sizes = range(1,np.linalg.matrix_rank(A) + 1)
    
    # Calculate the minimal rank s needed to satisfy the error bound.
    sigma = sigma - err
    s = len(sigma[sigma > 0])
    print(s)
    print(sigma + err)

    # Raise an error if it is impossible to satisfy the error bound.
    if s == len(sigma):
        raise ValueError("The error bound is too small")

    # Calculate the restricted SVD of A to rank s.
    As = svd_approx(A, s)[0]

    # Calculate the number of entries needed to store the truncated SVD.
    m, n = A.shape
    entries = m*s + s + s*n
    
    # Return the lowest rank approx and number of entries
    return As, entries


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    # Load the image and convert it to a matrix.
    image = np.array(plt.imread(filename)) / 255
    size = image.shape
    
    # If the image is RGB, split its components and compress each component.
    if len(size) == 3:
        # Split the image into its components and set cmap to None.
        imageR = image[:,:,0]
        imageG = image[:,:,1]
        imageB = image[:,:,2]
        map = None
        
        # Compress each component and get the number of entries
        imageRs, entries = svd_approx(imageR, s)
        imageGs = svd_approx(imageG, s)[0]
        imageBs = svd_approx(imageB, s)[0]

        # Join the components back together.
        images = np.dstack((imageRs, imageGs, imageBs))

    # Compress the image and get the number of entries if it is grayscale.
    else:
        images, entries = svd_approx(image, s)
        map = 'gray'

    # Plot the original image and the compressed image.
    plt.subplot(1,2,1)
    plt.title('Original Image:\n' + str(image.size) + ' Entries')
    plt.imshow(image, cmap = map)
    plt.axis('off')
    
    # Plot the compressed image.
    plt.subplot(1,2,2)
    plt.title('Compressed Image:\n' + str(entries) + ' Entries')
    plt.imshow(np.clip(images,0,1), cmap = map)
    plt.axis('off')

    # Show the plot and give it a title.
    plt.suptitle('Difference in Entries: ' + str(image.size - entries))
    plt.tight_layout()
    plt.show()