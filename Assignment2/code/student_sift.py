import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################

    fw = feature_width // 2
    size = feature_width // 4
    index = list(zip(y.astype(int), x.astype(int)))
    result = []
    for idx in index:
        tmp = np.array([])
        for x in range(idx[0] - fw, idx[0] + fw, size):
            row_sub = image[x:(x + size)]
            for y in range(idx[1] - fw, idx[1] + fw, size):

                sub_mat = row_sub[:, y:(y + size)]

                img_gradient_x = cv2.Sobel(sub_mat, -1, 1, 0, ksize=3)
                img_gradient_y = cv2.Sobel(sub_mat, -1, 0, 1, ksize=3)

                g = np.hypot(img_gradient_x, img_gradient_y)
                a = np.arctan2(img_gradient_y, img_gradient_x)
                # Statistical histogram of eight directions
                histogram = np.histogram(a, bins=8, range=(-np.pi, np.pi), weights=g)[0]
                tmp = np.append(tmp, histogram)

        tmp = tmp / np.sum(tmp)
        tmp = np.clip(tmp, 0, 0.2)
        tmp = tmp / np.sum(tmp)
        tmp = tmp.reshape((1, tmp.shape[0]))
        result.append(tmp)
    fv = np.array(result)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv
