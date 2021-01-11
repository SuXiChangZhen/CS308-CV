import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################

    # Obtaining Gaussian convolution kernel
    kernel = cv2.getGaussianKernel(7, 3)
    kernel = kernel * kernel.T

    # Calculate image gradient
    img_gradient_x = cv2.Sobel(image, -1, 1, 0, ksize=3)
    img_gradient_y = cv2.Sobel(image, -1, 0, 1, ksize=3)

    gradient_x_square = img_gradient_x * img_gradient_x
    gradient_y_square = img_gradient_y * img_gradient_y
    gradient_xy_square = img_gradient_x * img_gradient_y

    # Gauss blur
    gradient_x_square = cv2.filter2D(gradient_x_square, -1, kernel)
    gradient_y_square = cv2.filter2D(gradient_y_square, -1, kernel)
    gradient_xy_square = cv2.filter2D(gradient_xy_square, -1, kernel)

    # Calculate cornerness

    det = gradient_x_square * gradient_y_square - gradient_xy_square * gradient_xy_square
    trace = gradient_x_square + gradient_y_square

    # 0.04 ~ 0.06
    cornerness = det - 0.06 * (trace * trace)

    # threshold
    thr = 0.01 * np.max(cornerness)

    cornerness[cornerness < thr] = 0

    # shape[0] how many row
    img_h, img_w = image.shape[:2]
    size = 4

    # for x in range(feature_width, img_h - feature_width - size, size):
    #     for y in range(feature_width, img_w - feature_width - size, size):
    #         cornerness[x+2][y+2] += 2000

    local_max_idx_value = []
    for x in range(feature_width, img_h - feature_width, size):
        row_sub = cornerness[x:(x + size)]
        for y in range(feature_width, img_w - feature_width, size):
            sub_mat = row_sub[:, y: (y + size)]
            max_value = np.max(sub_mat)
            if max_value > thr:
                index = np.unravel_index(np.argmax(sub_mat), (size, size)) + np.array((x, y))
                local_max_idx_value.append((tuple(index), max_value))

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################

    local_max_idx_value = sorted(local_max_idx_value, key=lambda x: x[1], reverse=True)
    local_max_idx = [x[0] for x in local_max_idx_value]
    R_idx_value = [(local_max_idx[0], float('inf'))]
    for i in range(1, len(local_max_idx)):
        dist = []
        i0 = local_max_idx[i]
        for j in range(i):
            i1 = local_max_idx[j]
            dist.append((i0[0] - i1[0]) * (i0[0] - i1[0]) + (i0[1] - i1[1]) * (i0[1] - i1[1]))
        R_idx_value.append((i0, min(dist)))

    R_idx_value = sorted(R_idx_value, key=lambda x: x[1], reverse=True)
    R_idx = [x[0] for x in R_idx_value]
    x = [x[1] for x in R_idx]
    y = [x[0] for x in R_idx]
    x = np.asarray(x[:1500])
    y = np.asarray(y[:1500])

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x, y, confidences, scales, orientations
