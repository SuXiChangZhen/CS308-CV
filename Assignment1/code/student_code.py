import numpy as np


def my_imfilter(image, filter, pad='REFLECT'):
    """
    Apply a filter to an image. Return the filtered image.

    Args
    - image: numpy nd-array of dim (m, n, c)
    - filter: numpy nd-array of dim (k, k)
    Returns
    - filtered_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to work
     with matrices is fine and encouraged. Using opencv or similar to do the
     filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
     it may take an absurdly long time to run. You will need to get a function
     that takes a reasonable amount of time to run so that the TAs can verify
     your code works.
    - Remember these are RGB images, accounting for the final image dimension.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    ############################
    ### TODO: YOUR CODE HERE ###
    assert len(image.shape) == 3
    assert pad == 'REFLECT' or pad == 'CONSTANT'

    channel = image.shape[2]
    krnl_h, krnl_w = filter.shape[:2]  # kernel size
    img_h, img_w = image.shape[:2]  # image size
    shape_stride = (img_h, img_w, krnl_h, krnl_w)  # shape for stride
    pad_h = (krnl_h - 1) // 2
    pad_w = (krnl_w - 1) // 2
    pad_list = [(pad_h, pad_h), (pad_w, pad_w), (0, 0)]

    # pad the image
    if pad == 'REFLECT':
        padded_image = np.pad(image, pad_list, 'reflect')
    else:
        padded_image = np.pad(image, pad_list, 'constant')

    dst = []  # save the result of each channel

    if max(krnl_h, krnl_w) >= 11:
        # Use fft2 to accelerate
        fft_shape = (img_h + krnl_h - 1, img_w + krnl_w - 1)
        # pad filter
        filter_pad = np.pad(filter, ([(img_h+1)//2, img_h//2],[(img_w+1)//2, img_w//2]), 'constant')
        filter_shift = np.fft.fftshift(filter_pad)
        F = np.fft.fft2(filter_shift, fft_shape)
        for i in range(channel):
            image_c = padded_image[:, :, i].copy()  # select one channel
            fft_image_c = np.fft.fft2(image_c, fft_shape)
            result_c = np.fft.ifft2(fft_image_c * F)
            result_c = np.real(result_c)
            dst.append(result_c)
        filtered_image = np.dstack(dst)
        # cut the image to the original size
        filtered_image = filtered_image[pad_h:pad_h+img_h, pad_w:pad_w+img_w]
    else:
        strides = np.array([padded_image.shape[1], 1, padded_image.shape[1], 1]) * image.itemsize
        dst = []
        for i in range(channel):
            img_c = padded_image[:, :, i].copy()
            img_c = np.lib.stride_tricks.as_strided(img_c, shape_stride, strides)
            dst.append(np.tensordot(img_c, filter, axes=[(2, 3), (0, 1)]))
        filtered_image = np.dstack(dst)
        ### END OF STUDENT CODE ####
    ############################
    return filtered_image


def create_hybrid_image(image1, image2, filter, filter2=None):
    """
    Takes two images and creates a hybrid image. Returns the low
    frequency content of image1, the high frequency content of
    image 2, and the hybrid image.

    Args
    - image1: numpy nd-array of dim (m, n, c)
    - image2: numpy nd-array of dim (m, n, c)
    Returns
    - low_frequencies: numpy nd-array of dim (m, n, c)
    - high_frequencies: numpy nd-array of dim (m, n, c)
    - hybrid_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
      frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
      as 'clipping'.
    - If you want to use images with different dimensions, you should resize them
      in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    ############################
    ### TODO: YOUR CODE HERE ###
    if filter2 is None:
        filter2 = filter
    low_frequencies = my_imfilter(image1, filter)
    dst2 = my_imfilter(image2, filter2)
    high_frequencies = image2 - dst2
    hybrid_image = np.clip(high_frequencies + low_frequencies, 0, 1)
    high_frequencies = np.clip(high_frequencies - np.min(high_frequencies), 0, 1)
    ### END OF STUDENT CODE ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image
