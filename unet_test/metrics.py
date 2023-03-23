import sys, os, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from PIL import Image
from scipy import ndimage as ndi
from scipy.ndimage import uniform_filter
from scipy.stats import entropy
from scipy.spatial import cKDTree

from skimage._shared import utils
from skimage.util.dtype import dtype_range
from skimage._shared.utils import check_shape_equality, warn, convert_to_float
from skimage.util.arraycrop import crop

import functools
from collections.abc import Iterable


def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = np.result_type(image0.dtype, image1.dtype)
    image0 = np.asarray(image0, dtype=float_type)
    image1 = np.asarray(image1, dtype=float_type)
    return image0, image1


def mean_squared_error(image0, image1):
    """
    Compute the mean-squared error between two images.
    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.
    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.
    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.
    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    return np.mean((image0 - image1) ** 2, dtype=np.float64)


def normalized_root_mse(image_true, image_test, *, normalization='euclidean'):
    """
    Compute the normalized root mean-squared error (NRMSE) between two
    images.
    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    normalization : {'euclidean', 'min-max', 'mean'}, optional
        Controls the normalization method to use in the denominator of the
        NRMSE.  There is no standard method of normalization across the
        literature [1]_.  The methods available here are as follows:
        - 'euclidean' : normalize by the averaged Euclidean norm of
          ``im_true``::
              NRMSE = RMSE * sqrt(N) / || im_true ||
          where || . || denotes the Frobenius norm and ``N = im_true.size``.
          This result is equivalent to::
              NRMSE = || im_true - im_test || / || im_true ||.
        - 'min-max'   : normalize by the intensity range of ``im_true``.
        - 'mean'      : normalize by the mean of ``im_true``
    Returns
    -------
    nrmse : float
        The NRMSE metric.
    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_nrmse`` to
        ``skimage.metrics.normalized_root_mse``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    check_shape_equality(image_true, image_test)
    image_true, image_test = _as_floats(image_true, image_test)

    # Ensure that both 'Euclidean' and 'euclidean' match
    normalization = normalization.lower()
    if normalization == 'euclidean':
        denom = np.sqrt(np.mean((image_true * image_true), dtype=np.float64))
    elif normalization == 'min-max':
        denom = image_true.max() - image_true.min()
    elif normalization == 'mean':
        denom = image_true.mean()
    else:
        raise ValueError("Unsupported norm_type")
    return np.sqrt(mean_squared_error(image_true, image_test)) / denom


def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.
    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.
    Returns
    -------
    psnr : float
        The PSNR metric.
    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_psnr`` to
        ``skimage.metrics.peak_signal_noise_ratio``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    check_shape_equality(image_true, image_test)

    if data_range is None:
        if image_true.dtype != image_test.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "image_true.")
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = np.min(image_true), np.max(image_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "image_true has intensity values outside the range expected "
                "for its data type. Please manually specify the data_range.")
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    image_true, image_test = _as_floats(image_true, image_test)

    err = mean_squared_error(image_true, image_test)
    return 10 * np.log10((data_range ** 2) / err)


def _pad_to(arr, shape):
    """Pad an array with trailing zeros to a given target shape.
    Parameters
    ----------
    arr : ndarray
        The input array.
    shape : tuple
        The target shape.
    Returns
    -------
    padded : ndarray
        The padded array.
    Examples
    --------
    _pad_to(np.ones((1, 1), dtype=int), (1, 3))
    array([[1, 0, 0]])
    """
    if not all(s >= i for s, i in zip(shape, arr.shape)):
        raise ValueError(f'Target shape {shape} cannot be smaller than input'
                         f'shape {arr.shape} along any axis.')
    padding = [(0, s-i) for s, i in zip(shape, arr.shape)]
    return np.pad(arr, pad_width=padding, mode='constant', constant_values=0)


def normalized_mutual_information(image0, image1, *, bins=100):
    r"""Compute the normalized mutual information (NMI).
    The normalized mutual information of :math:`A` and :math:`B` is given by::
    ..math::
        Y(A, B) = \frac{H(A) + H(B)}{H(A, B)}
    where :math:`H(X) := - \sum_{x \in X}{x \log x}` is the entropy.
    It was proposed to be useful in registering images by Colin Studholme and
    colleagues [1]_. It ranges from 1 (perfectly uncorrelated image values)
    to 2 (perfectly correlated image values, whether positively or negatively).
    Parameters
    ----------
    image0, image1 : ndarray
        Images to be compared. The two input images must have the same number
        of dimensions.
    bins : int or sequence of int, optional
        The number of bins along each axis of the joint histogram.
    Returns
    -------
    nmi : float
        The normalized mutual information between the two arrays, computed at
        the granularity given by ``bins``. Higher NMI implies more similar
        input images.
    Raises
    ------
    ValueError
        If the images don't have the same number of dimensions.
    Notes
    -----
    If the two input images are not the same shape, the smaller image is padded
    with zeros.
    References
    ----------
    .. [1] C. Studholme, D.L.G. Hill, & D.J. Hawkes (1999). An overlap
           invariant entropy measure of 3D medical image alignment.
           Pattern Recognition 32(1):71-86
           :DOI:`10.1016/S0031-3203(98)00091-0`
    """
    if image0.ndim != image1.ndim:
        raise ValueError(f'NMI requires images of same number of dimensions. '
                         f'Got {image0.ndim}D for `image0` and '
                         f'{image1.ndim}D for `image1`.')
    if image0.shape != image1.shape:
        max_shape = np.maximum(image0.shape, image1.shape)
        padded0 = _pad_to(image0, max_shape)
        padded1 = _pad_to(image1, max_shape)
    else:
        padded0, padded1 = image0, image1

    hist, bin_edges = np.histogramdd(
            [np.reshape(padded0, -1), np.reshape(padded1, -1)],
            bins=bins,
            density=True,
            )

    H0 = entropy(np.sum(hist, axis=0))
    H1 = entropy(np.sum(hist, axis=1))
    H01 = entropy(np.reshape(hist, -1))

    return (H0 + H1) / H01


# @deprecate_multichannel_kwarg(multichannel_position=5)
def gaussian(image, sigma=1, output=None, mode='nearest', cval=0,
             multichannel=None, preserve_range=False, truncate=4.0, *,
             channel_axis=None):
    """Multi-dimensional Gaussian filter.
    Parameters
    ----------
    image : array-like
        Input image (grayscale or color) to filter.
    sigma : scalar or sequence of scalars, optional
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    output : array, optional
        The ``output`` parameter passes an array in which to store the
        filter output.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'nearest'.
    cval : scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default
        is 0.0
    multichannel : bool, optional (default: None)
        Whether the last axis of the image is to be interpreted as multiple
        channels. If True, each channel is filtered separately (channels are
        not mixed together). Only 3 channels are supported. If ``None``,
        the function will attempt to guess this, and raise a warning if
        ambiguous, when the array has shape (M, N, 3).
        This argument is deprecated: specify `channel_axis` instead.
    preserve_range : bool, optional
        If True, keep the original range of values. Otherwise, the input
        ``image`` is converted according to the conventions of ``img_as_float``
        (Normalized first to values [-1.0 ; 1.0] or [0 ; 1.0] depending on
        dtype of input)
        For more information, see:
        https://scikit-image.org/docs/dev/user_guide/data_types.html
    truncate : float, optional
        Truncate the filter at this many standard deviations.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    Returns
    -------
    filtered_image : ndarray
        the filtered array
    Notes
    -----
    This function is a wrapper around :func:`scipy.ndi.gaussian_filter`.
    Integer arrays are converted to float.
    The ``output`` should be floating point data type since gaussian converts
    to float provided ``image``. If ``output`` is not provided, another array
    will be allocated and returned as the result.
    The multi-dimensional filter is implemented as a sequence of
    one-dimensional convolution filters. The intermediate arrays are
    stored in the same data type as the output. Therefore, for output
    types with a limited precision, the results may be imprecise
    because intermediate results may be stored with insufficient
    precision.
    Examples
    --------
    a = np.zeros((3, 3))
    a[1, 1] = 1
    a
    array([[0., 0., 0.],
           [0., 1., 0.],
           [0., 0., 0.]])
    gaussian(a, sigma=0.4)  # mild smoothing
    array([[0.00163116, 0.03712502, 0.00163116],
           [0.03712502, 0.84496158, 0.03712502],
           [0.00163116, 0.03712502, 0.00163116]])
    gaussian(a, sigma=1)  # more smoothing
    array([[0.05855018, 0.09653293, 0.05855018],
           [0.09653293, 0.15915589, 0.09653293],
           [0.05855018, 0.09653293, 0.05855018]])
    # Several modes are possible for handling boundaries
    gaussian(a, sigma=1, mode='reflect')
    array([[0.08767308, 0.12075024, 0.08767308],
           [0.12075024, 0.16630671, 0.12075024],
           [0.08767308, 0.12075024, 0.08767308]])
    # For RGB images, each is filtered separately
    from skimage.data import astronaut
    image = astronaut()
    filtered_img = gaussian(image, sigma=1, channel_axis=-1)
    """
    if image.ndim == 3 and image.shape[-1] == 3 and channel_axis is None:
        msg = ("Images with dimensions (M, N, 3) are interpreted as 2D+RGB "
               "by default. Use `multichannel=False` to interpret as "
               "3D image with last dimension of length 3.")
        warn(RuntimeWarning(msg))
        channel_axis = -1
    if np.any(np.asarray(sigma) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")
    if channel_axis is not None:
        # do not filter across channels
        if not isinstance(sigma, Iterable):
            sigma = [sigma] * (image.ndim - 1)
        if len(sigma) == image.ndim - 1:
            sigma = list(sigma)
            sigma.insert(channel_axis % image.ndim, 0)
    image = convert_to_float(image, preserve_range)
    float_dtype = image.dtype
    image = image.astype(float_dtype, copy=False)
    if (output is not None) and (not np.issubdtype(output.dtype, np.floating)):
        raise ValueError("Provided output data type is not float")
    return ndi.gaussian_filter(image, sigma, output=output,
                               mode=mode, cval=cval, truncate=truncate)


# skimage.metrics.structural_similarity
# @utils.deprecate_multichannel_kwarg()
def structural_similarity(im1, im2,
                          *,
                          win_size=None, gradient=False, data_range=None,
                          channel_axis=None, multichannel=False,
                          gaussian_weights=False, full=False, **kwargs):
    """
    Compute the mean structural similarity index between two images.
    Parameters
    ----------
    im1, im2 : ndarray
        Images. Any dimensionality with same shape.
    win_size : int or None, optional
        The side-length of the sliding window used in comparison. Must be an
        odd value. If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    gradient : bool, optional
        If True, also return the gradient with respect to im2.
    data_range : float, optional
        The data range of the input image (distance between minimum and
        maximum possible values). By default, this is estimated from the image
        data-type.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    multichannel : bool, optional
        If True, treat the last dimension of the array as channels. Similarity
        calculations are done independently for each channel then averaged.
        This argument is deprecated: specify `channel_axis` instead.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    full : bool, optional
        If True, also return the full structural similarity image.
    Other Parameters
    ----------------
    use_sample_covariance : bool
        If True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        Algorithm parameter, K1 (small constant, see [1]_).
    K2 : float
        Algorithm parameter, K2 (small constant, see [1]_).
    sigma : float
        Standard deviation for the Gaussian when `gaussian_weights` is True.
    Returns
    -------
    mssim : float
        The mean structural similarity index over the image.
    grad : ndarray
        The gradient of the structural similarity between im1 and im2 [2]_.
        This is only returned if `gradient` is set to True.
    S : ndarray
        The full SSIM image.  This is only returned if `full` is set to True.
    Notes
    -----
    To match the implementation of Wang et. al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, and `use_sample_covariance` to False.
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_ssim`` to
        ``skimage.metrics.structural_similarity``.
    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`
    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       :arxiv:`0901.0065`
       :DOI:`10.1007/s10043-009-0119-z`
    """
    check_shape_equality(im1, im2)
    float_type = np.result_type(im1.dtype, im2.dtype)

    if channel_axis is not None:
        # loop over channels
        args = dict(win_size=win_size,
                    gradient=gradient,
                    data_range=data_range,
                    channel_axis=None,
                    gaussian_weights=gaussian_weights,
                    full=full)
        args.update(kwargs)
        nch = im1.shape[channel_axis]
        mssim = np.empty(nch, dtype=float_type)

        if gradient:
            G = np.empty(im1.shape, dtype=float_type)
        if full:
            S = np.empty(im1.shape, dtype=float_type)
        channel_axis = channel_axis % im1.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        for ch in range(nch):
            ch_result = structural_similarity(im1[_at(ch)],
                                              im2[_at(ch)], **args)
            if gradient and full:
                mssim[ch], G[_at(ch)], S[_at(ch)] = ch_result
            elif gradient:
                mssim[ch], G[_at(ch)] = ch_result
            elif full:
                mssim[ch], S[_at(ch)] = ch_result
            else:
                mssim[ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if gaussian_weights:
        # Set to give an 11-tap filter with the default sigma of 1.5 to match
        # Wang et. al. 2004.
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
        else:
            win_size = 7   # backwards compatibility

    if np.any((np.asarray(im1.shape) - win_size) < 0):
        raise ValueError(
            'win_size exceeds image extent. '
            'Either ensure that your images are '
            'at least 7x7; or pass win_size explicitly '
            'in the function call, with an odd value '
            'less than or equal to the smaller side of your '
            'images. If your images are multichannel '
            '(with color channels), set channel_axis to '
            'the axis number corresponding to the channels.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        if im1.dtype != im2.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "im1.dtype.", stacklevel=2)
        dmin, dmax = dtype_range[im1.dtype.type]
        data_range = dmax - dmin

    ndim = im1.ndim

    if gaussian_weights:
        filter_func = gaussian
        filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    im1 = im1.astype(float_type, copy=False)
    im2 = im2.astype(float_type, copy=False)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim. Use float64 for accuracy.
    mssim = crop(S, pad).mean(dtype=np.float64)

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * im1
        grad += filter_func(-S / B2, **filter_args) * im2
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D,
                            **filter_args)
        grad *= (2 / im1.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim


# calculate inception score in numpy
# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
    p_yx = p_yx - np.min(p_yx)
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    is_score = np.mean(sum_kl_d)
    # undo the logs
    #is_score = exp(avg_kl_d)
    return is_score


def calc_metrics(im1, im2, roi=None):
  """
  im1: output image
  im2: ground truth
  roi: (upper bound, lower bound, left bound, right bound) or whole image if None
  """
  im_res, im_gt = im1, im2
  if roi:
    top, bottom, left, right = roi
    im_res, im_gt = im1[top:bottom, left:right], im2[top:bottom, left:right]
  mse = mean_squared_error(im_gt, im_res)
  rmse = normalized_root_mse(im_gt, im_res)
  psnr = peak_signal_noise_ratio(im_gt, im_res)
  nmi = normalized_mutual_information(im_gt, im_res)
  ssim = structural_similarity(im_gt, im_res)
  # iscore = np.log(inception_score(np.array([im_gt]*5+[im_res]*5))[0])
  iscore = np.log(calculate_inception_score(np.array([im_gt, im_res])))
  print("mse: ", mse, " rmse: ", rmse, " psnr: ", psnr, " nmi: ", nmi, " ssim: ", ssim, " iscore: ", iscore)
