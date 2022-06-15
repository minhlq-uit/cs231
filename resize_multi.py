import resize_multi as seam_carving
from PIL import Image
from pathlib import Path
import time
import argparse
from typing import Tuple, Optional

import numpy as np
from scipy.ndimage import sobel

WIDTH_FIRST = 'width-first'
HEIGHT_FIRST = 'height-first'
OPTIMAL = 'optimal'
VALID_ORDERS = (WIDTH_FIRST, HEIGHT_FIRST, OPTIMAL)

FORWARD_ENERGY = 'forward'
BACKWARD_ENERGY = 'backward'
VALID_ENERGY_MODES = (FORWARD_ENERGY, BACKWARD_ENERGY)

DROP_MASK_ENERGY = 1e5
KEEP_MASK_ENERGY = 1e3

global keep_mask_
keep_mask_ = None


def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to a grayscale image"""
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
    return (rgb @ coeffs).astype(rgb.dtype)


def _get_seam_mask(src: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Convert a list of seam column indices to a mask"""
    return ~np.eye(src.shape[1], dtype=np.bool)[seam]


def _remove_seam_mask(src: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    """Remove a seam from the source image according to the given seam_mask"""
    if src.ndim == 3:
        h, w, c = src.shape
        seam_mask = np.dstack([seam_mask] * c)
        dst = src[seam_mask].reshape((h, w - 1, c))
    else:
        h, w = src.shape
        dst = src[seam_mask].reshape((h, w - 1))
    return dst


def _get_energy(gray: np.ndarray) -> np.ndarray:
    """Get backward energy map from the source image"""
    assert gray.ndim == 2

    gray = gray.astype(np.float32)
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy


def _get_backward_seam(energy: np.ndarray) -> np.ndarray:
    """Compute the minimum vertical seam from the backward energy map"""
    assert energy.size > 0 and energy.ndim == 2
    h, w = energy.shape
    cost = energy[0]
    parent = np.empty((h, w), dtype=np.int32)
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    for r in range(1, h):
        left_shift = np.hstack((cost[1:], np.inf))
        right_shift = np.hstack((np.inf, cost[:-1]))
        min_idx = np.argmin([right_shift, cost, left_shift],
                            axis=0) + base_idx
        parent[r] = min_idx
        cost = cost[min_idx] + energy[r]

    c = np.argmin(cost)
    seam_cost = np.min(cost)
    seam = np.empty(h, dtype=np.int32)

    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam, seam_cost


def _get_backward_seams(gray: np.ndarray, num_seams: int,
                        keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """Compute the minimum N vertical seams using backward energy"""
    h, w = gray.shape
    seams_mask = np.zeros((h, w), dtype=np.bool)
    rows = np.arange(0, h, dtype=np.int32)
    idx_map = np.tile(np.arange(0, w, dtype=np.int32), h).reshape((h, w))
    energy = _get_energy(gray)
    for _ in range(num_seams):
        if keep_mask is not None:
            energy[keep_mask] += KEEP_MASK_ENERGY
        seam, seam_cost = _get_backward_seam(energy)
        seams_mask[rows, idx_map[rows, seam]] = True

        seam_mask = _get_seam_mask(gray, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if keep_mask is not None:
            keep_mask = _remove_seam_mask(keep_mask, seam_mask)

        # Only need to re-compute the energy in the bounding box of the seam
        _, cur_w = energy.shape
        lo = max(0, np.min(seam) - 1)
        hi = min(cur_w, np.max(seam) + 1)
        pad_lo = 1 if lo > 0 else 0
        pad_hi = 1 if hi < cur_w - 1 else 0
        mid_block = gray[:, lo - pad_lo:hi + pad_hi]
        _, mid_w = mid_block.shape
        mid_energy = _get_energy(mid_block)[:, pad_lo:mid_w - pad_hi]
        energy = np.hstack((energy[:, :lo], mid_energy, energy[:, hi + 1:]))

    return seams_mask


def _get_forward_seam(gray: np.ndarray,
                      keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """Compute the minimum vertical seam using forward energy"""
    assert gray.size > 0 and gray.ndim == 2
    gray = gray.astype(np.float32)
    h, w = gray.shape

    top_row = gray[0]
    top_row_lshift = np.hstack((top_row[1:], top_row[-1]))
    top_row_rshift = np.hstack((top_row[0], top_row[:-1]))
    dp = np.abs(top_row_lshift - top_row_rshift)

    parent = np.zeros(gray.shape, dtype=np.int32)
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    for r in range(1, h):
        curr_row = gray[r]
        curr_lshift = np.hstack((curr_row[1:], curr_row[-1]))
        curr_rshift = np.hstack((curr_row[0], curr_row[:-1]))
        cost_top = np.abs(curr_lshift - curr_rshift)
        if keep_mask is not None:
            cost_top[keep_mask[r]] += KEEP_MASK_ENERGY

        prev_row = gray[r - 1]
        cost_left = cost_top + np.abs(prev_row - curr_rshift)
        cost_right = cost_top + np.abs(prev_row - curr_lshift)

        dp_left = np.hstack((np.inf, dp[:-1]))
        dp_right = np.hstack((dp[1:], np.inf))

        choices = np.vstack([cost_left + dp_left, cost_top + dp,
                             cost_right + dp_right])
        dp = np.min(choices, axis=0)
        parent[r] = np.argmin(choices, axis=0) + base_idx

    c = np.argmin(dp)
    seam_cost = np.min(dp)

    seam = np.empty(h, dtype=np.int32)
    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam, seam_cost


def _get_forward_seams(gray: np.ndarray, num_seams: int,
                       keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """Compute minimum N vertical seams using forward energy"""
    h, w = gray.shape
    seams_mask = np.zeros((h, w), dtype=np.bool)
    rows = np.arange(0, h, dtype=np.int32)
    idx_map = np.tile(np.arange(0, w, dtype=np.int32), h).reshape((h, w))
    for _ in range(num_seams):
        seam = _get_forward_seam(gray, keep_mask)
        seams_mask[rows, idx_map[rows, seam]] = True
        seam_mask = _get_seam_mask(gray, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if keep_mask is not None:
            keep_mask = _remove_seam_mask(keep_mask, seam_mask)

    return seams_mask


def _get_seams(gray: np.ndarray, num_seams: int, energy_mode: str,
               keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """Get the minimum N seams from the grayscale image"""
    assert energy_mode in VALID_ENERGY_MODES
    if energy_mode == BACKWARD_ENERGY:
        return _get_backward_seams(gray, num_seams, keep_mask)
    else:
        return _get_forward_seams(gray, num_seams, keep_mask)


def _reduce_width(src: np.ndarray, delta_width: int, energy_mode: str,
                  keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """Reduce the width of image by delta_width pixels"""
    assert src.ndim in (2, 3) and delta_width >= 0
    if src.ndim == 2:
        gray = src
        src_h, src_w = src.shape
        dst_shape = (src_h, src_w - delta_width)
    else:
        gray = _rgb2gray(src)
        src_h, src_w, src_c = src.shape
        dst_shape = (src_h, src_w - delta_width, src_c)

    seams_mask = _get_seams(gray, delta_width, energy_mode, keep_mask)
    dst = src[~seams_mask].reshape(dst_shape)

    return dst


def _expand_width(src: np.ndarray, delta_width: int, energy_mode: str,
                  keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """Expand the width of image by delta_width pixels"""
    assert src.ndim in (2, 3) and delta_width >= 0
    if src.ndim == 2:
        gray = src
        src_h, src_w = src.shape
        dst_shape = (src_h, src_w + delta_width)
    else:
        gray = _rgb2gray(src)
        src_h, src_w, src_c = src.shape
        dst_shape = (src_h, src_w + delta_width, src_c)

    keep_mask_shape = (src_h, src_w + delta_width)
    seams_mask = _get_seams(gray, delta_width, energy_mode, keep_mask)
    dst = np.empty(dst_shape, dtype=np.uint8)
    if keep_mask is not None:
        global keep_mask_
        keep_mask_ = np.empty(keep_mask_shape, dtype=np.uint8)

    for row in range(src_h):
        dst_col = 0
        for src_col in range(src_w):
            if seams_mask[row, src_col]:
                lo = max(0, src_col - 1)
                hi = src_col + 1
                dst[row, dst_col] = src[row, lo:hi].mean(axis=0)
                if keep_mask is not None:
                    keep_mask_[row, dst_col] = 0
                dst_col += 1
            dst[row, dst_col] = src[row, src_col]
            if keep_mask is not None:
                keep_mask_[row, dst_col] = keep_mask[row, src_col]
            dst_col += 1
        assert dst_col == src_w + delta_width

    return dst


def _resize_width(src: np.ndarray, width: int, energy_mode: str,
                  keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """Resize the width of image by removing vertical seams"""
    assert src.size > 0 and src.ndim in (2, 3)
    assert width > 0
    assert energy_mode in VALID_ENERGY_MODES

    src_w = src.shape[1]
    if src_w < width:
        dst = _expand_width(src, width - src_w, energy_mode, keep_mask)
    else:
        dst = _reduce_width(src, src_w - width, energy_mode, keep_mask)
    return dst


def _resize_height(src: np.ndarray, height: int, energy_mode: str,
                   keep_mask: Optional[np.ndarray]) -> np.ndarray:
    """Resize the height of image by removing horizontal seams"""
    assert src.ndim in (2, 3) and height > 0
    if src.ndim == 3:
        if keep_mask is not None:
            keep_mask = keep_mask.T
        src = _resize_width(src.transpose((1, 0, 2)), height, energy_mode,
                            keep_mask).transpose((1, 0, 2))
    else:
        src = _resize_width(src.T, height, energy_mode, keep_mask).T
    return src


def _check_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Ensure the mask to be a 2D grayscale map of specific shape"""
    mask = np.asarray(mask, dtype=np.bool)
    if mask.ndim != 2:
        raise ValueError('Invalid mask of shape {}: expected to be a 2D '
                         'binary map'.format(mask.shape))
    if mask.shape != shape:
        raise ValueError('The shape of mask must match the image: expected {}, '
                         'got {}'.format(shape, mask.shape))
    return mask


def _check_src(src: np.ndarray) -> np.ndarray:
    """Ensure the source to be RGB or grayscale"""
    src = np.asarray(src, dtype=np.uint8)
    if src.size == 0 or src.ndim not in (2, 3):
        raise ValueError('Invalid src of shape {}: expected an 3D RGB image or '
                         'a 2D grayscale image'.format(src.shape))
    return src


def _get_TBMap(src: np.ndarray, size: Tuple[int, int],
               energy_mode: str = 'backward', order: str = 'width-first',
               keep_mask: Optional[np.ndarray] = None) -> np.ndarray:
    n, m = src.shape[:2]
    m_, n_ = size
    r, c = abs(n-n_), abs(m-m_)
    if src.ndim == 2:
        gray = src
    else:
        gray = _rgb2gray(src)

    T = np.zeros((c+1, r+1))
    TBMap = np.ones((c+1, r+1)) * (-1)
    T[0, 0] = 0

    temp = gray.copy()
    print(T.shape)
    for i in range(1, c+1):
        energy = _get_energy(temp)
        _, seam_cost = _get_backward_seam(energy)
        T[i, 0] = T[i-1, 0] + seam_cost
        TBMap[i, 0] = 1

    temp = gray.T.copy()
    for j in range(1, r+1):
        energy = _get_energy(temp)
        _, seam_cost = _get_backward_seam(energy)
        T[0, j] = T[0, j-1] + seam_cost
        TBMap[0, j] = 0

    gray = _reduce_width(gray, 1, energy_mode, keep_mask)
    gray = _reduce_width(gray.T, 1, energy_mode, keep_mask).T

    print(gray.shape)

    temp = gray.copy()

    for i in range(1, c+1):
        imgWithoutRow = temp.copy()
        for j in range(1, r+1):
            energy = _get_energy(imgWithoutRow)

            _, seam_cost_V = _get_backward_seam(energy)

            seam_H, seam_cost_H = _get_backward_seam(energy.T)
            seam_mask_H = _get_seam_mask(imgWithoutRow.T, seam_H)
            imgNoCol = _remove_seam_mask(imgWithoutRow.T, seam_mask_H)

            neighbors = [(T[i-1, j] + seam_cost_V),
                         (T[i, j-1] + seam_cost_H)]
            ind, val = np.argmin(neighbors), np.min(neighbors)
            T[i, j], TBMap[i, j] = val, ind
            imgWithoutRow = imgNoCol.T.copy()
        energy1 = _get_energy(temp)
        seam1, _ = _get_backward_seam(energy1)
        seam_mask1 = _get_seam_mask(temp, seam1)
        temp = _remove_seam_mask(temp, seam_mask1)

    return TBMap


def resize(src: np.ndarray, size: Tuple[int, int],
           energy_mode: str = 'backward', order: str = 'width-first',
           keep_mask: Optional[np.ndarray] = None) -> np.ndarray:

    src = _check_src(src)
    src_h, src_w = src.shape[:2]

    width, height = size
    width = int(round(width))
    height = int(round(height))
    if width <= 0 or height <= 0:
        raise ValueError('Invalid size {}: expected > 0'.format(size))
    if width >= 2 * src_w:
        raise ValueError('Invalid target width {}: expected less than twice '
                         'the source width (< {})'.format(width, 2 * src_w))
    if height >= 2 * src_h:
        raise ValueError('Invalid target height {}: expected less than twice '
                         'the source height (< {})'.format(height, 2 * src_h))

    if order not in VALID_ORDERS:
        raise ValueError('Invalid order {}: expected {}'.format(
            order, VALID_ORDERS))

    if energy_mode not in VALID_ENERGY_MODES:
        raise ValueError('Invalid energy mode {}: expected {}'.format(
            energy_mode, VALID_ENERGY_MODES))

    if keep_mask is not None:
        keep_mask = _check_mask(keep_mask, (src_h, src_w))

    global keep_mask_
    keep_mask_ = keep_mask
    if order == WIDTH_FIRST:
        src = _resize_width(src, width, energy_mode, keep_mask)
        src = _resize_height(src, height, energy_mode, keep_mask_)

    elif order == HEIGHT_FIRST:
        src = _resize_height(src, height, energy_mode, keep_mask)
        src = _resize_width(src, width, energy_mode, keep_mask_)

    elif order == OPTIMAL:
        TBMap = _get_TBMap(src, size, energy_mode, order, keep_mask)
        i, j = TBMap.shape
        print(TBMap)
        i -= 1
        j -= 1
        r = 0
        c = 0
        while True:
            if TBMap[i, j] == -1:
                break
            if TBMap[i, j] == 0:
                if src_h > height:
                    r -= 1
                else:
                    r += 1
                src = _resize_height(src, src_h + r, energy_mode, keep_mask)
                j -= 1
            else:
                if src_w > width:
                    c -= 1
                else:
                    c += 1
                src = _resize_width(src, src_w + c, energy_mode, keep_mask)
                i -= 1
    return src
