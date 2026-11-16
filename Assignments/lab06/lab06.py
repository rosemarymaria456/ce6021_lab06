# -*- coding: utf-8 -*-
"""Lab 06 — Scale Invariant Blob Detector.

Detect blobs (flower heads) at their natural scale, using a Laplacian-of-
Gaussian (LoG) scale space. The work is split into small steps, each one a
method below that takes an array in and returns the result out:

  1. gauss_kernel(sigma)                   1D Gaussian kernel
  2. gauss_2nd_deriv_kernel(sigma)         1D second-derivative-of-Gaussian kernel
  3. second_derivatives(image, sigma)      Lxx, Lyy by separable convolution
  4. laplacian(image, sigma)               scale-normalised LoG at one scale
  5. build_scale_space(image)              LoG stacked over every scale
  6. candidate_points(scale_space)         points above the threshold
  7. lap_max_det(scale_space, row, col, k) is this point a local maximum?
  8. find_maxima(scale_space)              every local maximum
  9. blob_circles(maxima)                  centre and radius of each blob
 10. detect_blobs(image)                   put the steps together

Run evaluate_lab06.py after each step to see your progress.
"""
import numpy as np
import os
import sys
from scipy import signal
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import DATA_DIR, download_image, norm_img  # noqa: F401

FLOWERS_IMAGE_URL = (
    "https://github.com/tonyscan6003/etivities/blob/main/sunflowers_temp.jpg?raw=true"
)

class ScaleInvDet:
    """Scale Invariant Blob Detector using a Laplacian-of-Gaussian (LoG) scale space.

    All settings have defaults, so no arguments are needed:

        sid   = ScaleInvDet()
        blobs = sid.detect_blobs(image)

    Any setting can be changed with a keyword argument, e.g.:

        sid = ScaleInvDet(sigma_max=10, thresh=60)

    Use these settings (self.sigma_vals, self.thresh, ...) inside your
    methods rather than typing the numbers in directly.
    """

    def __init__(self, **kwargs):
        self.sigma_min = kwargs.get('sigma_min', 1)
        self.sigma_max = kwargs.get('sigma_max', 36)
        self.scale_fac = kwargs.get('scale_fac', 1.05)
        self.thresh    = kwargs.get('thresh',    40)

        # The scales of the scale space, worked out from the settings above
        self.num_scales = int((self.sigma_max - self.sigma_min) / self.scale_fac)
        self.sigma_vals = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)

        # Intermediate results, filled in by detect_blobs() (step 10)
        self.laplace_scale_space = None
        self.maxima              = None
        self.blobs               = None

    def gauss_kernel(self, sigma):
        """Step 1 — a 1D Gaussian kernel, as a 1 x k row vector.

        Args:
            sigma (float): Gaussian scale parameter.

        Returns:
            np.ndarray: shape (1, k), k = 2 * ((int(6 * sigma + 1) - 1) // 2) + 1.
        """
        k = int(6 * sigma + 1)
        half = (k - 1) // 2
        x = np.arange(-half, half + 1)
        g = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x ** 2) / (2 * sigma ** 2))
        return np.expand_dims(g, axis=0)

    def gauss_2nd_deriv_kernel(self, sigma):
        """Step 2 — the second derivative of the 1D Gaussian, as a 1 x k row vector.

        Args:
            sigma (float): Gaussian scale parameter.

        Returns:
            np.ndarray: shape (1, k) — the same size as gauss_kernel(sigma).
        """
        k = int(6 * sigma + 1)
        half = (k - 1) // 2
        x = np.arange(-half, half + 1)
        g = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x ** 2) / (2 * sigma ** 2))
        gxx = g * (x ** 2 - sigma ** 2) / (sigma ** 4)
        return np.expand_dims(gxx, axis=0)

    def second_derivatives(self, image, sigma):
        """Step 3 — the second derivatives Lxx and Lyy of the Gaussian-smoothed
        image, each built from two separable 1D convolutions.

        Args:
            image (np.ndarray): 2D greyscale image.
            sigma (float):      Gaussian scale parameter.

        Returns:
            (Lxx, Lyy): each a 2D array the same shape as image.
        """
        G = self.gauss_kernel(sigma)
        Gxx = self.gauss_2nd_deriv_kernel(sigma)
        
        Lxx = signal.convolve2d(image, Gxx, boundary='symm', mode='same')
        Lxx = signal.convolve2d(Lxx, G.T, boundary='symm', mode='same')
        
        Lyy = signal.convolve2d(image, Gxx.T, boundary='symm', mode='same')
        Lyy = signal.convolve2d(Lyy, G, boundary='symm', mode='same')
        
        return Lxx, Lyy

    def laplacian(self, image, sigma):
        """Step 4 — the scale-normalised Laplacian of Gaussian at one scale,
        sigma^2 * (Lxx + Lyy).

        Args:
            image (np.ndarray): 2D greyscale image.
            sigma (float):      Gaussian scale parameter.

        Returns:
            np.ndarray: 2D LoG response, same shape as image.
        """
        Lxx, Lyy = self.second_derivatives(image, sigma)
        return (sigma ** 2) * (Lxx + Lyy)

    def build_scale_space(self, image):
        """Step 5 — the LoG response at every scale in self.sigma_vals,
        stacked into one 3D array.

        Args:
            image (np.ndarray): 2D greyscale image of shape (m, n).

        Returns:
            np.ndarray: shape (m, n, num_scales) — layer k is the LoG at sigma_vals[k].
        """
        layers = []
        for sigma in tqdm(self.sigma_vals, desc="Building scale space"):
            layers.append(self.laplacian(image, sigma))
        return np.dstack(layers)

    def candidate_points(self, scale_space):
        """Step 6 — every interior point of the scale space whose response is
        above self.thresh. Points on the outer layer (first/last row, column
        or scale) are left out, since they don't have a full 3x3x3 neighbourhood.

        Args:
            scale_space (np.ndarray): (m, n, K) array from build_scale_space().

        Returns:
            np.ndarray: shape (P, 3) int array, one (row, col, k) per candidate.
        """
        interior = scale_space[1:-1, 1:-1, 1:-1]
        pts = np.argwhere(interior > self.thresh)
        pts += 1  # offset back to full volume coordinates
        return pts

    def lap_max_det(self, scale_space, row, col, k):
        """Step 7 — is (row, col, k) a local maximum of the scale space,
        compared with its 26 neighbours in the surrounding 3x3x3 block?

        Args:
            scale_space (np.ndarray): (m, n, K) array from build_scale_space().
            row, col    (int):        position in the image.
            k           (int):        scale index.

        Returns:
            list: [magnitude, row, col, sigma] if it is a maximum, else [].
        """
        val = scale_space[row, col, k]
        block = scale_space[row-1:row+2, col-1:col+2, k-1:k+2]
        if val + 0.001 > np.max(block):
            return [val, row, col, self.sigma_vals[k]]
        return []

    def find_maxima(self, scale_space):
        """Step 8 — every candidate point that is a local maximum.

        Args:
            scale_space (np.ndarray): (m, n, K) array from build_scale_space().

        Returns:
            np.ndarray: shape (N, 4); each row is [magnitude, row, col, sigma].
                        Shape (0, 4) if there are none.
        """
        maxima = []
        for row, col, k in tqdm(self.candidate_points(scale_space), desc="Finding maxima"):
            result = self.lap_max_det(scale_space, row, col, k)
            if result:
                maxima.append(result)
        
        if not maxima:
            return np.empty((0, 4))
        return np.array(maxima)

    def blob_circles(self, maxima):
        """Step 9 — the circle that describes each blob: its centre, and a
        radius of sqrt(2) * sigma, rounded up to a whole pixel.

        Args:
            maxima (np.ndarray): (N, 4) array from find_maxima().

        Returns:
            np.ndarray: shape (N, 3) int array; each row is [row, col, radius].
        """
        radius = np.ceil(maxima[:, 3] * np.sqrt(2))
        return np.column_stack((maxima[:, 1], maxima[:, 2], radius)).astype(int)

    def detect_blobs(self, image):
        """Step 10 — run the whole pipeline using the methods above.

        Stores each intermediate result on self so they can be displayed
        afterwards.

        Args:
            image (np.ndarray): 2D greyscale image.

        Returns:
            np.ndarray: shape (N, 3) int array of [row, col, radius], one per blob.
        """
        self.laplace_scale_space = self.build_scale_space(image)
        self.maxima = self.find_maxima(self.laplace_scale_space)
        self.blobs = self.blob_circles(self.maxima)
        return self.blobs
