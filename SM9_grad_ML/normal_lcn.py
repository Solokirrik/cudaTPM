import numpy as np
import scipy as sp
import scipy.ndimage

class LCN():

    @staticmethod
    def gaussian_kernel(sigma, size=None):
        if size is None:
            size = int(np.ceil(sigma * 2.))
            if size % 2 == 0:
                size += 1
        xs = np.linspace(-size / 2., size / 2., size)
        kernel = 1 / (np.sqrt(2 * np.pi)) * \
            np.exp(-xs**2 / (2 * sigma**2)) / sigma
        return kernel / np.sum(kernel)

    def __init__(self, sigma, eps=1e-1, subtractive=False):
        self.sigma = sigma
        self.eps = eps
        self.subtractive = subtractive
        self.kernel = LCN.gaussian_kernel(sigma, size=int(sigma * 3))
        self.kernel = np.outer(self.kernel, self.kernel)

    def _normalize(self, x):
        return (x.astype(np.float_) - self._mean) / self._std

    def fit(self, x):
        self._mean = np.mean(x, dtype=np.float64).astype(x.dtype)
        self._std = np.std(x, dtype=np.float64).astype(x.dtype)
        return self

    def _transform_img(self, img):
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        n_channels = img.shape[0]

        # Calculate local mean
        mean = np.zeros((1,) + img.shape[1:])
        for i in range(n_channels):
            mean += sp.ndimage.filters.convolve(
                img[i], self.kernel, mode='nearest')
        mean /= n_channels

        # Center input with local mean
        centered = img - mean
        if self.subtractive:
            return centered

        # Calculate local standard deviation
        centered_sqr = centered**2
        std = np.zeros((1,) + img.shape[1:])
        for i in range(n_channels):
            std += sp.ndimage.filters.convolve(
                centered_sqr[i], self.kernel, mode='nearest')
        std /= n_channels
        std = np.sqrt(std)

        # Scale centered input with standard deviation
        return centered / (std + self.eps)

    def transform(self, x):
        x = self._normalize(x)
        return np.array([self._transform_img(img) for img in x])