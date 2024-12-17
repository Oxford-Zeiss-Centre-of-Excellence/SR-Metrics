import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import pandas as pd
from scipy.fftpack import fft2, fftshift, ifft2

from SRmetric.utils import *

class SRMetric:
    """
    A Python class object to compute superresolution quality metric

    Attributes:
    -----------
    img1: numpy.ndarray
        2D numpy array of pre-processed image
    img2: numpy.ndarray
        2D numpy array of processed image
    pixel_size: float
        Physical pixel size of the image, in um
    """
    def __init__(self, 
                 img1,
                 img2,
                 pixel_size=1,
                 verbose = True,
                 ):
        assert len(img1.shape) == 2, "img1 must be a 2D numpy array"
        self.img1 = img1
        assert len(img2.shape) == 2, "img2 must be a 2D numpy array"
        self.img2 = img2

        assert img1.shape == img2.shape, "Both images must have the same shape."

        self.pixel_size=pixel_size

        self.verbose = verbose

    def compute_nMSE(self):
        img1_norm = (self.img1 - np.min(self.img1))/(np.max(self.img1)-np.min(self.img2))
        img2_norm = (self.img2 - np.min(self.img2))/(np.max(self.img2)-np.min(self.img2))

        # Compute the squared norm of the difference
        diff_squared = np.sum((img1_norm - img2_norm) ** 2)
        
        # Compute the squared norm of the reference image
        norm_squared = np.sum(img1_norm ** 2)
        
        # Compute NMSE
        nMSE = diff_squared / norm_squared
        return nMSE

    def compute_PSNR(self):
        """
        Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
                
        Returns:
            float: The PSNR value in decibels (dB).
        """

        img1_norm = (self.img1 - np.min(self.img1))/(np.max(self.img1)-np.min(self.img2))
        img2_norm = (self.img2 - np.min(self.img2))/(np.max(self.img2)-np.min(self.img2))
        mse = np.mean((img1_norm - img2_norm) ** 2)
        if mse == 0:
            return float('inf')  # Infinite PSNR if images are identical
        
        max_value = 1 # max value of the image input
        psnr = 10 * np.log10((max_value ** 2) / mse)
        return psnr
    
    def compute_SSIM(self, c1=1e-4, c2=9e-4):
        """
        Compute the Structural Similarity Index (SSIM) between two images.
        
        Parameters:
            c1 (float): Small constant to stabilize the division (default: 1e-4).
            c2 (float): Small constant to stabilize the division (default: 9e-4).
        
        Returns:
            float: The SSIM value.
        """
        img1_norm = (self.img1 - np.min(self.img1))/(np.max(self.img1)-np.min(self.img2))
        img2_norm = (self.img2 - np.min(self.img2))/(np.max(self.img2)-np.min(self.img2))

        # Means
        mu1 = np.mean(img1_norm)
        mu2 = np.mean(img2_norm)
        
        # Variances and Covariance
        sigma1_sq = np.mean((img1_norm - mu1) ** 2)
        sigma2_sq = np.mean((img2_norm - mu2) ** 2)
        sigma12 = np.mean((img1_norm - mu1) * (img2_norm - mu2))
        
        # SSIM formula
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim

    def compute_PCC(self):
        """
        Compute the Pearson Correlation Coefficient (PCC) between two images.
        
        Returns:
            float: The PCC value.
        """
        img1_norm = (self.img1 - np.min(self.img1))/(np.max(self.img1)-np.min(self.img2))
        img2_norm = (self.img2 - np.min(self.img2))/(np.max(self.img2)-np.min(self.img2))

        # Flatten the images to compute covariance and variance
        image1_flat = img1_norm.flatten()
        image2_flat = img2_norm.flatten()
        
        # Means
        mean1 = np.mean(image1_flat)
        mean2 = np.mean(image2_flat)
        
        # Covariance and standard deviations
        covariance = np.mean((image1_flat - mean1) * (image2_flat - mean2))
        std1 = np.std(image1_flat)
        std2 = np.std(image2_flat)
        
        pcc = covariance / (std1 * std2)
        return pcc
    
    def compute_mutual_info(self, bins=256):
        """
        Compute the Mutual Information (MI) between two images.
        
        Parameters:
            bins (int): The number of bins to use for the histogram (default: 256).
        
        Returns:
            float: The MI value.
        """
        img1_norm = (self.img1 - np.min(self.img1))/(np.max(self.img1)-np.min(self.img2))
        img2_norm = (self.img2 - np.min(self.img2))/(np.max(self.img2)-np.min(self.img2))

        # Compute the joint histogram
        joint_hist, _, _ = np.histogram2d(img1_norm.ravel(), img2_norm.ravel(), bins=bins)
        
        # Normalize the histogram to get the joint probability distribution
        joint_prob = joint_hist / np.sum(joint_hist)
        
        # Compute the marginal probabilities
        prob1 = np.sum(joint_prob, axis=1)
        prob2 = np.sum(joint_prob, axis=0)
        
        # Compute mutual information
        mi = 0.0
        for i in range(joint_prob.shape[0]):
            for j in range(joint_prob.shape[1]):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (prob1[i] * prob2[j]))

        return mi

    def compute_frc(self, num_bins=50, resolution_limit=1/7,flip_tile=False):
        """
        Computes the Fourier Ring Correlation (FRC) between two images.

        Parameters:
            num_bins: Number of frequency bins for the correlation.
            resolution_limit: Resolution threshold limit
        
        Returns:
            frequencies: Array of spatial frequencies.
            frc_values: Array of FRC values.
        """
        # Compute Fourier transforms and magnitudes
        if flip_tile:
            fft_1 = fftshift(fft2(image_flip_tile(self.img1)))
            fft_2 = fftshift(fft2(image_flip_tile(self.img2)))
        else:
            fft_1 = fftshift(fft2(self.img1))
            fft_2 = fftshift(fft2(self.img2))
        mag1 = np.abs(fft_1)
        mag2 = np.abs(fft_2)
        
        # Create coordinate grid
        ny, nx = fft_1.shape
        y, x = np.indices((ny, nx)) - np.array([ny // 2, nx // 2])[:, None, None]
        radius = np.sqrt(x**2 + y**2)
        max_radius = np.min([ny, nx]) // 2  # Nyquist limit

        # Bin frequencies into rings
        bins = np.linspace(0, max_radius, num_bins + 1)
        frc_values = []
        for i in range(len(bins) - 1):
            # Mask for the current ring
            mask = (radius >= bins[i]) & (radius < bins[i+1])
            num_points = np.sum(mask)
            if num_points > 0:
                # Compute FRC for the ring
                numerator = np.sum((fft_1[mask] * np.conj(fft_2[mask])).real)
                denominator = np.sqrt(np.sum(mag1[mask]**2) * np.sum(mag2[mask]**2))
                frc_values.append(numerator / denominator)
            else:
                frc_values.append(0)

        # Calculate frequencies as the midpoints of bins
        frequencies = (bins[:-1] + bins[1:]) / 2 / max_radius  # Normalized frequency

        # Find the first frequency where FRC drops below 1/7
        crossing_index = np.where(np.array(frc_values) < resolution_limit)[0]
        if len(crossing_index) == 0:
            raise ValueError("FRC does not drop below 1/7; resolution cannot be determined.")

        frc_cross_frequency = frequencies[crossing_index[0]]

        # Convert normalized frequency to spatial frequency
        spatial_frequency = frc_cross_frequency * (1 / (2 * self.pixel_size))

        # Convert spatial frequency to resolution
        resolution = 1 / spatial_frequency
        return frequencies, np.array(frc_values), resolution, frc_cross_frequency
    
    def compute_self_frc(self, num_bins=50, resolution_limit=1/7, sampling_rate=[1,1], sampling_method="max",flip_tile=True, plot=False):
        frequencies_res = []
        frc_values_res = [] 
        resolution_res = []
        frc_cross_frequency_res = []

        for i, img in enumerate([self.img1,self.img2]):
            fig, axs = plt.subplots(4,2, figsize=(8,16))

            if sampling_method == "max":
                # Ensure the array dimensions are divisible by 2
                assert img.shape[0] % sampling_rate[i] == 0 and img.shape[1] % sampling_rate[i] == 0, "Array dimensions must be divisible by sampling rate {}.".format(sampling_rate[i])

                # Reshape the array then perform max pooling
                reshaped = img.reshape(img.shape[0] // sampling_rate[i], sampling_rate[i], img.shape[1] // sampling_rate[i], sampling_rate[i])
                img = reshaped.max(axis=(1, 3))
            elif sampling_method == "Fourier":
                img_fft = fftshift(fft2(img))
                img_fft_cropped = img_fft[
                    (img_fft.shape[0] - img_fft.shape[0]//sampling_rate[i])//2:(img_fft.shape[0] + img_fft.shape[0]//sampling_rate[i])//2,
                    (img_fft.shape[1] - img_fft.shape[1]//sampling_rate[i])//2:(img_fft.shape[1] + img_fft.shape[1]//sampling_rate[i])//2,
                ]                    
                img = np.abs(ifft2(img_fft_cropped))

            if flip_tile:
                img = image_flip_tile(img)

            if sampling_method in ["max","Fourier"]:
                img1 = img[0::2, 0::2]
                img2 = img[1::2, 1::2]
            elif sampling_method == "skipping":
                img1 = img[::int(2*sampling_rate[i]), ::int(2*sampling_rate[i])]
                img2 = img[1*sampling_rate[i]::int(2*sampling_rate[i]), 1*sampling_rate[i]::int(2*sampling_rate[i])]

            # Compute Fourier transforms and magnitudes
            fft_1 = fftshift(fft2(img1))
            fft_2 = fftshift(fft2(img2))
            mag1 = np.abs(fft_1)
            mag2 = np.abs(fft_2)
                       
            # Create coordinate grid
            ny, nx = img1.shape
            y, x = np.indices((ny, nx)) - np.array([ny // 2, nx // 2])[:, None, None]
            radius = np.sqrt(x**2 + y**2)
            max_radius = np.min([ny, nx]) // 2  # Nyquist limit

            # Bin frequencies into rings
            bins = np.linspace(0, max_radius, num_bins//sampling_rate[i] + 1)
            # bins = np.linspace(0, max_radius, num_bins + 1)

            frc_values1 = []
            for j in range(len(bins) - 1):
                # Mask for the current ring
                mask = (radius >= bins[j]) & (radius < bins[j+1])
                num_points = np.sum(mask)
                if num_points > 0:
                    # Compute FRC for the ring
                    numerator = np.sum((fft_1[mask] * np.conj(fft_2[mask])).real)
                    denominator = np.sqrt(np.sum(mag1[mask]**2) * np.sum(mag2[mask]**2))
                    frc_values1.append(numerator / denominator)
                else:
                    frc_values1.append(0)

                if plot and j == (len(bins)-1-1):
                    axs[0,0].imshow(img1,cmap=cyan_hot_cmap)
                    axs[0,1].imshow(img2,cmap=cyan_hot_cmap)
                    axs[1,0].imshow(mag1,norm=colors.LogNorm(vmin=1e-4, vmax=mag1.max()))
                    axs[1,1].imshow(mag2,norm=colors.LogNorm(vmin=1e-4, vmax=mag2.max()))

                    axs[2,0].imshow(mask)
                    num = (fft_1*mask * np.conj(fft_2*mask)).real
                    axs[2,1].imshow(num,norm=colors.LogNorm(vmin=1e-4, vmax=num.max()))
                    axs[3,0].imshow((fft_1*mask).real,norm=colors.LogNorm(vmin=1e-4, vmax=np.max((fft_1*mask).real)))
                    axs[3,1].imshow((fft_2*mask).real,norm=colors.LogNorm(vmin=1e-4, vmax=np.max((fft_2*mask).real)))

            if sampling_method in ["max","Fourier"]:
                img1 = img[1::2, 0::2]
                img2 = img[0::2, 1::2]
            elif sampling_method == "skipping":
                img1 = img[1*sampling_rate[i]::int(2*sampling_rate[i]), 0::int(2*sampling_rate[i])]
                img2 = img[0::int(2*sampling_rate[i]), 1*sampling_rate[i]::int(2*sampling_rate[i])]

            # Compute Fourier transforms and magnitudes
            fft_1 = fftshift(fft2(img1))
            fft_2 = fftshift(fft2(img2))
            mag1 = np.abs(fft_1)
            mag2 = np.abs(fft_2)

            # Create coordinate grid
            ny, nx = img1.shape
            y, x = np.indices((ny, nx)) - np.array([ny // 2, nx // 2])[:, None, None]
            radius = np.sqrt(x**2 + y**2)
            max_radius = np.min([ny, nx]) // 2  # Nyquist limit

            # Bin frequencies into rings
            bins = np.linspace(0, max_radius, num_bins//sampling_rate[i] + 1)
            # bins = np.linspace(0, max_radius, num_bins + 1)

            frc_values2 = []
            for j in range(len(bins) - 1):
                # Mask for the current ring
                mask = (radius >= bins[j]) & (radius < bins[j+1])
                num_points = np.sum(mask)
                if num_points > 0:
                    # Compute FRC for the ring
                    numerator = np.sum((fft_1[mask] * np.conj(fft_2[mask])).real)
                    denominator = np.sqrt(np.sum(mag1[mask]**2) * np.sum(mag2[mask]**2))
                    frc_values2.append(numerator / denominator)
                else:
                    frc_values2.append(0)

            frc_values = (np.mean([frc_values1,frc_values2],axis=0))
            
            # Calculate frequencies as the midpoints of bins
            frequencies = (bins[:-1] + bins[1:]) / 2 / max_radius  # Normalized frequency

            # Find the first frequency where FRC drops below 1/7
            crossing_index = np.where(frc_values < resolution_limit)[0]
            if len(crossing_index) == 0:
                # raiseValueError("FRC does not drop below 1/7; resolution cannot be determined.")
                frc_cross_frequency, resolution = None, None
            else:
                frc_cross_frequency = frequencies[crossing_index[0]]

                # Convert normalized frequency to spatial frequency
                spatial_frequency = frc_cross_frequency * (1 / (2 * self.pixel_size * sampling_rate[i]))

                # Convert spatial frequency to resolution
                resolution = 1 / spatial_frequency

            frequencies_res.append(frequencies)
            frc_values_res.append(frc_values)
            resolution_res.append(resolution)
            frc_cross_frequency_res.append(frc_cross_frequency)

        return frequencies_res, frc_values_res, resolution_res, frc_cross_frequency_res

    def compute_metric(self):
        res = {
            "nMSE": self.compute_nMSE(),
            "PSNR": self.compute_PSNR(),
            "SSIM": self.compute_SSIM(),
            "PCC": self.compute_PCC(),
            "MI": self.compute_mutual_info(),
            }
        
        res = pd.Series(res)

        if self.verbose:
            print("Image Relative Metric:")
            print(res)

        return res