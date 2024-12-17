import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import tifffile
import mrcfile
import os

# Define a custom colormap: cyan to white to hot
colors_cyan = [
    (0.0, "black"),   # Start with cyan
    (0.7, "cyan"),   # Start with cyan
    (1.0, "white")     # End with hot (red at high values)
]
cyan_hot_cmap = LinearSegmentedColormap.from_list("cyan_hot", colors_cyan)

# Define a custom colormap for resolution map
colors_res = [
    (0.0, "green"),
    (0.3, "yellow"),
    # (0.7, "black"),
    (0.7, "red"),
    (1.0, "black")
]
res_cmap = LinearSegmentedColormap.from_list("res", colors_res)

def image_flip_tile(img):
    original = img
    flipped_horizontally = np.flip(img, axis=1)  # Left-right flip
    flipped_vertically = np.flip(img, axis=0)   # Top-bottom flip
    flipped_both = np.flip(img, axis=(0, 1))    # Both axes flip

    # Create a tiled grid: 2x2
    top_row = np.hstack((original, flipped_horizontally))
    bottom_row = np.hstack((flipped_vertically, flipped_both))
    img_flip_tile = np.vstack((top_row, bottom_row))

    return img_flip_tile

def im_read(path):
    file_name, file_extension = os.path.splitext(path)

    if file_extension in [".tif",".tiff"]:
        return tifffile.imread(path)
    elif file_extension == ".mrc":
        with mrcfile.open(path, permissive=True) as mrc:
            # Access the data
            mrc_data = mrc.data
            print("voxel size:",mrc.voxel_size)
        return np.transpose(mrc_data,axes=(0,2,1))
    else:
        raise NotImplementedError("File type {} not supported".format(file_extension))