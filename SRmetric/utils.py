import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Define a custom colormap: cyan to white to hot
colors_cyan = [
    (0.0, "black"),   # Start with cyan
    (0.7, "cyan"),   # Start with cyan
    (1.0, "white")     # End with hot (red at high values)
]
cyan_hot_cmap = LinearSegmentedColormap.from_list("cyan_hot", colors_cyan)

def image_filp_tile(img):
    original = img
    flipped_horizontally = np.flip(img, axis=1)  # Left-right flip
    flipped_vertically = np.flip(img, axis=0)   # Top-bottom flip
    flipped_both = np.flip(img, axis=(0, 1))    # Both axes flip

    # Create a tiled grid: 2x2
    top_row = np.hstack((original, flipped_horizontally))
    bottom_row = np.hstack((flipped_vertically, flipped_both))
    img_flip_tile = np.vstack((top_row, bottom_row))

    return img_flip_tile
