import numpy as np
import rawpy
import matplotlib.pyplot as plt


sony_img = "./imgs/AKG07755.ARW"

# Post Process a RAW iamge


def display_arw_image(file_path):
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess()  # convert sensor data into a standard RGB image
    plt.imshow(rgb)
    plt.title('Color Chart ARW Image')
    plt.axis('off')
    plt.show()

display_arw_image(sony_img)