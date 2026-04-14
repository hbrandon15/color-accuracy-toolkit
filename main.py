import numpy as np
import rawpy
import matplotlib.pyplot as plt
import colour
import colour_checker_detection as ccd


sony_img = "./imgs/AKG07755.ARW"

# Post Process a RAW iamge


def display_arw_image(file_path):
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess()  # convert sensor data into a standard RGB image
    plt.imshow(rgb)
    plt.title('Color Chart ARW Image')
    plt.axis('off')
    plt.show()


def detect_patches(file_path): 
    with rawpy.imread(file_path) as raw: 
        rgb = raw.postprocess(output_bps=16) # 16-bit for better precision

    # normalize to 0-1 float
    image = rgb.astype(np.float32) / 65535.0

    # detect the color checker
    swatches = ccd.detect_colour_checkers_segmentation(image)

    print(f'Found {len(swatches)} patches')
    print(swatches)

detect_patches(sony_img)




# display_arw_image(sony_img)