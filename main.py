import numpy as np
import rawpy
import matplotlib.pyplot as plt
import colour_checker_detection as ccd
import colour


sony_img = "./imgs/AKG07755.ARW"


def display_arw_image(file_path):
    """
    !!NOTE!!: This is intentially using defaults for quick preview. 

    Converting and displaying sensor data into a standard RGB image using rawpy using default settings. 

    """
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess()
    plt.imshow(rgb)
    plt.title('Color Chart ARW Image')
    plt.axis('off')
    plt.show()


def detect_patches(file_path):
    """

    Convert sensor data and removing auto balancing to get only linear light off the sensor and detect color patches by returning an array of RGB values per patch. 

    """
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess(output_bps=16,  # 16-bit for better precision
                              no_auto_bright=True,
                              use_camera_wb=True,
                              gamma=(1, 1),  # linear -- no gamma curve applied
                              output_color=rawpy.ColorSpace.sRGB) # set the output color space

    # normalize to 0-1 float
    image = rgb.astype(np.float32) / 65535.0

    # detect the color checker
    swatches = ccd.detect_colour_checkers_segmentation(image)

    print(f'Found {len(swatches)} patches')
    print(swatches)


# detect_patches(sony_img)

# print()
cc = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
# print(cc.data)

# We need to convert our reference values (xyY -> XYZ -> Linear sRGB)
# so both sides of the CCM equation live in the same space 

xyY_values = np.array(list(cc.data.values()))
print(f"xyY's shape: {xyY_values.shape}")

# converting xyY -> XYZ color space
XYZ_values = colour.xyY_to_XYZ(xyY_values)
print(f"Reference values converted to XYZ color space: \n{XYZ_values}")


# display_arw_image(sony_img)
