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
                              output_color=rawpy.ColorSpace.sRGB)  # set the output color space

    # normalize to 0-1 float
    image = rgb.astype(np.float32) / 65535.0

    # detect the color checker
    swatches = ccd.detect_colour_checkers_segmentation(image)

    return swatches


def get_reference_rgb(colour_checker):
    """
   We need to convert our reference values (xyY -> XYZ -> Linear sRGB) so both sides of the CCM equation live in the same space

    """

    xyY_values = np.array(list(colour_checker.data.values()))
    # print(f"xyY's shape: {xyY_values.shape}")

    # converting xyY -> XYZ color space
    XYZ_values = colour.xyY_to_XYZ(xyY_values)
    # print(f"Reference values converted to XYZ color space: \n{XYZ_values}")

    # converting XYZ -> sRGB
    # since camera data is linear (I explicitly set gamma=(1,1)), reference needs to match the camera data. We will add "apply_cctf_encoding=False" to make it linear.
    # RGB_reference = colour.XYZ_to_sRGB(XYZ_values, apply_cctf_encoding=False)

    # corrected chromatic adaptation mismatch. ColorChecker reference is measured under D50 while sRGB is defined under D65
    RGB_reference = colour.XYZ_to_sRGB(XYZ_values,
                                       illuminant=np.array(
                                           [0.3457, 0.3585]),  # D50 whitepoint
                                       apply_cctf_encoding=False
                                       )

    return RGB_reference


def compute_colour_correction_matrix(measured, reference):
    """
    Find the 3x3 matrix that best maps my measured patch colors to the reference patch colors, and give me just the matrix.
    """
    ccm, _, _, _ = np.linalg.lstsq(measured, reference, rcond=None)
    return ccm


colour_checker = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']

# print results of colour reference in sRGB:
# print(f"{get_reference_rgb(colour_checker)}")


# we now have both sides of the equation in sRGB (swatches and RGB_reference)

swatches = detect_patches(sony_img)
reference_rgb = get_reference_rgb(colour_checker)

measured = swatches[0]
colour_correction_matrix = compute_colour_correction_matrix(
    measured, reference_rgb)

corrected = swatches[0] @ colour_correction_matrix
# print(corrected)

print("corrected vs reference:")
for i, (c, r) in enumerate(zip(corrected, reference_rgb)):
    print(f"patch {i+1:2d}:  corrected={c}  ref={r}")


# print("measured last 6:")
# print(swatches[0][-6:])

# print("reference last 6:")
# print(reference_rgb[-6:])

# need to convert our corrected RGB values to the LAB color space.
XYZ = colour.RGB_to_XYZ(corrected)
Lab = colour.XYZ_to_Lab(XYZ)

