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


def get_RGB_reference(colour_checker):
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


def analyze_colour_accuracy(file_path):
    """

    The goal is to get the measured color swatch data and our RGB reference data into the same color space in order to compute the CCM. The we will all working colorspaces to LAB to compute the ΔE2000 analysis.

    """
    colour_checker = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']

    swatches = detect_patches(file_path)
    RGB_reference = get_RGB_reference(colour_checker)

    RGB_measured = swatches[0]

    colour_correction_matrix = compute_colour_correction_matrix(
        RGB_measured, RGB_reference)

    RGB_corrected = swatches[0] @ colour_correction_matrix

    # RGB is not one colorspace, we need to define we are talking about sRGB
    sRGB = colour.RGB_COLOURSPACES['sRGB']

    # convert uncorrected values to XYZ -> Lab color space
    XYZ_uncorrected = colour.RGB_to_XYZ(
        RGB_measured, sRGB, apply_cctf_decoding=False)
    Lab_uncorrected = colour.XYZ_to_Lab(XYZ_uncorrected)

    # convert corrected values to XYZ -> Lab color space
    XYZ_corrected = colour.RGB_to_XYZ(
        RGB_corrected, sRGB, apply_cctf_decoding=False)
    Lab_corrected = colour.XYZ_to_Lab(XYZ_corrected)

    # convert reference values to XYZ -> Lab color space
    XYZ_reference = colour.RGB_to_XYZ(
        RGB_reference, sRGB, apply_cctf_decoding=False)
    Lab_reference = colour.XYZ_to_Lab(XYZ_reference)

    # calculate ΔE2000 for corrected and uncorrected values
    delta_e_values = colour.delta_E(
        Lab_corrected, Lab_reference, method='CIE 2000')
    delta_e_uncorrected_values = colour.delta_E(
        Lab_uncorrected, Lab_reference, method='CIE 2000')

    print(f"\n{'patch':<8} {'uncorrected':>12} {'corrected':>12} {'improvement':>12}")
    print("-" * 48)
    for i, (u, c) in enumerate(zip(delta_e_uncorrected_values, delta_e_values)):
        print(f"patch {i+1:2d}   {u:>12.4f} {c:>12.4f} {u - c:>+12.4f}")
    print("-" * 48)
    print(f"{'mean':<8} {delta_e_uncorrected_values.mean():>12.4f} {delta_e_values.mean():>12.4f} {delta_e_uncorrected_values.mean() - delta_e_values.mean():>+12.4f}")
    print(f"{'max':<8} {delta_e_uncorrected_values.max():>12.4f} {delta_e_values.max():>12.4f}")

    return RGB_reference, RGB_corrected, delta_e_values


if __name__ == '__main__':
    analyze_colour_accuracy(sony_img)

