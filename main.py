import numpy as np
import rawpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colour_checker_detection as ccd
import colour


sony_img = "./imgs/AKG07755.ARW"
iphone_img = "./imgs/IMG_2188.DNG"


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
        wb = list(raw.daylight_whitebalance)
        wb[3] = wb[1]
        rgb = raw.postprocess(output_bps=16,  # 16-bit for better precision
                              no_auto_bright=True,
                              use_camera_wb=False,
                              user_wb=wb,
                              gamma=(1, 1),  # linear -- no gamma curve applied
                              output_color=rawpy.ColorSpace.sRGB)  # set the output color space # type: ignore

    # normalize to 0-1 float
    image = rgb.astype(np.float32) / 65535.0

    # detect the color checker
    swatches = ccd.detect_colour_checkers_segmentation(
        image, additional_data=True)
    data = swatches[0]

    return image, data.colour_checker, data.swatch_colours


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


def analyze_colour_accuracy(file_path, label):
    """

    The goal is to get the measured color swatch data and our RGB reference data into the same color space in order to compute the CCM. The we will all working colorspaces to LAB to compute the ΔE2000 analysis.

    """
    colour_checker = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']

    image, checker_crop, RGB_measured = detect_patches(file_path)
    RGB_reference = get_RGB_reference(colour_checker)

    colour_correction_matrix = compute_colour_correction_matrix(
        RGB_measured, RGB_reference)

    RGB_corrected = RGB_measured @ colour_correction_matrix

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

    print(f"\n{label}")
    print(f"\n{'patch':<8} {'uncorrected':>12} {'corrected':>12} {'improvement':>12}")
    print("-" * 48)
    for i, (u, c) in enumerate(zip(delta_e_uncorrected_values, delta_e_values)):
        print(f"patch {i+1:2d}   {u:>12.4f} {c:>12.4f} {u - c:>+12.4f}")
    print("-" * 48)
    print(f"{'mean':<8} {delta_e_uncorrected_values.mean():>12.4f} {delta_e_values.mean():>12.4f} {delta_e_uncorrected_values.mean() - delta_e_values.mean():>+12.4f}")
    print(f"{'max':<8} {delta_e_uncorrected_values.max():>12.4f} {delta_e_values.max():>12.4f}")

    return image, checker_crop, RGB_reference, RGB_corrected, delta_e_values, delta_e_uncorrected_values


def visualize_swatches(image, checker_crop, RGB_reference, RGB_corrected, delta_e, color_patches):
    """

    Before we can visualize, we need to understand our RGB data is LINEAR. We will need to apply gamma encoding before display so the colors look correct visually.

    """
    # add gamma to linear RGB values
    RGB_reference_with_gamma = colour.cctf_encoding(RGB_reference)
    RGB_corrected_with_gamma = colour.cctf_encoding(RGB_corrected)

    # create the figure
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(3, 24, height_ratios=[4, 1, 1])
    ax_photo = fig.add_subplot(gs[0, :12])
    ax_checker = fig.add_subplot(gs[0, 12:])
    axes = np.array([[fig.add_subplot(gs[r+1, c])
                    for c in range(24)] for r in range(2)])

    ax_photo.imshow(colour.cctf_encoding(np.clip(image, 0, 1)))
    ax_photo.axis('off')
    ax_photo.set_title('Scene', fontsize=9)

    ax_checker.imshow(colour.cctf_encoding(np.clip(checker_crop, 0, 1)))
    ax_checker.axis('off')
    ax_checker.set_title('Detected ColorChecker', fontsize=9)

    # fill each column
    for i in range(24):
        axes[0, i].imshow([[RGB_reference_with_gamma[i]]])
        axes[1, i].imshow([[RGB_corrected_with_gamma[i]]])
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        color = 'green' if delta_e[i] <= 1 else 'orange' if delta_e[i] <= 3 else 'red'
        axes[1, i].set_title(f"{delta_e[i]:.1f}", fontsize=10, color=color, fontweight='bold')
        axes[1, i].text(0.5, -0.1, color_patches[i], transform=axes[1, i].transAxes,
                        fontsize=6, rotation=45, ha='right', va='top')



    # label the rows and show
    axes[0, 0].text(-0.5, 0.5, 'Reference', transform=axes[0, 0].transAxes, fontsize=9, ha='right')
    axes[1, 0].text(-0.5, 0.5,'Corrected', transform=axes[1, 0].transAxes, fontsize=9, ha='right')
    plt.suptitle('ColorChecker: Reference vs Corrected (ΔE2000)')
    plt.tight_layout()

    green_patch = mpatches.Patch(color='green', label='≤1: Imperceptible')
    orange_patch = mpatches.Patch(color='orange', label='≤3: Acceptable')
    red_patch = mpatches.Patch(color='red', label='>3: Noticeable')
    fig.legend(handles=[green_patch, orange_patch, red_patch],
               title=f"Mean ΔE: {delta_e.mean():.2f}",
               loc='upper right', fontsize=8)
    
    plt.savefig('./delta_e_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_gamut(RGB_reference, RGB_corrected):

    sRGB = colour.RGB_COLOURSPACES['sRGB']

    # convert corrected values to XYZ from sRGB color space
    # convert corrected values to xy from XYZ
    RGB_corrected_XYZ = colour.RGB_to_XYZ(
        RGB_corrected, sRGB, apply_cctf_decoding=False)
    RGB_corrected_xy = colour.XYZ_to_xy(RGB_corrected_XYZ)

    # convert corrected values to XYZ from sRGB color space
    # convert corrected values to xy from XYZ
    RGB_reference_XYZ = colour.RGB_to_XYZ(
        RGB_reference, sRGB, apply_cctf_decoding=False)
    RGB_reference_xy = colour.XYZ_to_xy(RGB_reference_XYZ)

    # cordinates obtained for plot

    fig, ax = colour.plotting.plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931([
                                                                                    'sRGB'], show=False)

    # Add scatter points for x values [0] and y values [1]
    ax.scatter(RGB_reference_xy[:, 0],
               RGB_reference_xy[:, 1], label='Reference')
    ax.scatter(RGB_corrected_xy[:, 0],
               RGB_corrected_xy[:, 1], label='Corrected')
    ax.legend()

    for i in range(len(RGB_reference_xy)):
        ax.annotate('',
                    xy=RGB_corrected_xy[i],       # arrow tip
                    xytext=RGB_reference_xy[i],   # arrow tail
                    arrowprops=dict(arrowstyle='->', color='white', lw=0.8))

    # save file
    plt.savefig('./gamut_plot.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_cameras(iphone_delta_e: np.ndarray, sony_delta_e: np.ndarray, color_patches: list[str]) -> None:
    """
    Compare the results between the Iphone 13 Pro Max and Sony a7IV on a bar chart
    """
    # create the figure
    fig, ax = plt.subplots(figsize=(16, 6))

    sony_mean = sony_delta_e.mean()
    iphone_mean = iphone_delta_e.mean()


    x = np.arange(24)
    ax.set_xticks(x)
    ax.set_xticklabels( color_patches, rotation=45, ha='right',)
    width = .35

    ax.bar(x - width/2, sony_delta_e, width,
           color='steelblue', label='Sony a7IV')
    ax.bar(x + width/2, iphone_delta_e, width,
           color='dimgray', label='iPhone 13 Pro Max')
    ax.axhline(y=3, color='r', linestyle='--', label='Noticeable (ΔE=3)')
    ax.axhline(y=2, color='orange', linestyle='--',
               label='Noticeable but acceptable (ΔE=2)')
    ax.axhline(y=1, color='g', linestyle='--', label='Imperceptible (ΔE=1)')
    ax.legend()
    ax.text(0.01, 0.97, f"Sony Mean ΔE: {sony_mean:.2f}", transform=ax.transAxes)
    ax.text(0.01, 0.92, f"iPhone Mean ΔE: {iphone_mean:.2f}", transform=ax.transAxes)

    ax.set_xlabel('Patch')
    ax.set_ylabel('ΔE 2000')
    ax.set_title('Color Accuracy: Sony a7IV vs. iPhone 13 Pro Max')

    # TODO #11 create delta e mean annotation per camera
    # TODO #12 Add winner count subtitle
    # save file
    plt.savefig('./comparision.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    # Color patches:
    colour_checker = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
    color_patches = list(colour_checker.data.keys())

    # analyze Sony img
    sony_image, sony_checker_crop, sony_RGB_reference, sony_RGB_corrected, sony_delta_e, sony_delta_e_uncorrected = analyze_colour_accuracy(
        sony_img, 'Sony a7IV')
    # analyze iPhone img
    iphone_image, iphone_checker_crop, iphone_RGB_reference, iphone_RGB_corrected, iphone_delta_e, iphone_delta_e_uncorrected = analyze_colour_accuracy(
        iphone_img, 'iPhone 13 Pro Max')

    visualize_swatches(sony_image, sony_checker_crop, sony_RGB_reference,
                       sony_RGB_corrected, sony_delta_e, color_patches)
    plot_gamut(sony_RGB_reference, sony_RGB_corrected)
    compare_cameras(iphone_delta_e, sony_delta_e, color_patches)
