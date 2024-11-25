import numpy as np
from colour.appearance import (
    VIEWING_CONDITIONS_CIECAM02,
    XYZ_to_CIECAM02,
    CAM_Specification_CIECAM02
)


def convert_xyz_to_ciecam02(
        XYZ,
        XYZ_w,
        L_A,
        Y_b,
        surround_condition="Average"
):
    """
    Convert XYZ tristimulus values to CIECAM02 specifications.

    Parameters
    ----------
    XYZ : array_like
        XYZ tristimulus values of test sample
    XYZ_w : array_like
        XYZ tristimulus values of reference white
    L_A : numeric
        Adapting field luminance in cd/m²
    Y_b : numeric
        Relative luminance of background
    surround_condition : str, optional
        Surround condition (Average, Dim, Dark)

    Returns
    -------
    CAM_Specification_CIECAM02
        CIECAM02 color appearance model specification
    """
    # Convert inputs to numpy arrays if they aren't already
    XYZ = np.array(XYZ)
    XYZ_w = np.array(XYZ_w)

    # Get the surround parameters
    surround = VIEWING_CONDITIONS_CIECAM02[surround_condition]

    # Perform the conversion
    specification = XYZ_to_CIECAM02(
        XYZ=XYZ,
        XYZ_w=XYZ_w,
        L_A=L_A,
        Y_b=Y_b,
        surround=surround
    )

    return specification


# Example usage
if __name__ == "__main__":
    # Input values
    XYZ = np.array([[19.01, 20.00, 21.78],
                   [0.44, 27.21, 169.83]])
    XYZ_w = np.array([95.05, 100.00, 108.88])
    L_A = 318.31
    Y_b = 20.0

    # Perform conversion
    result = convert_xyz_to_ciecam02(XYZ, XYZ_w, L_A, Y_b)
    print(result)