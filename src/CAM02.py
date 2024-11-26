import numpy as np
import colour

class CIECAM02Converter:
    def __init__(self, XYZ_w=None, L_A=63, Y_b=25):
        """
        Initialize the CIECAM02Converter class with default or custom parameters.

        Parameters:
        - XYZ_w: Reference white point in XYZ (default is D65 white point).
        - L_A: Adapting luminance in cd/m² (default is 60).
        - Y_b: Relative luminance of the background (default is 25).
        """
        # Default to D65 white point if not provided
        self.XYZ_w = XYZ_w if XYZ_w is not None else np.array([193.25, 201.54, 197.48])
        self.XYZ_wl = np.array([9.74, 10.43, 11.14])
        self.L_A = L_A
        self.Y_b = Y_b

        # Define the surround conditions (average)
        self.surround = colour.VIEWING_CONDITIONS_CIECAM02["Average"]

    def convert_image(self, XYZ_image):
        """
        Convert an entire XYZ image to CIECAM02 parameters (lightness J, chroma C, hue H).

        Parameters:
        - XYZ_image: A 3D NumPy array of shape (H, W, 3), where H and W are the height and width of the image,
                     and the 3 channels represent X, Y, Z values.

        Returns:
        - A dictionary of 3D arrays with lightness (J), chroma (C), and hue (H), each of shape (H, W).
        """
        H, W, _ = XYZ_image.shape

        # Initialize empty arrays for lightness, chroma, and hue
        J_image = np.zeros((H, W))
        C_image = np.zeros((H, W))
        H_image = np.zeros((H, W))
        # i = 0
        # Process each pixel in the XYZ image
        for i in range(H):
            # print(i)
            for j in range(W):
                sample_XYZ = XYZ_image[i, j, :]
                cam02_result = colour.XYZ_to_CIECAM02(
                    sample_XYZ,
                    self.XYZ_w,
                    self.L_A,
                    self.Y_b,
                    self.surround
                )

                # Store results in the corresponding arrays
                J_image[i, j] = cam02_result.J  # Lightness
                C_image[i, j] = cam02_result.C  # Chroma
                H_image[i, j] = cam02_result.h  # Hue angle
            # i += 1
        return {"Lightness (J)": J_image, "Chroma (C)": C_image, "Hue (H)": H_image}

    def invert_image(self, J, C, K):
        """
        Convert an entire CIECAM02 image back to XYZ.

        Parameters:
        - J: A 2D NumPy array of lightness values (J) with shape (H, W).
        - C: A 2D NumPy array of chroma values (C) with shape (H, W).
        - K: A 2D NumPy array of hue angles (h) with shape (H, W).

        Returns:
        - A 3D NumPy array of shape (H, W, 3) representing the XYZ values of the image.
        """
        H, W = J.shape

        # Initialize empty array for XYZ values
        XYZ_image = np.zeros((H, W, 3))

        # Process each pixel in the CIECAM02 image
        for i in range(H):
            for j in range(W):
                cam02_result = colour.CAM_Specification_CIECAM02(
                    J=J[i, j],
                    C=C[i, j],
                    h=K[i, j]
                )

                XYZ_image[i, j, :] = colour.appearance.CIECAM02_to_XYZ(
                    cam02_result,
                    self.XYZ_wl,
                    self.L_A,
                    self.Y_b,
                    self.surround
                )

        return XYZ_image, J, C


# Example usage:
if __name__ == "__main__":
    # Initialize the converter with default parameters
    ciecam02_converter = CIECAM02Converter(L_A=60, Y_b=25)

    # Example XYZ image (927, 1920, 3)
    # Replace this with your actual XYZ image
    XYZ_image = np.random.rand(9, 10, 3) # Example: Random XYZ values
    print(XYZ_image)

    # Convert the XYZ image
    result = ciecam02_converter.convert_image(XYZ_image)

    # Access the results
    lightness_image = result["Lightness (J)"]
    chroma_image = result["Chroma (C)"]
    hue_image = result["Hue (H)"]

    print("Lightness (J) image shape:", lightness_image.shape)
    print("Chroma (C) image shape:", chroma_image.shape)
    print("Hue (H) image shape:", hue_image.shape)
