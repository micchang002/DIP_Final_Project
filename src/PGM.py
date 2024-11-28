import numpy as np
import cv2
import os
from DCM import ColorSpaceConverter


class PostGamutMapping:
    def __init__(self):
        # Define the gamma values
        self.gamma_r = 2.2212
        self.gamma_g = 2.1044
        self.gamma_b = 2.1835
        # Define the linear transformation matrix for converting XYZ to RGB
        self.M_l = np.array([
            [4.61, 3.35, 1.78],
            [2.48, 7.16, 0.79],
            [0.28, 1.93, 8.93]
        ])
        self.M_l_inv = np.linalg.inv(self.M_l)

    def step1_xyz_to_rgb(self, xyz_image):
        """
        Step 1: Convert XYZ to RGB values using the linear transformation matrix and apply gamma correction.
        Input: xyz_image shape (height, width, 3)
        """
        # Reshape to 2D array of pixels
        height, width, _ = xyz_image.shape
        xyz_pixels = xyz_image.reshape(-1, 3)
        # Convert XYZ to RGB (linear transformation)
        rgb_linear = np.dot(xyz_pixels, self.M_l_inv.T)
        # print(f'rgb_linear: {rgb_linear.shape}')
        # Apply gamma correction
        rgb_corrected = np.zeros_like(rgb_linear)
        rgb_corrected[:, 0] = rgb_linear[:, 0] ** (1 / self.gamma_r)  # R
        rgb_corrected[:, 1] = rgb_linear[:, 1] ** (1 / self.gamma_g)  # G
        rgb_corrected[:, 2] = rgb_linear[:, 2] ** (1 / self.gamma_b)  # B
        rgb_corrected = np.nan_to_num(rgb_corrected)
        # print(f'rgb_corrected: {rgb_corrected}')
        # Reshape back to image dimensions
        return rgb_corrected.reshape(height, width, 3)

    def step2_clip_rgb(self, rgb_image):
        """
        Step 2: Clip the RGB values to a hard threshold (0 to 1).
        """
        # print(np.clip(rgb_image, 0, 1))
        return np.clip(rgb_image, 0, 1)

    def step3_blend(self, original_rgb, clipped_rgb, lightness, chroma):
        """
        Step 3: Blend the clipped pixel values with the original pixel values.
        - lightness: Scalar or array of same shape as image (height, width)
        - chroma: Scalar or array of same shape as image    (height, width)
        # for each pixel:
        JC = lightness * chroma for each pixel
        (1-JC) * clipped_rgb + JC * original_rgb
        """
        blended_rgb = np.zeros_like(original_rgb)
        for i in range(original_rgb.shape[0]):
            for j in range(original_rgb.shape[1]):
                JC = lightness[i, j] * chroma[i, j] / 10000
                # JC = np.clip(JC, 0, 1)
                # print(JC)
                blended_rgb[i, j] = (1 - JC) * clipped_rgb[i, j] + JC * original_rgb[i, j]
                # blended_rgb[i, j] = np.clip(clipped_rgb[i, j], 0, 1)
        # calculate the difference between the clipped and blended image
        diff = np.sum(np.abs(clipped_rgb - blended_rgb))
        print(f'diff: {diff}')
        return blended_rgb

    def process_image(self, org_rgb, xyz_image, lightness, chroma):
        """
        Process the entire image.
        Parameters:
        - xyz_image: Input image in XYZ color space (height, width, 3)
        - lightness: Scalar or array of same shape as image (height, width)
        - chroma: Scalar or array of same shape as image    (height, width)
        """
        # Step 1: Convert XYZ to RGB
        original_rgb = self.step1_xyz_to_rgb(xyz_image)
        # Step 2: Clip the RGB values
        clipped_rgb = self.step2_clip_rgb(original_rgb)
        # Step 3: Blend the clipped values with the original values
        blended_rgb = self.step3_blend(org_rgb, clipped_rgb, lightness, chroma)

        return blended_rgb, clipped_rgb


def main():
    # Create converter instance
    converter = ColorSpaceConverter()

    # Create post-gamut mapping instance
    pgm = PostGamutMapping()

    # Convert image
    image_path = os.path.join("..", "images", "_big.png")
    try:
        # Convert to XYZ
        xyz_image = converter.convert_image_to_xyz(image_path)
        print("XYZ conversion successful!")
        print("XYZ image shape:", xyz_image.shape)

        # Apply post-gamut mapping
        processed_rgb = pgm.process_image(xyz_image)
        print("Post-gamut mapping successful!")
        print("Processed RGB shape:", processed_rgb.shape)

        # Optional: Save the processed image
        processed_bgr = cv2.cvtColor((processed_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_path = os.path.join("..", "images", "processed_output.png")
        cv2.imwrite(output_path, processed_bgr)
        print(f"Processed image saved to {output_path}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()