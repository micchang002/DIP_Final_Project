import cv2
import numpy as np
import os


class ColorSpaceConverter:
    def __init__(self):
        # Define constant parameters
        self.gamma_r = 2.4767
        self.gamma_g = 2.4286
        self.gamma_b = 2.3792

        # Define RGB-to-XYZ transformation matrix Mf
        self.M_f = np.array([
            [95.57, 64.67, 33.01],
            [49.49, 137.29, 14.76],
            [0.44, 27.21, 169.83]
        ])

    def load_image(self, image_path):
        """Load and check if image exists"""
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        return bgr_image

    def bgr_to_rgb(self, bgr_image):
        """Convert BGR to RGB and normalize"""
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        print(rgb_image)
        return rgb_image.astype(np.float32) / 255.0

    def apply_gamma_correction(self, rgb_image):
        """Apply gamma correction to RGB image"""
        rgb_linear = np.zeros_like(rgb_image)
        rgb_linear[:, :, 0] = rgb_image[:, :, 0] ** (self.gamma_r)  # R
        rgb_linear[:, :, 1] = rgb_image[:, :, 1] ** (self.gamma_g)  # G
        rgb_linear[:, :, 2] = rgb_image[:, :, 2] ** (self.gamma_b)  # B
        return rgb_linear

    def convert_to_xyz(self, rgb_linear):
        """Convert linear RGB to XYZ color space"""
        pixels = rgb_linear.reshape(-1, 3)
        xyz_pixels = np.dot(pixels, self.M_f.T)
        return xyz_pixels.reshape(rgb_linear.shape)

    def convert_image_to_xyz(self, image_path):
        """Main method to convert image from BGR to XYZ"""
        # Load image
        bgr_image = self.load_image(image_path)

        # Convert to RGB and normalize
        rgb_image = self.bgr_to_rgb(bgr_image)

        # Apply gamma correction
        rgb_linear = self.apply_gamma_correction(rgb_image)

        # Convert to XYZ
        xyz = self.convert_to_xyz(rgb_linear)
        # print(xyz)
        return xyz


# Example usage:
if __name__ == "__main__":
    # Create converter instance
    converter = ColorSpaceConverter()

    # Convert image
    image_path = os.path.join("..", "images", "_big.png")
    try:
        xyz_image = converter.convert_image_to_xyz(image_path)
        print("Conversion successful!")
        print("XYZ image shape:", xyz_image.shape)
        print("XYZ values:", xyz_image)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")