import numpy as np
import colour
import cv2
import os
from DCM import ColorSpaceConverter
from PGM import PostGamutMapping
from CAM02 import CIECAM02Converter

# Define the image path
image_path = os.path.join("..", "images", "img_1.png")

# call ColorSpaceConverter to convert the image to XYZ
converter = ColorSpaceConverter()
xyz_image = converter.convert_image_to_xyz(image_path)
print("XYZ image shape:", xyz_image.shape)
# print(xyz_image[5][6])

# call CAM02ColorSpaceConverter to convert the XYZ image to CIECAM02
ciecam02_converter = CIECAM02Converter(L_A=60, Y_b=25)
jch_value = ciecam02_converter.convert_image(xyz_image)
# print(jch_value)

# call PostGamutMapping to convert the CIECAM02 image back to RGB
ciecam02_inverted = ciecam02_converter.invert_image(jch_value["Lightness (J)"], jch_value["Chroma (C)"], jch_value["Hue (H)"])
pgm = PostGamutMapping()
processed_rgb = pgm.process_image(ciecam02_inverted)
print("Processed RGB shape:", processed_rgb.shape)

# Display the original and processed images
original_image = cv2.imread(image_path)
cv2.imshow("Original Image", original_image)
cv2.imshow("Processed Image", cv2.cvtColor((processed_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()



