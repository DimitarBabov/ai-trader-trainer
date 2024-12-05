# Script to Stretch an Image Exponentially from Left to Right and Top to Bottom

from PIL import Image
import numpy as np

def stretch_image_exponentially(input_image_path, output_image_path):
    """Stretch an input image exponentially from left to right and top to bottom."""
    # Open the input image
    input_img = Image.open(input_image_path)
    input_width, input_height = input_img.size

    # Create a new blank image of size 128x128
    output_width, output_height = 128, 128
    output_img = Image.new('RGB', (output_width, output_height), (0, 0, 0))  # Black background

    # Exponential scaling factors from 1 to 2 for horizontal and vertical scaling
    horizontal_scaling_factors = np.linspace(1, 2, num=input_width)
    vertical_scaling_factors = np.linspace(1, 2, num=input_height)

    # Stretch each column of the input image to the output image horizontally
    for x in range(input_width):
        horizontal_scale_factor = horizontal_scaling_factors[x]
        new_width = int(horizontal_scale_factor)

        for y in range(input_height):
            vertical_scale_factor = vertical_scaling_factors[y]
            new_height = int(vertical_scale_factor)
            
            pixel = input_img.getpixel((x, y))
            new_x = int((x / input_width) * output_width)
            new_x_end = min(new_x + new_width, output_width)
            new_y = int((y / input_height) * output_height)
            new_y_end = min(new_y + new_height, output_height)

            # Draw the stretched pixels in the output image
            for i in range(new_x, new_x_end):
                for j in range(new_y, new_y_end):
                    if i < output_width and j < output_height:
                        output_img.putpixel((i, j), pixel)

    # Save the stretched image
    output_img.save(output_image_path)
    print(f"Stretched image saved at: {output_image_path}")

# Example usage
if __name__ == "__main__":
    input_image_path = "input_image.png"  # Replace with your input image path
    output_image_path = "stretched_output_image.png"  # Replace with your output image path
    stretch_image_exponentially(input_image_path, output_image_path)
