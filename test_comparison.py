#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
import os
from commands import TextCommand

def original_draw_implementation(frame, text, font, position, anchor, width, height):
    """Recreate the original implementation for comparison purposes"""
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    x = int(position[0] * width)
    y = int(position[1] * height)

    if anchor:
        # Original implementation
        lines = text.split("\n")
        bboxes = [font.getbbox(line) for line in lines]
        line_widths = [bbox[2] - bbox[0] for bbox in bboxes]
        text_width = max(line_widths)

        # Adjust x position based on anchor
        if "m" in anchor:  # middle horizontally
            x = x - text_width // 2
        elif "r" in anchor:  # right
            x = x - text_width

    # Draw the text at the calculated position
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(0, 0, 0, 255),
    )
    return np.array(img_pil)

def test_comparison():
    """
    Create a comparison between original and fixed implementation
    to demonstrate the issue and its fix.
    """
    # Create a blank white image
    width, height = 800, 600
    frame = np.ones((height, width, 3), dtype="uint8") * 255
    
    # Try to load a font
    font_path = None
    font_directories = [
        "/usr/share/fonts/truetype/freefont",
        "/usr/share/fonts/TTF",
        "/Library/Fonts", 
        "C:\\Windows\\Fonts"
    ]
    
    font_names = ["FreeSans.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    
    for directory in font_directories:
        if os.path.exists(directory):
            for font_name in font_names:
                potential_path = os.path.join(directory, font_name)
                if os.path.exists(potential_path):
                    font_path = potential_path
                    break
            if font_path:
                break
    
    if not font_path:
        print("No suitable font found for testing. Using default font.")
        try:
            font = ImageFont.load_default()
        except:
            print("Error: Cannot load default font. Test cannot continue.")
            return
    else:
        font = ImageFont.truetype(font_path, 40)
    
    # Create test text with varying line lengths
    test_text = "Long line for testing alignment\nShorter line\nTiny"
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # ------ Original implementation ------
    original_frame = frame.copy()
    original_result = original_draw_implementation(
        original_frame, 
        test_text, 
        font, 
        (0.95, 0.3), 
        "rt", 
        width, 
        height
    )
    
    # Add visual guide
    anchor_x = int(0.95 * width)
    cv2.line(original_result, (anchor_x, 0), (anchor_x, height), (0, 0, 255), 1)
    
    # Save original implementation result
    original_img = Image.fromarray(original_result)
    original_path = os.path.join(output_dir, "original_implementation.png")
    original_img.save(original_path)
    print(f"Original implementation result saved to: {original_path}")
    
    # ------ Fixed implementation ------
    fixed_frame = frame.copy()
    text_command = TextCommand(
        type="text",
        time=(0, 1),
        position=(0.95, 0.3),
        text=test_text,
        font=font,
        bgra=(0, 0, 0, 255),
        stroke_width=0,
        stroke_fill=None,
        anchor="rt"
    )
    
    fixed_result = text_command.draw(fixed_frame, width, height)
    
    # Add same visual guide
    cv2.line(fixed_result, (anchor_x, 0), (anchor_x, height), (0, 0, 255), 1)
    
    # Save fixed implementation result
    fixed_img = Image.fromarray(fixed_result)
    fixed_path = os.path.join(output_dir, "fixed_implementation.png")
    fixed_img.save(fixed_path)
    print(f"Fixed implementation result saved to: {fixed_path}")
    
    # ------ Create a side-by-side comparison ------
    comparison_img = Image.new('RGB', (width * 2, height), color='white')
    comparison_img.paste(Image.fromarray(original_result), (0, 0))
    comparison_img.paste(Image.fromarray(fixed_result), (width, 0))
    
    # Add labels
    draw = ImageDraw.Draw(comparison_img)
    if font:
        draw.text((width // 4, 20), "Original Implementation", font=font, fill=(0, 0, 0, 255))
        draw.text((width + width // 4, 20), "Fixed Implementation", font=font, fill=(0, 0, 0, 255))
    
    comparison_path = os.path.join(output_dir, "comparison.png")
    comparison_img.save(comparison_path)
    print(f"Comparison image saved to: {comparison_path}")
    
    print("Test completed. Please check the output images.")

if __name__ == "__main__":
    test_comparison()