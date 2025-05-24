#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageFont
import cv2
from commands import TextCommand
import os

def test_right_alignment_issue():
    """
    Test to reproduce and validate the fix for the issue where multi-line text with "rt" anchor
    is not properly aligned per line.
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
    
    # Test case from the issue description
    test_text = "あいうえお\nかき"  # Japanese text example from the issue
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Test with our fixed implementation
    text_command = TextCommand(
        type="text",
        time=(0, 1),
        position=(0.95, 0.3),
        text=test_text,
        font=font,
        bgra=(0, 0, 0, 255),  # Black text
        stroke_width=0,
        stroke_fill=None,
        anchor="rt"
    )
    
    # Apply the text command
    result_frame = text_command.draw(frame, width, height)
    
    # Save the result
    img = Image.fromarray(result_frame)
    output_path = os.path.join(output_dir, "fixed_rt_alignment.png")
    img.save(output_path)
    print(f"Fixed implementation result saved to: {output_path}")
    
    # Add visual guides to the image to show alignment
    guide_frame = result_frame.copy()
    # Draw a vertical red line at the right anchor position
    anchor_x = int(0.95 * width)
    cv2.line(guide_frame, (anchor_x, 0), (anchor_x, height), (0, 0, 255), 1)
    
    # Save the guide image
    guide_img = Image.fromarray(guide_frame)
    guide_path = os.path.join(output_dir, "fixed_rt_alignment_with_guide.png")
    guide_img.save(guide_path)
    print(f"Guide image saved to: {guide_path}")
    
    print("Test completed. Please check the output images to verify alignment.")

if __name__ == "__main__":
    test_right_alignment_issue()