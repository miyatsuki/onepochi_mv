#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageFont
import cv2
from commands import TextCommand
import os

def test_right_alignment():
    # Create a blank white image
    width, height = 800, 600
    frame = np.ones((height, width, 3), dtype="uint8") * 255
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to find a font to use for testing
    font_path = None
    font_directories = [
        "/usr/share/fonts/truetype/freefont",  # Common Linux location
        "/usr/share/fonts/TTF",                # Another Linux location
        "/Library/Fonts",                      # macOS
        "C:\\Windows\\Fonts"                   # Windows
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
        # Use a basic font that should be available
        try:
            # PIL may have a default font we can use
            font = ImageFont.load_default()
        except:
            print("Error: Cannot load default font. Test cannot continue.")
            return
    else:
        font = ImageFont.truetype(font_path, 40)
    
    # Multi-line text for testing
    test_text = "First line\nSecond longer line\nThird"
    
    # Test with right alignment
    text_command_rt = TextCommand(
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
    result_frame = text_command_rt.draw(frame, width, height)
    
    # Save the result
    img = Image.fromarray(result_frame)
    output_path = os.path.join(current_dir, "test_rt_alignment.png")
    img.save(output_path)
    print(f"Test image saved to: {output_path}")
    
    # For comparison, also test with left and middle alignment
    text_command_lt = TextCommand(
        type="text",
        time=(0, 1),
        position=(0.05, 0.5),
        text=test_text,
        font=font,
        bgra=(0, 0, 0, 255),
        stroke_width=0,
        stroke_fill=None,
        anchor="lt"
    )
    
    text_command_mt = TextCommand(
        type="text",
        time=(0, 1),
        position=(0.5, 0.7),
        text=test_text,
        font=font,
        bgra=(0, 0, 0, 255),
        stroke_width=0,
        stroke_fill=None,
        anchor="mt"
    )
    
    # Apply the other text commands
    result_frame = text_command_lt.draw(result_frame, width, height)
    result_frame = text_command_mt.draw(result_frame, width, height)
    
    # Save the composite result
    img = Image.fromarray(result_frame)
    output_path = os.path.join(current_dir, "test_all_alignments.png")
    img.save(output_path)
    print(f"Composite test image saved to: {output_path}")
    
    print("Test completed. Please check the output images to verify alignment.")

if __name__ == "__main__":
    test_right_alignment()