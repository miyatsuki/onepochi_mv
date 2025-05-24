from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Command:
    type: str
    time: Tuple[float, float]
    position: Tuple[float, float]


@dataclass(frozen=True)
class ImageCommand(Command):
    path: Path

    def draw(
        self, frame: np.ndarray, width: int, height: int, image_cache
    ) -> np.ndarray:
        x = int(self.position[0] * width)
        y = int(self.position[1] * height)
        img_frame = Image.fromarray(frame)
        img_overlay = Image.fromarray(image_cache[self.path])

        # 透明度有りの画像を貼る場合はパラメータを変える
        # 参考: https://stackoverflow.com/questions/5324647/how-to-merge-a-transparent-png-image-with-another-image-using-pil
        img_frame.paste(img_overlay, (x, y), img_overlay)
        return np.array(img_frame)


@dataclass(frozen=True)
class TextCommand(Command):
    text: str
    font: ImageFont.ImageFont
    position: Tuple[float, float]
    bgra: Tuple[int, int, int, int]
    stroke_width: int
    stroke_fill: Optional[Tuple[int, int, int, int]]
    anchor: Optional[Literal["lt", "mt", "rt"]]

    def draw(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        x = int(self.position[0] * width)
        y = int(self.position[1] * height)

        if self.anchor and "\n" in self.text:
            # PILのdraw.text関数は改行を含むテキストに対してanchorパラメータをサポートしていないため、手動で位置調整を行う必要がある
            lines = self.text.split("\n")
            bboxes = [self.font.getbbox(line) for line in lines]
            line_heights = [bbox[3] - bbox[1] for bbox in bboxes]
            
            # Calculate average line height for proper line spacing
            avg_line_height = sum(line_heights) / len(line_heights) if line_heights else 0
            
            # Draw each line separately with its own alignment
            current_y = y
            for i, line in enumerate(lines):
                current_x = x
                if not line:  # Skip empty lines but advance y position
                    current_y += int(avg_line_height)
                    continue
                    
                bbox = self.font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                
                # Adjust x position based on anchor for each line individually
                if "m" in self.anchor:  # middle horizontally
                    current_x = x - line_width // 2
                elif "r" in self.anchor:  # right
                    current_x = x - line_width
                
                # Draw the individual line
                draw.text(
                    (current_x, current_y),
                    line,
                    font=self.font,
                    fill=tuple(self.bgra),
                    stroke_width=self.stroke_width,
                    stroke_fill=tuple(self.stroke_fill) if self.stroke_fill else None,
                )
                
                # Move to the next line position
                current_y += int(avg_line_height)
        else:
            # For single line text or when no anchor is specified, use the original approach
            if self.anchor:
                bboxes = [self.font.getbbox(self.text)]
                line_widths = [bbox[2] - bbox[0] for bbox in bboxes]
                text_width = max(line_widths)

                # Adjust x position based on anchor
                if "m" in self.anchor:  # middle horizontally
                    x = x - text_width // 2
                elif "r" in self.anchor:  # right
                    x = x - text_width

            # Draw the text at the calculated position
            draw.text(
                (x, y),
                self.text,
                font=self.font,
                fill=tuple(self.bgra),
                stroke_width=self.stroke_width,
                stroke_fill=tuple(self.stroke_fill) if self.stroke_fill else None,
            )
            
        frame = np.array(img_pil)
        return frame


@dataclass(frozen=True)
class ScreenshotCommand(Command):
    path_str: str

    def action(self, frame: np.ndarray, output_dir: Path) -> None:
        cv2.imwrite(str(output_dir / self.path_str), frame)
