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
    anchor: Literal["lt", "mt"]

    def draw(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)

        x = int(self.position[0] * width)
        y = int(self.position[1] * height)

        # Handle anchor positioning for multiline text
        # PIL doesn't support the anchor parameter for multiline text,
        # so we need to manually calculate the position adjustments
        if "\n" in self.text and self.anchor:
            lines = self.text.split("\n")
            # Calculate width of each line
            line_widths = [self.font.getbbox(line)[2] - self.font.getbbox(line)[0] for line in lines]
            max_width = max(line_widths)
            
            # Adjust x position based on anchor
            if "m" in self.anchor:  # middle horizontally
                x = x - max_width // 2
            elif "r" in self.anchor:  # right
                x = x - max_width
            
            # For multiline, we keep y as is since it represents the top position
            draw.text(
                (x, y),
                self.text,
                font=self.font,
                fill=tuple(self.bgra),
                stroke_width=self.stroke_width,
                stroke_fill=tuple(self.stroke_fill) if self.stroke_fill else None,
            )
        else:
            # Single line text or no anchor specified - use PIL's built-in anchor
            draw.text(
                (x, y),
                self.text,
                font=self.font,
                fill=tuple(self.bgra),
                stroke_width=self.stroke_width,
                stroke_fill=tuple(self.stroke_fill) if self.stroke_fill else None,
                anchor=self.anchor,
            )
        frame = np.array(img_pil)
        return frame


@dataclass(frozen=True)
class ScreenshotCommand(Command):
    path_str: str

    def action(self, frame: np.ndarray, output_dir: Path) -> None:
        cv2.imwrite(str(output_dir / self.path_str), frame)
