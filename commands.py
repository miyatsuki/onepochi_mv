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

        if self.anchor:
            # PILのdraw.text関数は改行を含むテキストに対してanchorパラメータをサポートしていないため、手動で位置調整を行う必要がある
            lines = self.text.split("\n")
            bboxes = [self.font.getbbox(line) for line in lines]
            line_widths = [bbox[2] - bbox[0] for bbox in bboxes]
            text_width = max(line_widths)

            # Adjust x position based on anchor
            if "m" in self.anchor:  # middle horizontally
                x = x - text_width // 2
            elif "r" in self.anchor:  # right
                x = x - text_width

            # We could also handle vertical anchoring here if needed

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
