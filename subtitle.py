import datetime
import os
import json
import sys
import pathlib
import tempfile
import datetime
import shutil

import cv2
import numpy as np
from tqdm import trange
from PIL import ImageFont, ImageDraw, Image

import ffmpeg

fps = 24
movie_sec = 110
frame_num = fps * movie_sec

title_dir = pathlib.Path(sys.argv[1])
src_dir = pathlib.Path(__file__).parent.resolve()
materials_dir = title_dir / "materials"
command_file = materials_dir / "commands.json"
setting_file = materials_dir / "settings.json"


def resolve_path(path_string):
    p = pathlib.Path(path_string)
    if p.is_absolute():
        return p.resolve()
    else:
        return materials_dir / p


with open(setting_file) as f:
    setting = json.load(f)

setting["background_image"] = resolve_path(setting["background_image"])
setting["header_font"] = resolve_path(setting["header_font"])
setting["font"] = resolve_path(setting["font"])
setting["audio_file"] = resolve_path(setting["audio_file"])

with open(command_file) as f:
    commands = json.load(f)

background_path = str(setting["background_image"])
background_image = cv2.imread(background_path)
(height, width, _) = background_image.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

frame_command = [{}] * frame_num
for command in commands:
    start_sec = command["time"][0]
    end_sec = command["time"][1]

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    for frame in range(start_frame, end_frame):
        frame_command[frame] = {"text": command["text"]}

fontpath = str(setting["font"])
font = ImageFont.truetype(fontpath, 60)
position = (30, int(height * 0.91))

fontpath = str(setting["header_font"])
header_font = ImageFont.truetype(fontpath, 48)
header_position = (30, 30)
bgra = (255, 255, 255, 0)

with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_dir_path = pathlib.Path(tmp_dir)
    tempfile = tmp_dir_path / "tmp.mp4"
    out = cv2.VideoWriter(str(tempfile), fourcc, int(fps), (int(width), int(height)))

    for i in trange(frame_num):
        frame = np.copy(background_image)

        cv2.rectangle(
            frame, (0, 0), (width, int(height * 0.075) + 10), (0, 0, 0), thickness=-1
        )
        cv2.rectangle(
            frame, (0, int(height * 0.9) - 10), (width, height), (0, 0, 0), thickness=-1
        )

        if "text" in frame_command[i]:
            text = frame_command[i]["text"]
        else:
            text = ""

        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=bgra)

        # ヘッダー
        draw.text(header_position, setting["header"], font=header_font, fill=bgra)
        frame = np.array(img_pil)

        out.write(frame)

    out.release()

    output_movie_file = str(tmp_dir_path / setting["output_file"])
    movie_input = ffmpeg.input(tempfile)
    audio_input = ffmpeg.input(setting["audio_file"])
    output = ffmpeg.output(movie_input, audio_input, output_movie_file)
    print(ffmpeg.compile(output))
    ffmpeg.run(output)

    yyyymmddhhmmss = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_dir = title_dir / f"archive/{yyyymmddhhmmss}/"
    shutil.copytree(tmp_dir_path, result_dir)

shutil.copytree(materials_dir, result_dir / "materials")
shutil.copytree(src_dir, result_dir / "src")
