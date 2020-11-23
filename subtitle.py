import datetime
import json
import math
import pathlib
import shutil
import sys
import tempfile
import copy
from itertools import product

import cv2
import ffmpeg
import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

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

fps = setting["fps"]
y, sr = librosa.load(str(setting["audio_file"]))
movie_sec = math.ceil(librosa.get_duration(y, sr))
frame_num = fps * movie_sec

with open(command_file) as f:
    commands = json.load(f)

background_path = str(setting["background_image"])
background_image = cv2.imread(background_path)
(height, width, _) = background_image.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# deepcopyしておかないとリスト内のdictが全部同じIDになって死ぬ
frame_commands = [copy.deepcopy({}) for _ in range(frame_num)]

for command in commands:
    start_sec = command["time"][0]
    end_sec = command["time"][1]

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    for frame in range(start_frame, end_frame):
        frame_command = frame_commands[frame]
        if "text" in command:
            frame_command["text"] = command["text"]
        elif "color-transition" in command:
            # 連続変化はフレーム単位では単純な差し替えなので差し替えっぽい命令に書き換える
            # color-changeコマンドは今のところ使ってないけど。
            frame_command["color-change"] = {}
            frame_command["color-change"]["range"] = command["color-transition"][
                "range"
            ]
            frame_command["color-change"]["from-color"] = command["color-transition"][
                "from-color"
            ]
            frame_command["color-change"]["to-color"] = [0, 0, 0]
            for rgb in range(3):  # rgb
                base_color = command["color-transition"]["from-color"][rgb]
                target_color = command["color-transition"]["to-color"][rgb]
                color_coef = (target_color - base_color) / (end_frame - start_frame)
                to_color = color_coef * (frame - start_frame) + base_color
                frame_command["color-change"]["to-color"][rgb] = to_color
        elif "fadeout" in command:
            frame_command["alpha-blend"] = {}
            frame_command["alpha-blend"]["to-color"] = command["fadeout"]
            alpha_coef = -1 / (end_frame - start_frame)
            elapsed_frame = frame - start_frame
            frame_command["alpha-blend"]["alpha"] = 1 + alpha_coef * elapsed_frame
        else:
            raise ValueError

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

    is_first = True
    for command in tqdm(frame_commands):
        frame = np.copy(background_image)

        cv2.rectangle(
            frame, (0, 0), (width, int(height * 0.075) + 10), (0, 0, 0), thickness=-1
        )
        cv2.rectangle(
            frame, (0, int(height * 0.9) - 10), (width, height), (0, 0, 0), thickness=-1
        )

        if "text" in command:
            text = command["text"]
        else:
            text = ""

        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=bgra)

        # ヘッダー
        draw.text(header_position, setting["header"], font=header_font, fill=bgra)
        frame = np.array(img_pil)

        if "color-change" in command:
            lu = command["color-change"]["range"][0]  # 左上
            rd = command["color-change"]["range"][1]  # 右下
            base_rgb = np.array(command["color-change"]["from-color"])
            to_bgr = np.flip(np.array(command["color-change"]["to-color"]))
            for y in range(lu[1], rd[1] + 1):
                for x in range(lu[0], rd[0] + 1):
                    if (frame[y, x] == base_rgb).all():  # 指定の色と一致してたら色を差し替える
                        frame[y, x] = to_bgr  # rgbじゃなくてbgrで格納されてるので
        if "alpha-blend" in command:
            rd = [frame.shape[0], frame.shape[1]]  # 右下(rangeなので-1しなくていい)
            alpha = command["alpha-blend"]["alpha"]
            to_bgr = np.flip(np.array(command["alpha-blend"]["to-color"]))
            frame = frame * alpha + to_bgr * (1 - alpha)
            frame = frame.astype("uint8")

        out.write(frame)

        # サムネ用画像
        if is_first:
            cv2.imwrite(str(tmp_dir_path / "Thumbnail.jpg"), frame)
            is_first = False

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
