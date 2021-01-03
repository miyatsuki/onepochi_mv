import copy
import datetime
import json
import pathlib
import shutil
import sys
import tempfile
from typing import NamedTuple, Optional, Tuple

import audioread
import cv2
import ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

title_dir = pathlib.Path(sys.argv[1])
src_dir = pathlib.Path(__file__).parent.resolve()
materials_dir = title_dir / "materials"
command_file = materials_dir / "commands.json"
setting_file = materials_dir / "settings.json"


class Header(NamedTuple):
    text: str
    font: ImageFont.ImageFont
    position: Tuple[int, int]


class Subtitle(NamedTuple):
    font: ImageFont.ImageFont
    position: Tuple[int, int]


class Setting(NamedTuple):
    header: Optional[Header]
    subtitle: Subtitle
    fps: int
    frame_num: int
    duration: float
    background_image: Optional[np.array]
    width: int
    height: int
    output_file: pathlib.Path
    audio_file: pathlib.Path


def resolve_path(path_string):
    p = pathlib.Path(path_string)
    if p.is_absolute():
        return p.resolve()
    else:
        return materials_dir / p


def load_image(path_string: str) -> np.array:
    assert resolve_path(path_string).exists(), f"{path_string}は存在しません"
    path = str(resolve_path(path_string))
    image = cv2.imread(path)
    return image


def load_setting(setting_file: str) -> Setting:
    with open(setting_file) as f:
        setting = json.load(f)

    setting["font"] = resolve_path(setting["font"])
    setting["audio_file"] = resolve_path(setting["audio_file"])
    fps = setting["fps"]
    with audioread.audio_open(setting["audio_file"]) as f:
        duration = f.duration

    frame_num = int(fps * duration)

    background_image = None
    if "background_image" in setting:
        background_image = load_image(setting["background_image"])
        (height, width, _) = background_image.shape
    else:
        width = setting["width"]
        height = setting["height"]

    # subtitle_setting
    fontpath = str(setting["font"])
    font = ImageFont.truetype(fontpath, 60)
    position = (30, int(height * 0.91))
    subtitle = Subtitle(font, position)

    # header_setting
    header = None
    if "header" in setting:
        setting["header_font"] = resolve_path(setting["header_font"])
        fontpath = str(setting["header_font"])
        header_font = ImageFont.truetype(fontpath, 48)
        header_position = (30, 30)
        header = Header(setting["header"], header_font, header_position)

    return Setting(
        header=header,
        subtitle=subtitle,
        fps=fps,
        frame_num=frame_num,
        duration=duration,
        background_image=background_image,
        width=width,
        height=height,
        output_file=setting["output_file"],
        audio_file=setting["audio_file"],
    )


setting = load_setting(setting_file)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
with open(command_file) as f:
    commands = json.load(f)

# deepcopyしておかないとリスト内のdictが全部同じIDになって死ぬ
frame_commands = [copy.deepcopy({}) for _ in range(setting.frame_num)]

for command in commands:
    start_sec = command["time"][0]
    if command["time"][1] != "end":
        end_sec = command["time"][1]
    else:
        end_sec = setting.duration

    start_frame = int(start_sec * setting.fps)
    end_frame = int(end_sec * setting.fps)

    for frame in range(start_frame, end_frame):
        frame_command = frame_commands[frame]
        if "text" in command:
            frame_command["text"] = command["text"]

        if "color-transition" in command:
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

        if "fadeout" in command:
            frame_command["alpha-blend"] = {}
            frame_command["alpha-blend"]["to-color"] = command["fadeout"]
            alpha_coef = -1 / (end_frame - start_frame)
            elapsed_frame = frame - start_frame
            frame_command["alpha-blend"]["alpha"] = 1 + alpha_coef * elapsed_frame

        if "background-image" in command:
            frame_command["background_image"] = command["background-image"]

bgra = (255, 255, 255, 0)
with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_dir_path = pathlib.Path(tmp_dir)
    tempfile = tmp_dir_path / "tmp.mp4"
    out = cv2.VideoWriter(
        str(tempfile),
        fourcc,
        int(setting.fps),
        (int(setting.width), int(setting.height)),
    )

    is_first = True
    for command in tqdm(frame_commands):

        if "background_image" in command:
            frame = np.copy(load_image(command["background_image"]))
        elif setting.background_image is not None:
            frame = np.copy(setting.background_image)
        else:
            # 真っ白で初期化
            frame = np.ones((1080, 1920, 3), dtype="uint8") * 255

        # サムネ用画像
        if is_first:
            cv2.imwrite(str(tmp_dir_path / "Thumbnail.jpg"), frame)
            is_first = False

        cv2.rectangle(
            frame,
            (0, int(setting.height * 0.9) - 10),
            (setting.width, setting.height),
            (0, 0, 0),
            thickness=-1,
        )

        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text = command["text"] if "text" in command else ""
        draw.text(
            setting.subtitle.position, text, font=setting.subtitle.font, fill=bgra
        )
        frame = np.array(img_pil)

        # ヘッダー
        if setting.header is not None:
            cv2.rectangle(
                frame,
                (0, 0),
                (setting.width, int(setting.height * 0.075) + 10),
                (0, 0, 0),
                thickness=-1,
            )

            # color=bgra
            draw.text(
                setting.header.position,
                setting.header.text,
                font=setting.header.font,
                fill=(255, 255, 255, 0),
            )

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

    out.release()

    output_movie_file = str(tmp_dir_path / setting.output_file)
    movie_input = ffmpeg.input(tempfile)
    audio_input = ffmpeg.input(setting.audio_file)
    output = ffmpeg.output(movie_input, audio_input, output_movie_file)
    print(ffmpeg.compile(output))
    ffmpeg.run(output)

    yyyymmddhhmmss = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_dir = title_dir / f"archive/{yyyymmddhhmmss}/"
    shutil.copytree(tmp_dir_path, result_dir)

shutil.copytree(materials_dir, result_dir / "materials")
