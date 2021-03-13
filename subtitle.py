import copy
import datetime
import json
import pathlib
import shutil
import sys
import tempfile
from typing import Any, Dict, NamedTuple, Optional, Tuple

import audioread
import cv2
import ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm, trange

title_dir = pathlib.Path(sys.argv[1])
src_dir = pathlib.Path(__file__).parent.resolve()
materials_dir = title_dir / "materials"
command_file = materials_dir / "commands.json"
setting_file = materials_dir / "settings.json"


class Text(NamedTuple):
    text: str
    font: ImageFont.ImageFont
    position: Tuple[int, int]
    bgra: Tuple[int, int, int, int]

    def draw(self, frame: np.array) -> np.array:
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text(self.position, self.text, font=self.font, fill=bgra)
        frame = np.array(img_pil)
        return frame


class Setting(NamedTuple):
    fps: int
    frame_num: int
    duration: float
    default_command: Dict
    width: int
    height: int
    output_file: pathlib.Path
    audio_file: pathlib.Path
    sec_base: float
    sec_offset: int


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
    with audioread.audio_open(str(setting["audio_file"])) as f:
        duration = f.duration

    frame_num = int(fps * duration)

    default_command = {}
    if "default_command" in setting:
        default_command = setting["default_command"]

    width = setting["width"]
    height = setting["height"]

    if "sec_base" in setting:
        sec_base = setting["sec_base"]
    else:
        sec_base = 1

    if "sec_offset" in setting:
        sec_offset = setting["sec_offset"]
    else:
        sec_offset = 0

    return Setting(
        fps=fps,
        frame_num=frame_num,
        duration=duration,
        default_command=default_command,
        width=width,
        height=height,
        output_file=setting["output_file"],
        audio_file=setting["audio_file"],
        sec_base=sec_base,
        sec_offset=sec_offset,
    )


image_cache: Dict[str, Any] = {}
movie_cache = {}


def parse_command(command: Dict, elpased_sec: float) -> Dict:
    ans = {}
    if "text" in command:
        ans["text"] = command["text"]

    if "color-transition" in command:
        # 連続変化はフレーム単位では単純な差し替えなので差し替えっぽい命令に書き換える
        # color-changeコマンドは今のところ使ってないけど。
        ans["color-change"] = {}
        ans["color-change"]["range"] = command["color-transition"]["range"]
        ans["color-change"]["from-color"] = command["color-transition"]["from-color"]
        ans["color-change"]["to-color"] = [0, 0, 0]
        for rgb in range(3):  # rgb
            base_color = command["color-transition"]["from-color"][rgb]
            target_color = command["color-transition"]["to-color"][rgb]
            color_coef = (target_color - base_color) / (end_frame - start_frame)
            to_color = color_coef * (frame - start_frame) + base_color
            ans["color-change"]["to-color"][rgb] = to_color

    if "fadeout" in command:
        ans["alpha-blend"] = {}
        ans["alpha-blend"]["to-color"] = command["fadeout"]
        alpha_coef = -1 / (end_frame - start_frame)
        elapsed_frame = frame - start_frame
        ans["alpha-blend"]["alpha"] = 1 + alpha_coef * elapsed_frame

    if "background-image" in command:
        image_file_name = command["background-image"]
        if image_file_name not in image_cache:
            image_cache[image_file_name] = np.copy(
                load_image(command["background-image"])
            )

        ans["background_image"] = command["background-image"]

    if "background-movie" in command:
        movie_file_name = command["background-movie"]
        if command["background-movie"] not in movie_cache:
            movie_cache[movie_file_name] = cv2.VideoCapture(
                str(resolve_path(movie_file_name))
            )

        cap_file = movie_cache[movie_file_name]
        fps = cap_file.get(cv2.CAP_PROP_FPS)
        frame_num = int(fps * elpased_sec)

        image_file_name = command["background-movie"] + "." + str(frame_num)
        if image_file_name not in image_cache:
            cap_file.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            _, frame = cap_file.read()
            image_cache[image_file_name] = frame

        ans["background_image"] = image_file_name

    return ans


setting = load_setting(setting_file)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
with open(command_file) as f:
    commands = json.load(f)

# deepcopyしておかないとリスト内のdictが全部同じIDになって死ぬ
frame_commands = [
    copy.deepcopy(parse_command(setting.default_command, frame / setting.fps))
    for frame in trange(setting.frame_num)
]

for command in tqdm(commands):
    start_sec = (command["time"][0] - setting.sec_offset) * setting.sec_base
    if command["time"][1] != "end":
        end_sec = (command["time"][1] - setting.sec_offset) * setting.sec_base
    else:
        end_sec = setting.duration

    start_frame = int(start_sec * setting.fps)
    end_frame = int(end_sec * setting.fps)

    for frame in trange(start_frame, end_frame):
        frame_commands[frame] |= parse_command(command, frame / setting.fps - start_sec)

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
            image_file_name = command["background_image"]
            frame = np.copy(image_cache[image_file_name])
        else:
            # 真っ白で初期化
            frame = np.ones((setting.height, setting.width, 3), dtype="uint8") * 255

        if "text" in command:
            x = int(command["text"]["position"][0] * setting.width)
            y = int(command["text"]["position"][1] * setting.height)
            text = Text(
                text=command["text"]["text"],
                font=command["text"]["font"],
                position=(x, y),
                bgra=bgra,
            )
            text.draw()

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
