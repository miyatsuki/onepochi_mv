import datetime
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

import audioread
import cv2
import ffmpeg
import numpy as np
from PIL import ImageFont
from tqdm import tqdm

from commands import Command, ImageCommand, ScreenshotCommand, TextCommand

title_dir = Path(sys.argv[1])
src_dir = Path(__file__).parent.resolve()
materials_dir = title_dir / "materials"
command_file = materials_dir / "commands.json"
setting_file = materials_dir / "settings.json"


class Setting(NamedTuple):
    fps: int
    frame_num: int
    duration: float
    default_command: Dict
    width: int
    height: int
    output_file: Path
    audio_file: Path
    sec_base: float
    sec_offset: int
    presets: Dict[str, Dict[str, Any]]


def resolve_path(path_string):
    p = Path(path_string)
    if p.is_absolute():
        return p.resolve()
    else:
        return materials_dir / p


def load_image(path_string: str) -> cv2.Mat:
    assert resolve_path(path_string).exists(), f"{path_string}は存在しません"
    path = str(resolve_path(path_string))
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if image.ndim == 3:  # RGBならアルファチャンネル追加
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    return image


def load_setting(setting_file: Path) -> Setting:
    with open(setting_file) as f:
        setting = json.load(f)

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
    elif "bpm" in setting:
        # TODO: 4/4を仮定
        sec_base = 60 / setting["bpm"] * 4
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
        presets=setting.get("presets", {}),
    )


image_cache: Dict[Path, np.ndarray] = {}
movie_cache: Dict[Any, Any] = {}


def parse_command(
    base_command: Dict[str, Any], elpased_sec: float, elapsed_ratio: float
) -> Command:

    command: Dict[str, Any] = {}
    for preset_name in base_command.get("presets", []):
        command |= setting.presets[preset_name]

    command |= base_command

    # match command:
    if command["type"] == "image":
        # case {"type": "image"}:
        image_path = resolve_path(command["path"])

        if image_path not in image_cache:
            image_cache[image_path] = np.copy(load_image(image_path))

        params = command.copy()
        params["path"] = image_path
        return ImageCommand(**params)
    elif command["type"] == "sequence-image":
        # case {"type": "sequence-image"}:
        image_dir = resolve_path(command["path"])
        elapsed_ms = elpased_sec * 1000

        seqs = sorted(int(p.stem) for p in image_dir.glob("*.png"))
        seq = [s for s in seqs if s <= elapsed_ms][-1]

        image_path = image_dir / (str(seq) + ".png")
        if image_path not in image_cache:
            image_cache[image_path] = np.copy(load_image(image_path))

        params = command.copy()
        params["path"] = image_path
        params["type"] = "image"
        return ImageCommand(**params)
    elif command["type"] == "text":
        # case {"type": "text"}:
        font_size = command.get("font-size", 60)
        return TextCommand(
            type="text",
            time=command["time"],
            position=command["position"],
            text=command["text"],
            font=ImageFont.truetype(str(resolve_path(command["font"])), font_size),
            bgra=command.get("color", [255, 255, 255, 255]),
            stroke_width=command.get("stroke-width", 0),
            stroke_fill=command.get("stroke-fill", None),
            anchor=command.get("anchor", None),
        )
    elif command["type"] == "screenshot":
        # case {"type": "screenshot"}:
        return ScreenshotCommand(
            type="screenshot",
            time=command["time"],
            path_str=command["path"],
            position=(0.0, 0.0),  # 使わないので適当に入れとく
        )
    else:
        # case _:
        raise ValueError(command)

    """
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
        ans["alpha-blend"]["alpha"] = 1 - elapsed_ratio

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
    """


setting = load_setting(setting_file)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
with open(command_file) as f:
    commands = json.load(f)

# deepcopyしておかないとリスト内のdictが全部同じIDになって死ぬ
# frame_commands = [parse_command() for command in setting.default_command for frame in trange(setting.frame_num)]
# for frame in trange(setting.frame_num):
#    frame_commands += copy.deepcopy(
#        parse_command(
#            setting.default_command, frame / setting.fps, frame / setting.frame_num
#        )
#    )
frame_commands: List[List[Command]] = [[] for _ in range(setting.frame_num)]

for command in tqdm(commands):
    start_sec: float = (
        0
        if command["time"][0] == "start"
        else (command["time"][0] - setting.sec_offset) * setting.sec_base
    )
    end_sec: float = (
        setting.duration
        if command["time"][1] == "end"
        else (command["time"][1] - setting.sec_offset) * setting.sec_base
    )

    start_frame = int(start_sec * setting.fps)
    end_frame = int(end_sec * setting.fps)

    if start_frame < end_frame:
        for frame in range(start_frame, end_frame):
            frame_commands[frame].append(
                parse_command(
                    command,
                    frame / setting.fps - start_sec,
                    (frame - start_frame) / (end_frame - start_frame),
                )
            )
    elif start_frame == end_frame:
        frame_commands[start_frame].append(
            parse_command(command, start_frame / setting.fps - start_sec, 0)
        )
    else:
        print(command)
        assert start_frame <= end_frame


bgra = (255, 255, 255, 0)
with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_dir_path = Path(tmp_dir)
    temp_file = tmp_dir_path / "tmp.mp4"
    out = cv2.VideoWriter(
        str(temp_file),
        fourcc,
        int(setting.fps),
        (int(setting.width), int(setting.height)),
    )

    for i, frame_command in tqdm(enumerate(frame_commands), total=len(frame_commands)):
        # print("frame", i, frame_command)

        # 真っ白で初期化
        frame = np.ones((setting.height, setting.width, 3), dtype="uint8") * 255

        for command in frame_command:
            # match command:
            if command.type == "image":
                # case ImageCommand():
                frame = command.draw(frame, setting.width, setting.height, image_cache)
            elif command.type == "text":
                # case TextCommand():
                frame = command.draw(frame, setting.width, setting.height)
            elif command.type == "screenshot":
                # case ScreenshotCommand():
                command.action(frame, tmp_dir_path)
            else:
                # case _:
                raise ValueError(command)

        """
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
            alpha = command["alpha-blend"]["alpha"]
            to_bgr = np.flip(np.array(command["alpha-blend"]["to-color"]))
            frame = frame * alpha + to_bgr * (1 - alpha)
            frame = frame.astype("uint8")
        """

        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        out.write(frame)

    out.release()

    output_movie_file = str(tmp_dir_path / setting.output_file)
    movie_input = ffmpeg.input(temp_file)
    audio_input = ffmpeg.input(setting.audio_file)
    output = ffmpeg.output(movie_input, audio_input, output_movie_file)
    print(ffmpeg.compile(output))
    ffmpeg.run(output)

    yyyymmddhhmmss = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_dir = title_dir / f"archive/{yyyymmddhhmmss}/"
    shutil.copytree(tmp_dir_path, result_dir)

shutil.copytree(materials_dir, result_dir / "materials")
