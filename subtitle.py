import copy
import datetime
import json
import pathlib
import shutil
import sys
import tempfile
from typing import Any, Dict, Literal, NamedTuple, Optional, Tuple

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
    stroke_width: Optional[int]
    stroke_fill: Optional[Tuple[int, int, int, int]]
    anchor: Literal["left", "center", "right"]

    def draw(self, frame: np.array) -> np.array:
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text(self.position, self.text, 
            font=self.font, fill=self.bgra, 
            stroke_width=self.stroke_width,
            stroke_fill=self.stroke_fill,
            # multiline textではanchorが使えない
            anchor=self.anchor if "\n" not in self.text else None
        )
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
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if image.ndim == 3:  # RGBならアルファチャンネル追加
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    return image


def load_setting(setting_file: pathlib.Path) -> Setting:
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
        sec_base = setting["bpm"]/60
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
movie_cache: Dict[Any, Any] = {}


def parse_command(command: Dict, elpased_sec: float, elapsed_ratio: float) -> Dict:
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
        ans["alpha-blend"]["alpha"] = 1 - elapsed_ratio

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

    if "screenshot" in command:
        ans["screenshot"] = command["screenshot"]

    if "image" in command:
        if command["image"]["path"] not in image_cache:
            image_cache[command["image"]["path"]] = np.copy(
                load_image(command["image"]["path"])
            )

        if "image" not in ans:
            ans["image"] = []

        ans["image"] += command["image"]

    if "sequence-images" in command:
        elapsed_ms = elpased_sec * 1000
        if "image" not in ans:
            ans["image"] = []

        for sequence_image in command["sequence-images"]:
            seqs = sorted(int(p.stem) for p in resolve_path(sequence_image["path"]).glob("*.png"))
            seq = [s for s in seqs if s <= elapsed_ms][-1]

            image_path = sequence_image["path"] + "/" + str(seq) + ".png"
            if image_path not in image_cache:
                image_cache[image_path] = np.copy(
                    load_image(image_path)
                )
            ans["image"].append({
                "path": image_path,
                "position": sequence_image["position"]
            })


    return ans


setting = load_setting(setting_file)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
with open(command_file) as f:
    commands = json.load(f)

# deepcopyしておかないとリスト内のdictが全部同じIDになって死ぬ
frame_commands = [
    copy.deepcopy(
        parse_command(
            setting.default_command, frame / setting.fps, frame / setting.frame_num
        )
    )
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

    if start_frame < end_frame:
        for frame in trange(start_frame, end_frame):
            frame_commands[frame] |= parse_command(
                command,
                frame / setting.fps - start_sec,
                (frame - start_frame) / (end_frame - start_frame),
            )
    elif start_frame == end_frame:
        frame_commands[start_frame] |= parse_command(
            command, start_frame / setting.fps - start_sec, 0
        )
    else:
        print(command)
        assert start_frame <= end_frame


bgra = (255, 255, 255, 0)
with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_dir_path = pathlib.Path(tmp_dir)
    temp_file = tmp_dir_path / "tmp.mp4"
    out = cv2.VideoWriter(
        str(temp_file),
        fourcc,
        int(setting.fps),
        (int(setting.width), int(setting.height)),
    )

    for i, command in tqdm(enumerate(frame_commands), total=len(frame_commands)):
        print("frame", i, command)

        if "background_image" in command:
            image_file_name = command["background_image"]
            frame = np.copy(image_cache[image_file_name])
        else:
            # 真っ白で初期化
            frame = np.ones((setting.height, setting.width, 3), dtype="uint8") * 255

        if "image" in command:
            for image_command in command["image"]:
                x = int(image_command["position"][0] * setting.width)
                y = int(image_command["position"][1] * setting.height)
                img_frame = Image.fromarray(frame)
                img_overlay = Image.fromarray(image_cache[image_command["path"]])

                # 透明度有りの画像を貼る場合はパラメータを変える
                # 参考: https://stackoverflow.com/questions/5324647/how-to-merge-a-transparent-png-image-with-another-image-using-pil
                img_frame.paste(img_overlay, (x, y), img_overlay)
                
                frame = np.array(img_frame)

        if "text" in command:
            x = int(command["text"]["position"][0] * setting.width)
            y = int(command["text"]["position"][1] * setting.height)
            font_path = resolve_path(command["text"]["font"])
            font = ImageFont.truetype(str(font_path), 60)
            text = Text(
                text=command["text"]["text"],
                font=font,
                position=(x, y),
                bgra=tuple(command["text"].get("color", bgra)),
                stroke_width=command["text"].get("stroke-width", 0),
                stroke_fill = tuple(command["text"]["stroke-fill"]) if "stroke-fill" in command["text"] else None,
                anchor=command["text"].get("anchor", "lt")
            )
            frame = text.draw(frame)

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

        # サムネ用画像
        if "screenshot" in command:
            cv2.imwrite(str(tmp_dir_path / command["screenshot"]), frame)

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
