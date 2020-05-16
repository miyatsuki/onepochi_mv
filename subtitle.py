import datetime
import os
import json
import sys

import cv2
import numpy as np
from tqdm import trange
from PIL import ImageFont, ImageDraw, Image

import ffmpeg

command_file = sys.argv[1]
setting_file = sys.argv[2]

with open(setting_file) as f:
	setting = json.load(f)

with open(command_file) as f:
    commands = json.load(f)

background_image = cv2.imread(setting["background_image"])
(height, width, _) = background_image.shape

fps = 24
movie_sec = 110
frame_num = fps * movie_sec

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
tempfile = "tmp.mp4"
out = cv2.VideoWriter(tempfile, fourcc, int(fps), (int(width), int(height)))

frame_command = [{}] * frame_num
for command in commands:
    start_sec = command["time"][0]
    end_sec = command["time"][1]

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    for frame in range(start_frame, end_frame):
        frame_command[frame] = {"text": command["text"]}

fontpath = setting["font"]
font = ImageFont.truetype(fontpath, 60)
position = (30, int(height * 0.91))

fontpath = setting["header_font"]
header_font = ImageFont.truetype(fontpath, 48)
header_position = (30, 30)

bgra = (255, 255, 255, 0)

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
    draw.text(
        header_position, setting["header"], font=header_font, fill=bgra
    )
    frame = np.array(img_pil)

    out.write(frame)

out.release()

output_movie_file = setting["output_file"]
if os.path.exists(output_movie_file):
    os.remove(output_movie_file)

movie_input = ffmpeg.input(tempfile)
audio_input = ffmpeg.input(setting["audio_file"])
output = ffmpeg.output(movie_input, audio_input, setting["output_file"])
print(ffmpeg.compile(output))
ffmpeg.run(output)