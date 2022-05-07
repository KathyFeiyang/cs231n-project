import cv2
import glob
import sys
import os
import numpy as np

from pathlib import Path

"""
Example usage (note the quotation marks):
`python data/generate_image.py "hmdb51_sta/cartwheel/*.avi" ~/HMDB51`
"""

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("need to input a format for avi files and a save name")
        quit()

    path_format = sys.argv[1]
    video_paths = glob.glob(path_format, recursive=True)
    print(f"#Videos: {len(video_paths)}")
    # name = Path(path_format).stem
    root = sys.argv[2]

    dir = os.path.join(root, 'dataset/test/')

    if not os.path.exists(dir):
        os.makedirs(dir)

    for video_path in video_paths:
        video_dir = os.path.join(dir, Path(video_path.split(".")[-2]).stem)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        vidcap = cv2.VideoCapture(video_path)
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        replen = len(str(num_frames))
        # print(num_frames)
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(video_dir, str(count).rjust(replen, '0') + ".png"), image)
            success,image = vidcap.read()
            count += 1
        print("Frames written:", count)
