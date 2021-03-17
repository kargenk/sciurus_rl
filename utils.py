from __future__ import annotations

import numpy as np
import moviepy.editor as mpy


def save_video(frames: list[np.ndarray], path: str, fps: int = 30) -> None:
    """
    動画を保存する関数．

    Args:
        frames (list[np.ndarray]): 動画にする連続した画像のリスト
        path (str): 保存先のパス
        fps (int, optional): 保存する動画のfps. Defaults to 30.
    """
    clip = mpy.ImageSequenceClip(frames, fps=30)
    clip.write_videofile(path, fps)
