from __future__ import annotations
import time

import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import pybullet as p
import pybullet_data


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


if __name__ == '__main__':
    # 物理シミュレータに接続して床を配置
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    plane_id = p.loadURDF('plane.urdf')

    # 位置と角度を指定してSciurus17を召喚
    init_pos = [0, 0, 0]
    init_orientation = p.getQuaternionFromEuler([0, 0, 3.14])
    sciurus_id = p.loadURDF('data/sciurus17/sciurus17.urdf',
                            init_pos, init_orientation)

    # 300フレームシミュレーションを行う
    force = 200
    frames = []
    for t in range(600):
        p.stepSimulation()
        if t % 10 == 0:
            p.setJointMotorControl2(sciurus_id, 14, p.TORQUE_CONTROL, force=force)  # r_hand_mimic
            p.setJointMotorControl2(sciurus_id, 24, p.TORQUE_CONTROL, force=force)  # l_hand_mimic
            # 10フレーム毎の計30フレーム(1s)シミュレーション結果を画像として取得
            width, height, img_rgb, img_depth, img_seg = p.getCameraImage(360, 240)
            frames.append(img_rgb)
    save_video(frames, 'result/sample.mp4')  # 結果を動画で保存

    # # ジョイントの情報を出力する
    # for i in range(p.getNumJoints(sciurus_id)):
    #     info = p.getJointInfo(sciurus_id, i)
    #     print(i, info[1].decode('utf-8'), info[10])

    p.disconnect()  # 物理シミュレーションから切断
