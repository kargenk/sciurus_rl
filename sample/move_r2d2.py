from __future__ import annotations

import pybullet as p
import pybullet_data

from utils import save_video


def joint_motor_control(obj_id: int, joint_num: int, target_vel: float) -> None:
    """
    ジョイントのモーターを制御する関数．

    Args:
        obj_id (int): オブジェクトのID
        joint_num (int): ジョイントのID
        target_vel (float): ジョイントの希望測度
    """
    max_force = 500  # 目標値に到達するために使用されるモーターの最大力
    mode = p.VELOCITY_CONTROL
    p.setJointMotorControl2(obj_id, joint_num, controlMode=mode,
                            targetVelocity=target_vel, force=max_force)


if __name__ == '__main__':
    # 物理シミュレータに接続して床を配置
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    plane_id = p.loadURDF('plane.urdf')

    # 位置と角度を指定してR2D2を召喚
    init_pos = [0, 0, 1]
    init_orientation = p.getQuaternionFromEuler([0, 0, 3.14])
    r2d2_id = p.loadURDF('r2d2.urdf', init_pos, init_orientation)

    # # ジョイントの情報を出力する
    # for i in range(p.getNumJoints(r2d2_id)):
    #     info = p.getJointInfo(r2d2_id, i)
    #     print(i, info[1].decode('utf-8'), info[10])

    # シミュレーションして動画を保存
    frames = []
    for t in range(300):
        joint_motor_control(r2d2_id, 2, target_vel=10000)  # 2: right_front_wheel_jointにトルクをかける
        p.stepSimulation()
        if t % 10 == 0:
            width, height, img_rgb, img_depth, img_seg = p.getCameraImage(360, 240)  # 最新の3.1.0ではタプルで返すので注意s
            frames.append(img_rgb)
    save_video(frames, 'result/r2d2_summon.mp4')

    p.disconnect()  # 物理シミュレーションから切断
