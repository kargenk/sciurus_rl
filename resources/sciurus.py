from __future__ import annotations
import math
import os
from typing import Tuple

import pybullet as p
import pybullet_data


class Sciurus:
    def __init__(self, client, urdf_rootpath=pybullet_data.getDataPath()) -> None:
        self.client = client
        self.urdf_rootpath = urdf_rootpath
        self.max_force = 4.0
        self.max_velocity = 5.969211
        self.r_end_effector_id = 13  # r_hand_joint
        self.r_gripper_id = 14  # r_hand_mimic_joint

        # # 自由に動かせるジョイントのID
        # self.free_joints = [0, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14,
        #                     15, 16, 17, 18, 19, 21, 22, 23, 24]

        # エンドエフェクタとハンドの初期位置と角度
        self.r_end_effector_pos = [0.537, 0.0, 0.5]
        self.r_end_effector_angle = 0

        self.reset()

    def reset(self) -> None:
        """
        環境の初期化を行う．
        """
        # トレイの読み込み
        self.tray_id = p.loadURDF(os.path.join(self.urdf_rootpath, 'tray/tray.urdf'),
                                  basePosition=[0.640000, 0.075000, -0.190000],
                                  baseOrientation=[0.000000, 0.000000, 1.000000, 0.000000],
                                  physicsClientId=self.client)

        # Sciurusの読み込み
        file_path = os.path.join(os.path.dirname(__file__), 'sciurus17', 'sciurus17.urdf')
        orn = p.getQuaternionFromEuler([0, 0, math.pi])
        self.sciurus_id = p.loadURDF(file_path, basePosition=[1, 0, -0.2], baseOrientation=orn,
                                     physicsClientId=self.client)
        self.num_joints = p.getNumJoints(self.sciurus_id, physicsClientId=self.client)

        # 質量を0にして倒れないようにする
        for l_id in range(self.num_joints):
            p.changeDynamics(self.sciurus_id, l_id, mass=0.0001)

        # 初期ポーズを取らせる
        # Sciurusの両腕のjoint IDと初期位置(right: 7 + left: 7)
        # 13と23はハンドの垂直方向の角度, 14と24はハンドの把持具合
        # ref: github.com/rt-net/sciurus17_ros/sciurus17_moveit_config/config/sciurus17.srdf *_arm_waist_init_pose
        self.arm_joints = [5, 6, 7, 8, 9, 11, 12, 13,
                           15, 16, 17, 18, 19, 21, 22, 23]
        self.joint_positions = [-0.3574, -1.5707, 0, 2.7262, 0, -1.1155, 0, 0,
                                0.3574, 1.5707, 0, -2.7262, 0, 1.1155, 0, 0]
        for i, j_id in enumerate(self.arm_joints):
            p.resetJointState(self.sciurus_id, j_id, self.joint_positions[i], physicsClientId=self.client)
            p.setJointMotorControl2(self.sciurus_id, j_id,
                                    p.POSITION_CONTROL,
                                    targetPosition=self.joint_positions[i],
                                    force=self.max_force,
                                    physicsClientId=self.client)
        # シミュレーションを実行して初期ポーズを反映させる
        for _ in range(10):
            p.stepSimulation()

        # 自由に動かせるジョイントの名前とIDを取得
        self.motor_names = []
        self.motor_ids = []
        for j_id in range(self.num_joints):
            joint_info = p.getJointInfo(self.sciurus_id, j_id, physicsClientId=self.client)
            q_index = joint_info[3]
            if q_index >= -1:
                self.motor_names.append(str(joint_info[1]))
                self.motor_ids.append(j_id)

    def get_ids(self) -> Tuple[int, int]:
        """
        シミュレーション環境とsciurusのオブジェクトIDを返す．

        Returns:
            Tuple[int, int]: pybullet環境とsciurusのオブジェクトID
        """
        return self.client, self.sciurus_id

    def get_observation(self) -> list[list]:
        """
        把持を行う右手のグリッパーの座標と角度(オイラー角)を観測として返す．

        Returns:
            list[list]: 右手のグリッパーの座標(x, y, z)とオイラー角(roll, pitch, yaw)からなる6次元のベクトル
        """
        observation = []
        state = p.getLinkState(self.sciurus_id, self.r_gripper_id, physicsClientId=self.client)
        pos = state[0]
        orn = state[1]
        orn_euler = p.getEulerFromQuaternion(orn, physicsClientId=self.client)  # 四元数からオイラー角に変換

        observation.extend(list(pos))
        observation.extend(list(orn_euler))

        return observation

    def apply_action(self, actions: list[float]) -> None:
        """
        x, y, z座標とyaw, ハンドの角度を行動として受け取り，動かす．

        Args:
            actions (list[float]): x, y, z座標とyaw, ハンドの角度を要素とするリスト
        """
        for i, action in enumerate(actions):
            motor = self.arm_joints[i]
            print(f'i: {i}, action: {action}, motor: {motor}')
            p.setJointMotorControl2(self.sciurus_id, motor, p.POSITION_CONTROL,
                                    targetPosition=action,
                                    force=self.max_force,
                                    physicsClientId=self.client)


if __name__ == '__main__':
    client = p.connect(p.GUI)
    sciurus = Sciurus(client)
    for _ in range(100):
        sciurus.apply_action([-0.3574, -1.5707, 0, 2.7262, 0, -1.1155, 0, 0])  # 初期ポーズ
        for _ in range(80):
            p.stepSimulation()
