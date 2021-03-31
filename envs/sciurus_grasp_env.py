from __future__ import annotations
import glob
import os
import random
import sys
import time

import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from resources.sciurus import Sciurus


class SciurusGraspEnv(gym.Env):
    """
    Sciurus17で物体把持の強化学習を行うための環境．
    ref: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kuka_diverse_object_gym_env.py
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 urdf_rootpath: str = pybullet_data.getDataPath(),
                 action_repeat: int = 80,
                 is_enable_selfcollision: bool = True,
                 render: bool = False,
                 is_discrete: bool = False,
                 max_steps: int = 8,
                 dv: float = 0.06,
                 remove_height_hack: bool = False,
                 block_random: float = 0.3,
                 camera_random: float = 0,
                 width: int = 48,
                 height: int = 48,
                 num_objects: int = 5,
                 is_test: bool = False) -> None:
        """
        環境の初期化．

        Args:
            urdf_rootpath (str, optional): URDFなどのファイルを読み込むディレクトリ. Defaults to pybullet_data.getDataPath().
            action_repeat (int, optional): それぞれの行動に対してシミュレーションを実行する回数. Defaults to 80.
            is_enable_selfcollision (bool, optional): 自己衝突を有効にするかのフラグ. Defaults to True.
            render (bool, optional): PyBulletのGUIを有効にするかのフラグ. Defaults to False.
            is_discrete (bool, optional): 行動空間を離散(True)にするか連続にするか(False)のフラグ. Defaults to False.
            max_steps (int, optional): エピソードごとの行動の回数. Defaults to 8.
            dv (float, optional): 各行動の各次元における速度. Defaults to 0.06.
            remove_height_hack (bool, optional): Falseの場合はグリッパーが自動的に下に移動する"height hack"がある. Trueの場合は環境が厳しくなり，戦略(policy)が高さの変位を選択する． Defaults to False.
            block_random (float, optional): [0, 1]の値をとるブロックの乱数性. 0だと決定論的になる. Defaults to 0.3.
            camera_random (int, optional): [0, 1]の値をとるカメラ位置の乱数性. 0だと決定論的になる. Defaults to 0.
            width (int, optional): 観測である画像の横幅. Defaults to 48.
            height (int, optional): 観測である画像の縦幅. Defaults to 48.
            num_objects (int, optional): トレー内の物体の数. Defaults to 5.
            is_test (bool, optional): Trueならテストセットの物体を，Falseなら訓練セットの物体を用いるフラグ. Defaults to False.
        """

        self._urdf_rootpath = urdf_rootpath
        self._action_repeat = action_repeat
        self._is_enable_selfcollision = is_enable_selfcollision
        self._render = render
        self._is_discrete = is_discrete
        self._max_steps = max_steps
        self._dv = dv
        self._remove_height_hack = remove_height_hack
        self._block_random = block_random
        self._camera_random = camera_random
        self._width = width
        self._height = height
        self._num_objects = num_objects
        self._is_test = is_test

        self._timestep = 1. / 240
        self._observation = []
        self._env_step_counter = 0
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._p = p

        # 描画モードの設定
        if self._render:
            self.cid = p.connect(p.SHARED_MEMORY)
            if (self.cid < 0):
                self.cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            self.cid = p.connect(p.DIRECT)  # GUIで描画する場合はp.GUI，訓練する場合などはp.DIRECT
        self.seed()

        # 離散or連続の問題設定と行動空間(現在位置からの差分)，観測空間(RGB画像の画素値)
        if self._is_discrete:
            if self._remove_height_hack:
                self.action_space = spaces.Discrete(9)
            else:
                self.action_space = spaces.Discrete(7)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(9,))  # 右腕のジョイントのモーター角
            # Now, not working ...
            if self._remove_height_hack:
                self.action_space = spaces.Box(low=-1, high=1, shape=(4,))  # dx, dy, dz, da
        self.observation_space = spaces.Box(low=0, high=255, shape=(self._height, self._width, 3))

        self.viewer = None

    def _get_random_object(self, num_objects: int, is_test: bool) -> list[str]:
        """
        pybullet_data以下のrandom_urdfsディレクトリからランダムに物体を選ぶ関数.

        Args:
            num_objects (int): 物体の数
            is_test (bool): テスト(True)か訓練(False)か

        Returns:
            list[str]: URDFのファイル名を要素とするリスト
        """
        # 訓練かテストで配置する物体を変える
        if is_test:
            urdf_pattern = os.path.join(self._urdf_rootpath, 'random_urdfs/*0/*.urdf')  # 10%
        else:
            urdf_pattern = os.path.join(self._urdf_rootpath, 'random_urdfs/*[1-9]/*.urdf')  # 90%
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)

        # 指定個数だけランダムに選出
        selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
        selected_objects_filenames = []
        for object_id in selected_objects:
            selected_objects_filenames += [found_object_directories[object_id]]

        return selected_objects_filenames

    def _randomly_place_objects(self, urdf_list: list[str]) -> list[int]:
        """
        トレー内のランダムな場所に物体を配置する.

        Args:
            urdf_list (list[str]): トレー内に配置する物体のurdfファイルへのパスリスト

        Returns:
            list[int]: 配置した物体のIDを要素とするリスト
        """
        object_ids = []
        for urdf_name in urdf_list:
            # 位置と角度を決める
            pos_x = 0.4 + self._block_random * random.random()
            pos_y = self._block_random * (random.random() - 0.5)
            angle = (np.pi / 2) + (self._block_random * np.pi * random.random())
            orn = p.getQuaternionFromEuler([0, 0, angle])

            # 実際に配置
            urdf_path = os.path.join(self._urdf_rootpath, urdf_name)
            obj_id = p.loadURDF(urdf_path, [pos_x, pos_y, 0.15], [orn[0], orn[1], orn[2], orn[3]])
            object_ids.append(obj_id)
            # 物体が交差しないようにそれぞれの物体を落下させる
            for _ in range(50):
                p.stepSimulation()

        return object_ids

    def _get_observation(self) -> np.ndarray:
        """
        観測としてSciurusが得た画像を返す．

        Returns:
            np.ndarray: 観測である画像
        """
        width, height, img_rgba, img_depth, img_seg = p.getCameraImage(width=self._width,
                                                                      height=self._height,
                                                                      viewMatrix=self._view_matrix,
                                                                      projectionMatrix=self._proj_matrix)
        img_rgb = img_rgba[:, :, :3]
        return img_rgb

    def reset(self) -> np.ndarray:
        """
        各エピソードのはじめに呼ばれ初期化を行う.

        Returns:
            np.ndarray: 観測であるRGB画像
        """
        # カメラの設定
        look = [0.23, 0.2, 0.54]
        distance = 1.
        pitch = -56 + self._camera_random * np.random.uniform(-3, 3)
        yaw = 245 + self._camera_random * np.random.uniform(-3, 3)
        roll = 0
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        fov = 20. + self._camera_random * np.random.uniform(-2, 2)
        aspect = self._width / self._height
        near = 0.01
        far = 10
        self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        # エピソードごとに初期化する必要のある変数
        self._attempted_grasp = False
        self._grasp_success = False
        self._env_step = 0
        self._env_step_counter = 0
        self.terminated = 0
        self._action = [-0.3574, -1.5707, 0, 2.7262, 0, -1.1155, 0, 0, 0]  # 右腕の初期モーター角

        p.resetSimulation()  # 全オブジェクトを消去
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timestep)

        # 床を配置
        plane_pos = [0, 0, -1]
        plane_id = p.loadURDF(os.path.join(self._urdf_rootpath, 'plane.urdf'), plane_pos)

        # テーブルを配置
        table_pos = [0.5000000, 0.00000, -0.820000]
        table_orn = [0.000000, 0.000000, 0.0, 1.0]
        table_id = p.loadURDF(os.path.join(self._urdf_rootpath, 'table/table.urdf'), table_pos, table_orn)

        p.setGravity(0, 0, -10)  # 重力
        # sciurusを配置
        self.sciurus = Sciurus(self.cid, self._urdf_rootpath)
        p.stepSimulation()

        # 物体の配置
        urdf_list = self._get_random_object(self._num_objects, self._is_test)
        self._object_ids = self._randomly_place_objects(urdf_list)
        self._observation = self._get_observation()

        # time.sleep()  # for debug

        return np.array(self._observation)

    def _termination(self) -> bool:
        """
        エピソードが終了したか否かを返す．

        Returns:
            bool: 終了判定フラグ
        """
        # 把持成功か最大ステップ数を越えれば終了
        return self._grasp_success or self._env_step >= self._max_steps

    def _reward(self) -> int:
        """
        エピソードごとの報酬を計算して返す.
        物体のうち一つでも0.2以上の高さにあれば報酬として1を返す．

        Returns:
            int: 報酬
        """
        # TODO: 報酬がスパースすぎるので要検討

        reward = 0
        self._grasp_success = False

        # 物体の内一つでも0.2以上の高さにあれば報酬を与える
        for obj_id in self._object_ids:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            if pos[2] > 0.2:
                self._grasp_success = True
                reward = 1
                break

        return reward

    def step(self, action: list) -> tuple[np.ndarray, float, bool, dict]:
        """
        Sciurusに行動をさせて，観測，報酬，タスクが終了したかを得る．

        Args:
            action (list): xyz座標, vertical angle(rad)の各差分とgrasp angle(rad)からなる5次元のベクトル

        Returns:
            tuple[np.ndarray, float, bool, dict]: 順番に観測画像，報酬，タスクが終了したか，追加の情報
        """
        dv = self._dv  # ステップごとの速度

        self._env_step += 1

        # 初期位置を加味して差分を加えた位置に更新する
        for i in range(len(action)):
            self._action[i] += dv * action[i]

        # Sciurusに行動させる
        self.sciurus.apply_action(self._action)
        for _ in range(self._action_repeat):
            p.stepSimulation()
            if self._render:
                time.sleep(self._timestep)
            if self._termination():
                break

        # 現在のエンドエフェクタの位置を取得
        r_state = p.getLinkState(self.sciurus.sciurus_id, self.sciurus.r_end_effector_id)
        r_end_effector_pos = r_state[0]

        # トレーに十分近ければ(z座標が0.1以下)
        if r_end_effector_pos[2] <= 0.1:
            finger_angle = 0.9
            grasp_action = self._action
            # 把持動作を行う
            for _ in range(500):
                grasp_action[-1] = finger_angle  # グリッパーの角度
                self.sciurus.apply_action(grasp_action)
                p.stepSimulation()
                if self._render:
                    time.sleep(self._timestep)
                # グリッパーの角度を徐々に小さくして把持を試みる
                finger_angle -= 0.3 / 100
                if finger_angle <= 0:
                    finger_angle = 0
            # # 少し高めでも試す
            # for _ in range(500):
            #     grasp_action[-1] = finger_angle
            #     grasp_action[n] = 0.001  # set z to 0.001
            #     self.sciurus.apply_action(grasp_action)
            #     p.stepSimulation()
            #     if self._render:
            #         time.sleep(self._timestep)
            #     finger_angle -= 0.03 / 100
            #     if finger_angle <= 0:
            #         finger_angle = 0
            self._attempted_grasp = True

        observation = self._get_observation()  # 観測(状態)を得る
        reward = self._reward()                # 報酬の計算
        done = self._termination()             # 終了判定の更新
        debug = {'attempted_grasp': self._attempted_grasp,
                 'grasp_success': self._grasp_success}

        return observation, reward, done, debug

    # not to use
    def my_render(self, mode: str = 'human') -> None:
        """
        シミュレーション中の画像をレンダリングする関数．

        Args:
            mode (str, optional): レンダリングのモード. Defaults to 'human'.
        """
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros(self._width, self._height, 4))

        # 表示のためのカメラ設定(cameraTargetPositionで位置を微調整)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[2.25, -3, 1.75],  # [2, -3, 2]もいい感じ
            distance=0.5,
            yaw=30,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.client)

        # 表示のための投影マトリクス計算(fovで距離を変えられる)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=20, aspect=320 / 240,  # 10
            nearVal=0.1, farVal=100.0,
            physicsClientId=self.client)

        # 画像のリアルタイム表示
        frame = p.getCameraImage(self._width, self._height,
                                 viewMatrix=view_matrix,
                                 projectionMatrix=proj_matrix,
                                 physicsClientId=self.client)[2]
        self.rendered_img.set_data(frame)  # 描画画像の更新
        plt.draw()
        plt.pause(.00001)

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        """
        シミュレーション中の画像をレンダリングする関数.

        Args:
            mode (str, optional): レンダリングのモード. Defaults to 'rgb_array'.

        Returns:
            np.ndarray: シミュレーション中のRGB画像
        """

        RENDER_WIDTH = 960
        RENDER_HEIGHT = 720

        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self.sciurus.sciurus_id)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1,
            farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL)

        # renderer = self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
        rgb_array = rgb_array[:, :, :3]

        return rgb_array

    def close(self) -> None:
        """pybulletのシミュレーション環境を開放する関数."""
        p.disconnect(self.client)

    def __del__(self) -> None:
        """pybulletのシミュレーション環境を開放するデストラクタ."""
        p.disconnect()
        print('delete env!')

    def seed(self, seed: int = None):
        """
        異なる実験での結果の再現性を担保するために，numpyのシード値を返す関数．

        Args:
            seed (int, optional): 乱数生成のアルゴリズムを固定するためのシード値. Defaults to None.

        Returns:
            [type]: シード値
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':
    # 環境の生成
    env = SciurusGraspEnv()
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    # 初期の状態を観測
    obs_init = env.reset()
    print('initial observation shape:', obs_init.shape)
    plt.imshow(obs_init)
    plt.title('initial observation')
    plt.show()

    # ランダムに行動を洗濯して次の状態を観測
    action = env.action_space.sample()
    obs_next, r, done, info = env.step(action)
    plt.imshow(obs_next)
    plt.title('next observation')
    plt.show()
    print('next obserbation shape:', obs_next.shape)
    print('reward:', r)
    print('done:', done)
    print('info:', info)

    # # GUIでレンダリングする場合はアンコメント
    # env.render()

    # # 把持テスト用
    # for _ in range(2):
    #     observation, reward, done, debug = env.step([5, 0, 0, -10, 0, 0, 0, 0, 0])  # 把持テスト用
    #     print(debug)
