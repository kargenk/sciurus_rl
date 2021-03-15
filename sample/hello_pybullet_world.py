import pybullet as p
import time
import pybullet_data

# 物理シミュレーションへの接続
physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
# PyBulletに同梱されているpybullet_dataパッケージを利用するため，ディレクトリを登録
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)  # 重力の設定[x, y, z]

# 床のオブジェクトを読み込んでオブジェクトの非負整数値のIDを返す，読み込めなければ-1が返る
plane_id = p.loadURDF("plane.urdf")

# 初期位置と角度を指定してオブジェクトを読み込む
start_pos = [0, 0, 1]
start_orientation = p.getQuaternionFromEuler([0, 0, 0])  # [roll, pitch, yaw], radians
boxId = p.loadURDF("r2d2.urdf", start_pos, start_orientation)

for i in range(100):
    p.stepSimulation()
    time.sleep(1. / 240.)  # PyBullet内で1ステップ分待つ

# 位置[x, y, z]と角度[x, y, z, w]を取得
cube_pos, cube_orn = p.getBasePositionAndOrientation(boxId)
print(cube_pos, cube_orn)

p.disconnect()  # 物理シミュレーションから切断
