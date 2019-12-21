import numpy as np
import numpy.linalg as LA
import random

import figure2 as F
from method import *
from view_method import *

def RandomPlane(high=1000):
    # 法線作成
    n = np.array([0, 0, 0])
    while LA.norm(n) == 0:
        n = np.random.rand(3)

    n = n / LA.norm(n)
    a, b, c = n

    # d作成
    d = Random(-high, high)

    print(a, b, c, d)

    return F.plane([a, b, c, d])

def CheckInternal(p, AABB):
    a, b, c, d = p
    xmin, xmax, ymin, ymax, zmin, zmax = AABB

    # AABBの各辺(直線とする)の交点の算出式
    x_node = lambda y, z: (d - b * y - c * z) / a
    y_node = lambda x, z: (d - a * x - c * z) / b
    z_node = lambda x, y: (d - a * x - b * y) / c

    # 12辺すべての交点
    X = np.array([x_node(ymin, zmin), x_node(ymin, zmax), x_node(ymax, zmin), x_node(ymax, zmax)])
    Y = np.array([y_node(xmin, zmin), y_node(xmin, zmax), y_node(xmax, zmin), y_node(xmax, zmax)])
    Z = np.array([z_node(xmin, ymin), z_node(xmin, ymax), z_node(xmax, ymin), z_node(xmax, ymax)])

    print(X, Y, Z)

    # 交点が1つでもAABB内にあればTrue
    if np.any((xmin <= X) & (X <= xmax)) or np.any((ymin <= Y) & (Y <= ymax)) or np.any((zmin <= Z) & (Z <= zmax)):
        return True

    return False

# 平面の点群＋ランダムに生成したAABB内にノイズ作成
# <条件>
# 1. 平面の点群+ノイズの合計値はNとし、平面点群の割合(最低0.5以上)をランダムで出す
# 2. AABB内に平面が入っていなかったら再生成
def MakeDataset(N, low=-100, high=100):
    # 平面点群の割合
    rate = Random(0.5, 1)
    size = int(N*rate//1)
    print(size)

    # 平面ランダム生成
    plane = RandomPlane(high=high)

    wrongAABB = [0,0,0,0,0,0]

    # AABBランダム生成
    while True:
        AABB = [Random(low, high) for i in range(6)]
        print(AABB)

        # AABB内に平面がなければ再生成
        if CheckInternal(plane.p, AABB):
            break

        wrongAABB = AABB[:]

    print(AABB)

    # 点群X, Y, Z, pointsを作成
    plane_points, _, _, _ = MakePoints(plane.f_rep, AABB, size, grid_step=size)

    xmin, xmax, ymin, ymax, zmin, zmax = AABB
    noise = np.array([[Random(xmin, xmax), Random(ymin, ymax), Random(zmin, zmax)] for i in range(N-size)])

    print(plane_points.shape, noise.shape)

    points = np.concatenate([plane_points, noise])

    print(type(points))


    return plane, points, AABB, wrongAABB
