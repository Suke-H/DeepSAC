import numpy as np
import numpy.linalg as LA
import random

# aからbまでのランダムな実数値を返す
def Random(a, b):
    return (b - a) * np.random.rand() + a


def norm(normal):
    # ベクトルが一次元のとき
    if len(normal.shape) == 1:
        if LA.norm(normal) == 0:
            # print("Warning: 法線ベクトルがゼロです！")
            return normal

        else:
            return normal / LA.norm(normal)

    # ベクトルが二次元
    # (figureのnormalにX,Y,Z(=x,y,zの一次元配列)をいれると二次元のnormalになる)
    else:
        # 各法線のノルムをnormに格納
        norm = LA.norm(normal, ord=2, axis=1)

        # normが0の要素は1にする(normalをnormで割る際に0除算を回避するため)
        norm = np.where(norm == 0, 1, norm)

        # normalの各成分をノルムで割る
        norm = np.array([np.full(3, norm[i]) for i in range(len(norm))])
        return normal / norm


# 点群データなどをx, y, zに分解する

# [x1, y1, z1]         [x1, x2, ..., xn]
#      :        ->    [y1, y2, ..., yn]
# [xn, yn, zn]         [z1, z2, ..., zn]
def Disassemble(XYZ):
    XYZ = XYZ.T[:]
    X = XYZ[0, :]
    Y = XYZ[1, :]
    Z = XYZ[2, :]

    return X, Y, Z


def MakePoints(fn, AABB, sampling_size, grid_step=50, epsilon=0.05):
    # import time
    # start = time.time()
    xmin, xmax, ymin, ymax, zmin, zmax = AABB

    while True:

        # 点群X, Y, Z, pointsを作成
        x = np.linspace(xmin, xmax, grid_step)
        y = np.linspace(ymin, ymax, grid_step)
        z = np.linspace(zmin, zmax, grid_step)

        X, Y, Z = np.meshgrid(x, y, z)

        # 格子点X, Y, Zをすべてfnにぶち込んでみる
        W = np.array([[fn(X[i][j], Y[i][j], Z[i][j]) for j in range(grid_step)] for i in range(grid_step)])
        # 変更前
        # W = fn(X, Y, Z)

        # Ｗが0に近いインデックスを取り出す
        index = np.where(np.abs(W) <= epsilon)
        index = [(index[0][i], index[1][i], index[2][i]) for i in range(len(index[0]))]
        # print(index)

        # indexがsampling_size分なかったらgrid_stepを増やしてやり直し
        if len(index) >= sampling_size:
            break

        grid_step += 10

    # サンプリング
    index = random.sample(index, sampling_size)

    # 格子点から境界面(fn(x,y,z)=0)に近い要素のインデックスを取り出す
    pointX = np.array([X[i] for i in index])
    pointY = np.array([Y[i] for i in index])
    pointZ = np.array([Z[i] for i in index])

    # points作成([[x1,y1,z1],[x2,y2,z2],...])
    points = np.stack([pointX, pointY, pointZ])
    points = points.T

    # end = time.time()
    # print("time:{}s".format(end-start))

    return points, pointX, pointY, pointZ


###OBB生成####
def buildOBB(points):
    # 分散共分散行列Sを生成
    S = np.cov(points, rowvar=0, bias=1)

    # 固有ベクトルを算出
    w, svd_vector = LA.eig(S)

    # 固有値が小さい順に固有ベクトルを並べる
    svd_vector = svd_vector[np.argsort(w)]

    # print(S)
    # print(svd_vector)
    # print("="*50)

    # 正規直交座標にする(=直行行列にする)
    #############################################
    u = np.asarray([svd_vector[i] / np.linalg.norm(svd_vector[i]) for i in range(3)])

    # 点群の各点と各固有ベクトルとの内積を取る
    # P V^T = [[p1*v1, p1*v2, p1*v3], ... ,[pN*v1, pN*v2, pN*v3]]
    inner_product = np.dot(points, u.T)

    # 各固有値の内積最大、最小を抽出(max_stu_point = [s座標max, tmax, umax])
    max_stu_point = np.amax(inner_product, axis=0)
    min_stu_point = np.amin(inner_product, axis=0)

    # xyz座標に変換・・・単位ベクトル*座標
    # max_xyz_point = [[xs, ys, zs], [xt, yt, zt], [xu, yu, zu]]
    max_xyz_point = np.asarray([u[i] * max_stu_point[i] for i in range(3)])
    min_xyz_point = np.asarray([u[i] * min_stu_point[i] for i in range(3)])

    """
    max_index = 
    print(max_index)
    max_point = np.asarray([points[max_index[i]] for i in range(3)])

    min_index = np.argmin(inner_product, axis=0)
    min_point = np.asarray([points[min_index[i]] for i in range(3)])
    """
    # 対角線の長さ
    vert_max = min_xyz_point[0] + min_xyz_point[1] + max_xyz_point[2]
    vert_min = max_xyz_point[0] + max_xyz_point[1] + min_xyz_point[2]
    l = np.linalg.norm(vert_max - vert_min)

    return max_xyz_point, min_xyz_point, l


###AABB生成####
def buildAABB(points):
    # なんとこれで終わり
    max_p = np.amax(points, axis=0)
    min_p = np.amin(points, axis=0)

    return max_p, min_p

