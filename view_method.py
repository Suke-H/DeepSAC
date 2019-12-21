import numpy as np
import itertools

# seabornはimportしておくだけでもmatplotlibのグラフがきれいになる
import seaborn as sns

sns.set_style("darkgrid")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from method import *


def line(a, b):
    t = np.arange(0, 1, 0.01)

    x = a[0] * t + b[0] * (1 - t)
    y = a[1] * t + b[1] * (1 - t)
    z = a[2] * t + b[2] * (1 - t)

    return x, y, z


def ViewerInit(AABB, points=None):
    # グラフの枠を作っていく
    fig = plt.figure()
    ax = Axes3D(fig)

    # 軸にラベルを付けたいときは書く
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if points is not None:

        X, Y, Z = Disassemble(points)

        # 点群を描画
        ax.plot(X, Y, Z, marker=".", linestyle='None', color="blue")

    """
    if len(normals) != 0:
        #法線を描画
        U, V, W = Disassemble(normals)
        ax.quiver(X, Y, Z, U, V, W,  length=0.1, normalize=True, color="blue")
    """

    # OBBを描画
    # OBBViewer(ax, points)

    # AABB描画
    AABBviewer(ax, AABB)

    return ax


# 点群を入力としてOBBを描画する
def OBBViewer(ax, points):
    # OBB生成
    max_p, min_p, _ = buildOBB(points)

    # 直積：[smax, smin]*[tmax, tmin]*[umax, umin] <=> 頂点
    s_axis = np.vstack((max_p[0], min_p[0]))
    t_axis = np.vstack((max_p[1], min_p[1]))
    u_axis = np.vstack((max_p[2], min_p[2]))

    products = np.asarray(list(itertools.product(s_axis, t_axis, u_axis)))
    vertices = np.sum(products, axis=1)

    # 各頂点に対応するビットの列を作成
    bit = np.asarray([1, -1])
    vertices_bit = np.asarray(list(itertools.product(bit, bit, bit)))

    # 頂点同士のハミング距離が1なら辺を引く
    for i, v1 in enumerate(vertices_bit):
        for j, v2 in enumerate(vertices_bit):
            if np.count_nonzero(v1 - v2) == 1:
                x, y, z = line(vertices[i], vertices[j])
                ax.plot(x, y, z, marker=".", color="orange")

    # OBBの頂点の1つ
    vert_max = min_p[0] + min_p[1] + max_p[2]
    vert_min = max_p[0] + max_p[1] + min_p[2]

    # xyzに分解
    Xmax, Ymax, Zmax = Disassemble(max_p)
    Xmin, Ymin, Zmin = Disassemble(min_p)

    # 頂点なども描画
    ax.plot(Xmax, Ymax, Zmax, marker="X", linestyle="None", color="red")
    ax.plot(Xmin, Ymin, Zmin, marker="X", linestyle="None", color="blue")
    ax.plot([vert_max[0], vert_min[0]], [vert_max[1], vert_min[1]], [vert_max[2], vert_min[2]], marker="o",
            linestyle="None", color="black")


# AABB = [xmin, xmax, ymin, ymax, zmin, zmax]を入力にAABB描画
def AABBviewer(ax, AABB):
    # x,y,zに分ける
    x_axis = [AABB[0], AABB[1]]
    y_axis = [AABB[2], AABB[3]]
    z_axis = [AABB[4], AABB[5]]

    # 直積が頂点の座標になる
    vertices = np.asarray(list(itertools.product(x_axis, y_axis, z_axis)))

    # 各頂点に対応するビットの列を作成
    bit = np.asarray([1, -1])
    vertices_bit = np.asarray(list(itertools.product(bit, bit, bit)))

    # 頂点同士のハミング距離が1なら辺を引く
    for i, v1 in enumerate(vertices_bit):
        for j, v2 in enumerate(vertices_bit):
            if np.count_nonzero(v1 - v2) == 1:
                x, y, z = line(vertices[i], vertices[j])
                ax.plot(x, y, z, marker=".", color="orange")


# ラベルの色分け
def LabelViewer(ax, points, label_list, max_label):
    colorlist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                 '#17becf']

    # ラベルの数
    label_num = np.max(label_list)

    # ラベルなしの点群を白でプロット

    X, Y, Z = Disassemble(points[np.where(label_list == 0)])
    ax.plot(X, Y, Z, marker=".", linestyle="None", color="white")

    for i in range(1, label_num + 1):
        # 同じラベルの点群のみにする
        same_label_points = points[np.where(label_list == i)]

        print("{}:{}".format(i, same_label_points.shape[0]))

        # plot
        X, Y, Z = Disassemble(same_label_points)
        if i == max_label:
            ax.plot(X, Y, Z, marker="o", linestyle="None", color=colorlist[i % len(colorlist)])
        else:
            ax.plot(X, Y, Z, marker=".", linestyle="None", color=colorlist[i % len(colorlist)])


# 陰関数のグラフ描画
# fn  ...fn(x, y, z) = 0の左辺
# AABB_size ...AABBの各辺をAABB_size倍する
def plot_implicit(ax, fn, AABB, AABB_size=1.5, contourNum=30):
    xmin, xmax, ymin, ymax, zmin, zmax = AABB

    # AABBの各辺がAABB_size倍されるように頂点を変更
    xmax = xmax + (xmax - xmin) / 2 * AABB_size
    xmin = xmin - (xmax - xmin) / 2 * AABB_size
    ymax = ymax + (ymax - ymin) / 2 * AABB_size
    ymin = ymin - (ymax - ymin) / 2 * AABB_size
    zmax = zmax + (zmax - zmin) / 2 * AABB_size
    zmin = zmin - (zmax - zmin) / 2 * AABB_size

    A_X = np.linspace(xmin, xmax, 100)  # resolution of the contour
    A_Y = np.linspace(ymin, ymax, 100)
    A_Z = np.linspace(zmin, zmax, 100)
    B_X = np.linspace(xmin, xmax, 15)  # number of slices
    B_Y = np.linspace(ymin, ymax, 15)
    B_Z = np.linspace(zmin, zmax, 15)
    # A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B_Z:  # plot contours in the XY plane
        X, Y = np.meshgrid(A_X, A_Y)
        Z = fn(X, Y, z)
        ax.contour(X, Y, Z + z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B_Y:  # plot contours in the XZ plane
        X, Z = np.meshgrid(A_X, A_Z)
        Y = fn(X, y, Z)
        ax.contour(X, Y + y, Z, [y], zdir='y')

    for x in B_X:  # plot contours in the YZ plane
        Y, Z = np.meshgrid(A_Y, A_Z)
        X = fn(x, Y, Z)
        ax.contour(X + x, Y, Z, [x], zdir='x')

    # (拡大した)AABBの範囲に制限
    ax.set_zlim3d(zmin, zmax)
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)


def plot_normal(ax, figure, X, Y, Z):
    # 図形の方程式から点群を作る
    # points, X, Y, Z = MakePoints(figure.f_rep, epsilon=0.01)

    # 法線
    normals = figure.normal(X, Y, Z)
    U, V, W = Disassemble(normals)

    # 法線を描画
    ax.quiver(X, Y, Z, U, V, W, length=0.1, color='red', normalize=True)
