import keras
from sklearn.preprocessing import Imputer

import figure2 as F
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

    # 交点が1つでもAABB内にあればTrue
    if np.any((xmin <= X) & (X <= xmax)) or np.any((ymin <= Y) & (Y <= ymax)) or np.any((zmin <= Z) & (Z <= zmax)):
        return True

    return False

# 平面の点群＋ランダムに生成したAABB内にノイズ作成
# <条件>
# 1. 平面の点群+ノイズの合計値はNとし、平面点群の割合(最低0.5以上)をランダムで出す
# 2. AABB内に平面が入っていなかったら再生成
def MakePointsData(N, low=-100, high=100, grid_step=50):
    # 平面点群の割合をランダムで決める
    rate = Random(0.5, 1)
    size = int(N*rate//1)

    # 平面ランダム生成
    plane = RandomPlane(high=high)

    # AABBランダム生成
    while True:
        AABB = [Random(low, high) for i in range(6)]

        # AABB内に平面がなければ再生成
        if CheckInternal(plane.p, AABB):
            break

    # 平面点群を生成
    plane_points, _, _, _ = MakePoints(plane.f_rep, AABB, size, grid_step=grid_step)

    # N-size点のノイズ生成
    xmin, xmax, ymin, ymax, zmin, zmax = AABB
    noise = np.array([[Random(xmin, xmax), Random(ymin, ymax), Random(zmin, zmax)] for i in range(N-size)])

    #print(plane_points.shape, noise.shape)

    # 平面点群とノイズの結合
    # シャッフルもしておく
    points = np.concatenate([plane_points, noise])
    np.random.shuffle(points)

    return plane, points, AABB

def MakeIndices(plane, points):
    X, Y, Z = Disassemble(points)

    # 点群の各点と平面との距離
    indices = np.abs(plane.f_rep(X,Y,Z))

    # 標準化
    s = np.std(indices)
    m = np.mean(indices)
    indices = (indices - m) / s

    # nanやinfが出たらやり直し
    #if np.any(np.isnan(indices)) or np.any(np.isinf(indices)):
        #return None

    # 0, 1になったらOK
    #print(np.mean(indices), np.std(indices))

    return indices


def PlaneDict(points, N):
    # ランダムに3点ずつN組抽出
    n = points.shape[0]
    points_set = points[np.array([np.random.choice(n, 3, replace=False) for i in range(N)]), :]
    #points_set = points[np.random.choice(n, size=(int((n - n % 3) / 3), 3), replace=False), :]

    # print("points:{}".format(points_set.shape))

    # 分割
    # [a1, b1, c1] -> [a1] [b1, c1]
    a0, a1 = np.split(points_set, [1], axis=1)

    # a2 = [[b1-a1], ...,[bn-an]]
    #      [[c1-a1], ...,[cn-an]]
    a2 = np.transpose(a1 - a0, (1, 0, 2))

    # n = (b-a) × (c-a)
    n = np.cross(a2[0], a2[1])

    # 単位ベクトルに変換
    n = norm(n)

    # d = n・a
    a0 = np.reshape(a0, (a0.shape[0], 3))
    d = np.sum(n * a0, axis=1)

    # パラメータ
    # p = [nx, ny, nz, d]
    d = np.reshape(d, (d.shape[0], 1))
    p = np.concatenate([n, d], axis=1)

    # print("平面生成")

    # 平面生成
    Planes = [F.plane(p[i]) for i in range(p.shape[0])]

    return Planes

def MakeOneData(M, N):
    # 点群データ作成
    True_plane, points, AABB = MakePointsData(N)
    # その点群に対してM-1個ランダムに平面生成する
    planes = PlaneDict(points, M-1)
    # 0～M-1から正解ラベルをランダムに選択
    y = np.random.choice(M)
    planes.insert(y, True_plane)
    # 入力データ作成
    x = np.array([MakeIndices(planes[i], points) for i in range(M)])

    # nanを平均値で補完する
    #imr = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    #x = imr.fit_transform(x)

    print(x.shape)

    return x, y

def MakeDataset(N, m, n):
    # N個onedataを重ねるだけ
    dataset = [MakeOneData(m, n) for i in range(N)]
    x_data = np.array([dataset[i][0] for i in range(N)], dtype='float32')
    y_data = np.array([dataset[i][1] for i in range(N)])

    # 訓練データと検証データに分ける(9:1にする)
    perm = np.random.permutation(N)
    N1 = int(N * 0.9)

    x_train = x_data[perm[:N1]]
    x_test = x_data[perm[N1:N]]
    y_train = y_data[perm[:N1]]
    y_test = y_data[perm[N1:N]]

    # one_hotに変換
    y_train = keras.utils.to_categorical(y_train, m)
    y_test = keras.utils.to_categorical(y_test, m)

    return x_train, x_test, y_train, y_test


# # M: RANSACの候補数
# # N: 点群数
# def MakeDataset(M, N):
#     plane_points_set = [MakePointsData(N)[:2] for i in range(M)]
#     indices_set = np.array([MakeIndices(plane_points_set[i][0], plane_points_set[i][1]) for i in range(M)])
#
#     print(indices_set.shape)
#
#     x_data = np.array(indices_set, dtype='float32')
#
#     return

