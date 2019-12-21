from MakeDataset import *


# plane, points, AABB = MakePointsData(500)
#
# indices = MakeIndices(plane, points)
#
# print(indices.shape)
#
# ax = ViewerInit(AABB, points=points)
# #plot_implicit(ax, plane.f_rep, AABB)
# plt.show()

# N, m, n
x_train, x_test, y_train, y_test = MakeDataset(5, 5, 100)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_test)
print(y_test)
