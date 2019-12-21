from MakeDataset import *


plane, points, AABB, wrongAABB = MakeDataset(500)

ax = ViewerInit(AABB, points=points)
#plot_implicit(ax, plane.f_rep, AABB)
plt.show()

# print("wrong:{}".format(wrongAABB))
#
# ax = ViewerInit(points, wrongAABB)
# plot_implicit(ax, plane.f_rep, wrongAABB)
# plt.show()

# p = [0.28952861718766604,0.6746372473172653,0.6789976173460265,5.749558831586938]
# AABB = [70.78533877376952, 7.260174537064074, 38.26752396122055, -47.24771407687409, 53.631459297425636, -99.34200107430935]
# print(CheckInternal(p, AABB))
#
# plane = F.plane(p)
# ax = ViewerInit(AABB)
# plot_implicit(ax, plane.f_rep, AABB)
# plt.show()