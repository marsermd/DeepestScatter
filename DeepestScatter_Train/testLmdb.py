from lmdb import Environment
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PythonProtocols.ScatterSample_pb2 import ScatterSample


env = Environment("./dataset.lmdb", subdir=False, max_dbs=128, mode=0, create=False, readonly=True)

db = env.open_db(ScatterSample.DESCRIPTOR.full_name.encode('ascii'), integerkey=True, create=False)
with env.begin() as transaction:
    print(transaction.stat(db))

    x = np.zeros(32768)
    y = np.zeros(32768)
    z = np.zeros(32768)

    sample = ScatterSample()
    for id in range(32768):
        result = transaction.get(id.to_bytes(4, 'little'), db=db)
        sample.ParseFromString(result)
        x[id] = sample.point.x
        y[id] = sample.point.y
        z[id] = -sample.point.z

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, zdir='y', s=1, alpha=0.16)
    ax.axis('equal')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)

    x_scale = 1
    y_scale = 1
    z_scale = 1

    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0


    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)


    ax.get_proj = short_proj

    plt.show()