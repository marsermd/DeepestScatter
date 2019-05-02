import onnx
from caffe2.python import workspace
from caffe2.python.onnx.backend import Caffe2Backend
from torch.utils import data
import numpy as np
import glob
import os

from DisneyDescriptorDataset import DisneyDescriptorDataset
from LmdbDataset import LmdbDatasets


def getFileOptions():
    options = []
    for filename in glob.iglob('runs/**/*.onnx', recursive=True):
        options.append(filename)
    return options


def selectONNXFile():
    options = getFileOptions()
    print("\n".join(options))
    option = input("Select model to export\n")
    return option

if __name__ == '__main__':
    onnxFile = selectONNXFile()
    print("Selected:", onnxFile)

    directory = os.path.dirname(onnxFile)

    lmdbDatasets = LmdbDatasets("..\Data\Dataset")
    trainingSet = DisneyDescriptorDataset(lmdbDatasets.train)
    dataloader = data.DataLoader(trainingSet, batch_size=1, shuffle=False)
    zLayers, labels = next(iter(dataloader))
    zLayers = [z.fill_(1.5).numpy() for z in zLayers]
    for z in zLayers:
        for i in range(225):
            z[0][i] = i / 225
        z[0][225] = 1

    onnxModel = onnx.load(onnxFile)
    initNet, predictNet = Caffe2Backend.onnx_graph_to_caffe2_net(onnxModel)

    with open(os.path.join(directory, "onnxInit.pb"), "wb") as f:
        f.write(initNet.SerializeToString())
    with open(os.path.join(directory, "onnxInit.pbtxt"), "w") as f:
        f.write(str(initNet))
    with open(os.path.join(directory, "onnxPredict.pb"), "wb") as f:
        f.write(predictNet.SerializeToString())
    with open(os.path.join(directory, "onnxPredict.pbtxt"), "w") as f:
        f.write(str(predictNet))

    with open(os.path.join(directory, "onnxInit.pb"), "rb") as f:
        initNet = f.read()
    with open(os.path.join(directory, "onnxPredict.pb"), "rb") as f:
        predictNet = f.read()

    p = workspace.Predictor(initNet, predictNet)
    print(p.run(zLayers))

    print("exported")