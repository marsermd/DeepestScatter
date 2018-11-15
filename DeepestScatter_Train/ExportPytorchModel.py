import onnx
from onnx_caffe2.backend import Caffe2Backend
import glob
import os

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

    onnxModel = onnx.load(onnxFile)
    initNet, predictNet = Caffe2Backend.onnx_graph_to_caffe2_net(onnxModel.graph)
    with open(os.path.join(directory, "onnxInit.pb"), "wb") as f:
        f.write(initNet.SerializeToString())
    with open(os.path.join(directory, "onnxInit.pbtxt"), "w") as f:
        f.write(str(initNet))
    with open(os.path.join(directory, "onnxPredict.pb"), "wb") as f:
        f.write(predictNet.SerializeToString())
    with open(os.path.join(directory, "onnxPredict.pbtxt"), "w") as f:
        f.write(str(predictNet))

    print("exported")