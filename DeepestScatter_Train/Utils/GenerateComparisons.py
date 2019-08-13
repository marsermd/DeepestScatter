import OpenEXR
import os
import numpy as np
from PIL import Image

def readImage(inputFile):
    inputFile = OpenEXR.InputFile(inputFile)
    window = inputFile.header()['dataWindow']

    size = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
    image = np.fromstring(inputFile.channel('R'), dtype='f')
    image = image.reshape(size)
    image = np.flip(image, 0)
    return image


def writeToneMappedFile(image, outputFile):
    avgLuminance = np.sum(image) / image.size

    exposure = 0.4
    lw = image
    ld = lw * exposure / avgLuminance
    ld = ld / (1 + ld)

    image = image * ld / lw
    image = np.power(image, 1/2.2)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    image = np.clip(image, 0, 0.9999) * 256

    Image.fromarray(image.astype('uint8')).save(outputFile)

def writeDiff(image, gtroundTruth, outputFile):
    if "PT" in outputFile:
        return
    delta = np.power(image - gtroundTruth, 2)
    bias = np.sqrt(np.sum(delta)) / delta.size
    print(outputFile, bias)

    delta = np.abs(image - gtroundTruth) / 4
    delta = np.repeat(delta[:, :, np.newaxis], 3, axis=2)
    delta = np.clip(delta, 0, 0.9999) * 256

    Image.fromarray(delta.astype('uint8')).save(outputFile)

if __name__ == '__main__':
    print("======================================")
    print("=========Generate Comparisons=========")
    print("======================================")

    renders = ["PT", "NN", "BNN"]

    directory = "../Data/Renders"
    for fileName in os.listdir(directory):
        if ".PT." in fileName:
            groundTruth = readImage(os.path.join(directory, fileName))
            renderFiles = [os.path.join(directory, fileName.replace("PT", render)) for render in renders]

            for renderFile in renderFiles:
                inputImage = readImage(renderFile)

                outputFile = renderFile.replace("converged.exr", "tonemapped.png").replace("/Renders", "/Preview")
                writeToneMappedFile(inputImage, outputFile)

                outputFile = renderFile.replace("converged.exr", "diff.png").replace("/Renders", "/Preview")
                writeDiff(inputImage, groundTruth, outputFile)