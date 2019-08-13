import os
import argparse
import glob
import numpy as np

import progressbar

from PythonProtocols.SceneSetup_pb2 import SceneSetup
from LmdbDataset import LmdbDatasets

def uniformOnSphere():
    cosTheta = np.random.uniform(-1, 1)
    phi = np.random.uniform(0, np.pi * 2)

    sinTheta = np.sqrt(1 - cosTheta * cosTheta)

    x = np.cos(phi) * sinTheta
    y = np.sin(phi) * sinTheta
    z = cosTheta

    return np.array([x, y, z])

def getAllClouds(cloudsRoot):
    escapedRoot = glob.escape(cloudsRoot)
    clouds = glob.glob(escapedRoot + '/**/*.vdb', recursive=True)
    return list(map(lambda x: os.path.relpath(x, cloudsRoot), clouds))

def generateScene(cloud, sizeMeters, lightDirection):
    scene = SceneSetup()
    scene.cloud_path = cloud
    scene.cloud_size_m = sizeMeters

    scene.light_direction.x = lightDirection[0]
    scene.light_direction.y = lightDirection[1]
    scene.light_direction.z = lightDirection[2]

    return scene

def addScenes(clouds, scenesPerCloud, datasets):
    datasets, weights = zip(*[[datasets.train, 70], [datasets.test, 15], [datasets.validation, 15]])
    weights = np.array(weights, dtype=float)
    weights /= sum(weights)

    for cloud in progressbar.progressbar(clouds):
        dataset = np.random.choice(datasets, p=weights)

        for i in range(scenesPerCloud):
            minSize = 1000
            maxSize = 12_000
            logSizeMeters = np.random.uniform(np.log(minSize), np.log(maxSize))
            sizeMeters = np.exp(logSizeMeters)
            lightDirection = uniformOnSphere()
            scene = generateScene(cloud, sizeMeters, lightDirection)

            dataset.append(scene)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloudsRoot")
    parser.add_argument("--datasetRoot")

    parser.add_argument("--scenesPerCloud", type=int, default=30)

    args = parser.parse_args()

    clouds = getAllClouds(args.cloudsRoot)
    datasets = LmdbDatasets(args.datasetRoot, readonly=False)
    addScenes(clouds, args.scenesPerCloud, datasets)

