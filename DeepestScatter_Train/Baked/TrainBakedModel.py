import torch
from torch.utils import data

from BakedDataset import BakedDataset
from BakedModel import BakedModel
from Trainer import Trainer


class BakedTrainer(Trainer):
    BAKED_LAYERS = 9
    REALTIME_LAYERS = 3

    def __init__(self):
        super(BakedTrainer, self).__init__()

    def createDataset(self, lmdbDataset):
        return BakedDataset(lmdbDataset, self.BAKED_LAYERS, self.REALTIME_LAYERS)

    def createModel(self):
        return BakedModel(self.BAKED_LAYERS, self.REALTIME_LAYERS)

    def save(self, model, dataset):
        torch.set_printoptions(precision=10)

        lightProbeModel = model.lightProbeModel
        rendererModel = model.rendererModel

        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
        args, labels = next(iter(dataloader))
        bakedDescriptors, bakedPowers, disneyDescriptor, omega, alpha = self.toCuda(args)

        lightProbes = [lightProbeModel(descriptor) for descriptor in bakedDescriptors]
        lightProbe = sum([lightProbe * power.repeat(1, 200) for (lightProbe, power) in zip(lightProbes, bakedPowers)])

        lightProbe = model.applyAnglesToLightProbe(lightProbe, omega, alpha)

        out = rendererModel(lightProbe, disneyDescriptor)

        print(labels)
        print(out)

        self.exportModel(lightProbeModel, bakedDescriptors[0])
        self.exportModel(rendererModel, (lightProbe, disneyDescriptor))



if __name__ == '__main__':
    print("======================================")
    print("=========TRAINING BAKED MODEL=========")
    print("======================================")
    BakedTrainer().run()