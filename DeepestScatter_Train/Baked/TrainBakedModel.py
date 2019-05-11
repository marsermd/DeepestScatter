import torch
from torch.utils import data

from BakedDataset import BakedDataset
from BakedModel import BakedModel
from Trainer import Trainer


class BakedTrainer(Trainer):
    BAKED_LAYERS = 7
    REALTIME_LAYERS = 4

    def __init__(self):
        super(BakedTrainer, self).__init__()

    def createDataset(self, lmdbDataset):
        return BakedDataset(lmdbDataset, self.BAKED_LAYERS, self.REALTIME_LAYERS)

    def createModel(self):
        return BakedModel(self.BAKED_LAYERS, self.REALTIME_LAYERS)

    def save(self, model, dataset):
        torch.set_printoptions(precision=10)
        torch.set_printoptions(threshold=5000)

        lightProbeModel = model.lightProbe
        rendererModel = model.renderer

        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
        args, labels = next(iter(dataloader))
        bakedDescriptor, disneyDescriptor, omega, alpha = [x.to(self.device) for x in args]

        lightProbe = lightProbeModel(bakedDescriptor)
        lightProbe = torch.cat(
            (
                lightProbe,
                omega.unsqueeze(1),
                alpha.unsqueeze(1)
            ),
            dim=1
        )
        out = rendererModel(lightProbe, disneyDescriptor)

        print(labels)
        print(out)

        self.exportModel(lightProbeModel, bakedDescriptor)
        self.exportModel(rendererModel, (lightProbe, disneyDescriptor))



if __name__ == '__main__':
    print("======================================")
    print("=========TRAINING BAKED MODEL=========")
    print("======================================")
    BakedTrainer().run()