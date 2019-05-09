import torch
from torch.utils import data

from DisneyDescriptorDataset import DisneyDescriptorDataset
from DisneyModel import DisneyModel
from Trainer import Trainer


class DisneyTrainer(Trainer):

    def __init__(self):
        super(DisneyTrainer, self).__init__()

    def createDataset(self, lmdbDataset):
        return DisneyDescriptorDataset(lmdbDataset)

    def createModel(self):
        return DisneyModel()

    def save(self, model, dataset):
        torch.set_printoptions(precision=10)
        torch.set_printoptions(threshold=5000)

        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
        z, labels = next(iter(dataloader))
        z = z.to(self.device)

        print(labels)
        print(model(z))

        self.exportModel(model, z)



if __name__ == '__main__':
    print("======================================")
    print("========TRAINING DISNEY MODEL=========")
    print("======================================")
    DisneyTrainer().run()