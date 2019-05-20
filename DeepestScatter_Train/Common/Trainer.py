import abc
from abc import abstractmethod
import datetime
import math
import shutil
import random

import numpy as np

import torch
from torch.utils import data
import torch.optim as optim
import torch.onnx

from DisneyDescriptorDataset import DisneyDescriptorDataset
from DisneyModel import DisneyModel
from LmdbDataset import LmdbDatasets

from tensorboardX import SummaryWriter


class LogModel(torch.nn.Module):
    def __init__(self, model):
        super(LogModel, self).__init__()
        self.model = model

    @staticmethod
    def logEps(x):
        val = x * 1e1 + 1
        # if value is between -1 and 0.01, log will still work.
        val = torch.max(val, 0.0099 + val / 100)
        return torch.log(val)

    def forward(self, input):
        if isinstance(input, tuple) or isinstance(input, list):
            return LogModel.logEps(self.model(*input))
        else:
            return LogModel.logEps(self.model(input))


def set_seed(seed: int):
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.deterministic = True
    torch.backends.benchmark = False


class Trainer:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        useCuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if useCuda else "cpu")
        torch.set_default_tensor_type('torch.cuda.FloatTensor' if useCuda else 'torch.FloatTensor')

        set_seed(566)

    def saveCheckpoint(self, state, is_best, filename='runs/checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'runs/model_best.pth.tar')

    def exportModel(self, model, args):
        tracedModel = torch.jit.trace(model, args)
        tracedModel.save(f'runs/{type(model).__name__}.pt')

    @abstractmethod
    def save(self, model, dataset):
        pass

    @abstractmethod
    def createDataset(self, lmdbDatasets):
        pass

    @abstractmethod
    def createModel(self):
        pass

    def run(self):
        # Parameters
        params = {'batch_size': 1024,
                  'shuffle': True,
                  'num_workers': 8,
                  'drop_last': True}
        maxEpochs = 200

        writer = SummaryWriter()

        # Generators
        lmdbDatasets = LmdbDatasets("D:\\Dataset")

        trainingSet = self.createDataset(lmdbDatasets.train)
        trainingGenerator = data.DataLoader(trainingSet, **params)

        model = self.createModel()
        logModel = LogModel(model)
        logModel.to(self.device)

        criterion = torch.nn.MSELoss().to(self.device)
        lr = 1.e-3
        optimizer = optim.Adam(logModel.parameters(), lr, amsgrad=True)

        for epoch in range(maxEpochs):
            optimizer.lr = lr / math.sqrt(epoch + 1)
            print(f"going through epoch {epoch} with lr: {optimizer.lr}")

            start = datetime.datetime.now()
            # Training
            for batchId, (args, labels) in enumerate(trainingGenerator):
                # Transfer to GPU
                if isinstance(args, tuple) or isinstance(args, list):
                    args = [x.to(self.device) for x in args]
                else:
                    args = args.to(self.device)
                labels = labels.to(self.device)
                labels = logModel.logEps(labels)

                def closure():
                    optimizer.zero_grad()
                    out = logModel(args)
                    loss = criterion(out.squeeze(1), labels.float())
                    loss.backward()
                    return loss

                loss = closure()
                print("Train #", epoch, batchId, loss)
                writer.add_scalar('loss', loss, batchId + epoch * len(trainingGenerator))

                if math.isnan(loss):
                    print("Got NAN loss!")
                    exit(0)

                optimizer.step(closure)

                print((datetime.datetime.now() - start))
                self.saveCheckpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),  # using model on purpose, LogModel is only used for calculating loss
                    'optimizer': optimizer.state_dict(),
                }, True)

                self.save(model, trainingSet)

            # Todo: calculate loss on validation dataset
            # id = 0
            # # Validation
            # with torch.set_grad_enabled(False):
            #     for descriptors, angles, labels in validation_generator:
            #         descriptors = descriptors.to(device)
            #         angles = angles.to(device)
            #         labels = labels.to(device)
            #         labels = torch.log(labels + 1)
            #
            #         outputs = logModel((descriptors, angles))
            #         loss = criterion(outputs.squeeze(1), labels.float())
            #         print("Validation #", epoch, id, loss)
            #
            #         id += 1
            #         if id > 20:
            #             break

        writer.close()
