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

def logEps(x):
    val = x * 1e1 + 1
    # if value is between -1 and 0.01, log will still work.
    val = torch.max(val, 0.0099 + val / 100)
    return torch.log(val)

class LogModel(torch.nn.Module):
    def __init__(self, model):
        super(LogModel, self).__init__()
        self.model = model

    def forward(self, input):
        return logEps(model(input))

def set_seed(seed: int):
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.deterministic = True
    torch.backends.benchmark = False

def saveCheckpoint(state, is_best, filename='runs/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/model_best.pth.tar')

def exportModel(dataset, model):
    torch.set_printoptions(precision=10)
    torch.set_printoptions(threshold=5000)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    z, labels = next(iter(dataloader))
    z = z.to(device)

    print(model(z))
    print(labels)

    tracedModel = torch.jit.trace(model, z)
    tracedModel.save('runs/checkpoint.pt')

if __name__ == '__main__':

    useCuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if useCuda else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if useCuda else 'torch.FloatTensor')
    set_seed(566)

    # Parameters
    params = {'batch_size': 1024,
              'shuffle': True,
              'num_workers': 8,
              'drop_last': True}
    maxEpochs = 200

    lmdbDatasets = LmdbDatasets("..\Data\Dataset")
    writer = SummaryWriter()

    # Generators
    trainingSet = DisneyDescriptorDataset(lmdbDatasets.train)
    trainingGenerator = data.DataLoader(trainingSet, **params)

    model = DisneyModel()
    logModel = LogModel(model)
    logModel.to(device)

    criterion = torch.nn.MSELoss().to(device)
    lr = 1.e-3
    optimizer = optim.Adam(logModel.parameters(), lr, amsgrad=True)

    for epoch in range(maxEpochs):
        optimizer.lr = lr / math.sqrt(epoch + 1)
        print(f"going through epoch {epoch} with lr: {optimizer.lr}")

        start = datetime.datetime.now()
        # Training
        for batchId, (z, labels) in enumerate(trainingGenerator):
            # Transfer to GPU
            z = z.to(device)
            labels = labels.to(device)
            labels = logEps(labels)

            def closure():
                optimizer.zero_grad()
                out = logModel(z)
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
            saveCheckpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),  # using model on purpose, LogModel is only used for calculating loss
                'optimizer': optimizer.state_dict(),
            }, True)
            exportModel(trainingSet, model)

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