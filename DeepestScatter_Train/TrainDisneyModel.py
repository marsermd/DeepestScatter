import datetime
import math
import shutil

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

    def forward(self, input):
        return torch.log(model(*input) + 1)

def saveCheckpoint(state, is_best, filename='runs/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/model_best.pth.tar')

def exportModel(model):
    descriptor = torch.randn(1, DisneyModel.BLOCK_COUNT, DisneyModel.DESCRIPTOR_LAYER_DIMENSION, requires_grad=True)
    angle = torch.randn(1, requires_grad=True)

    _ = torch.onnx._export(
        model,
        (descriptor, angle),
        "runs/DisneyModel.onnx",
        export_params=True
    )

if __name__ == '__main__':

    useCuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if useCuda else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if useCuda else 'torch.FloatTensor')

    # Parameters
    params = {'batch_size': 4096,
              'shuffle': True,
              'num_workers': 8,
              'drop_last': True}
    maxEpochs = 200

    lmdbDatasets = LmdbDatasets("..\Data\Dataset")
    writer = SummaryWriter()

    # Generators
    trainingSet = DisneyDescriptorDataset(lmdbDatasets.train)
    trainingGenerator = data.DataLoader(trainingSet, **params)

    validationSet = DisneyDescriptorDataset(lmdbDatasets.validation)
    validationGenerator = data.DataLoader(validationSet, **params)

    model = DisneyModel()
    logModel = LogModel(model)
    logModel.to(device)

    criterion = torch.nn.MSELoss().to(device)
    lr = 1.e-3
    optimizer = optim.Adam(logModel.parameters(), lr)

    for epoch in range(maxEpochs):
        optimizer.lr = lr / math.sqrt(epoch + 1)

        start = datetime.datetime.now()
        # Training
        for batchId, (descriptors, angles, labels) in enumerate(trainingGenerator):
            # Transfer to GPU
            descriptors = descriptors.to(device)
            angles = angles.to(device)
            labels = labels.to(device)
            labels = torch.log(labels + 1)

            def closure():
                optimizer.zero_grad()
                out = logModel((descriptors, angles))
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
        exportModel(model)

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