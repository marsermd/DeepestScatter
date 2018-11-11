import datetime

import torch
from python_utils import time
from torch.utils import data
import torch.optim as optim

from DisneyDescriptorDataset import DisneyDescriptorDataset
from DisneyModel import DisneyModel
from LmdbDataset import LmdbDatasets


class LogModel(torch.nn.Module):
    def __init__(self, model):
        super(LogModel, self).__init__()
        self.model = model

    def forward(self, input):
        return torch.log(model(*input) + 1)

if __name__ == '__main__':

    useCuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if useCuda else "cpu")

    # Parameters
    params = {'batch_size': 1024,
              'shuffle': True,
              'num_workers': 4,
              'drop_last': True}
    max_epochs = 100

    lmdbDatasets = LmdbDatasets("..\Data\Dataset")

    # Generators
    training_set = DisneyDescriptorDataset(lmdbDatasets.train)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = DisneyDescriptorDataset(lmdbDatasets.validation)
    validation_generator = data.DataLoader(validation_set, **params)

    model = DisneyModel()
    logModel = LogModel(model)
    logModel.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(logModel.parameters(), lr=0.01)

    for epoch in range(max_epochs):
        id = 0
        start = datetime.datetime.now()
        # Training
        for descriptors, angles, labels in training_generator:
            # Transfer to GPU
            descriptors = descriptors.to(device)
            angles = angles.to(device)
            labels = labels.to(device)
            labels = torch.log(labels + 1)

            optimizer.zero_grad()
            outputs = logModel((descriptors, angles))
            loss = criterion(outputs.squeeze(1), labels.float()).to(device)
            print("Train #", epoch, id, loss)

            loss.backward()
            optimizer.step()
            id += 1
            if id > 100:
                break
        print((datetime.datetime.now() - start) / training_generator.batch_size)


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