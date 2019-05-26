import torch

from DisneyBlock import DisneyBlock


class ProbeRendererModel(torch.nn.Module):
    DESCRIPTOR_LAYER_DIMENSION = 5 * 5 * 9
    DESCRIPTOR_LAYER_WITH_META_DIMENSION = DESCRIPTOR_LAYER_DIMENSION + 1

    def __init__(self, lightProbeDimension, blockCount):
        """
        :param lightProbeDimension: length of 1d vector, which will be representing the light probe
        """
        super(ProbeRendererModel, self).__init__()

        self.inputDimension = lightProbeDimension
        self.blockCount = blockCount

        self.inputFullyConnected = self.__createInputFullyConnected()
        self.blocks = self.__createBlocks()
        self.outputFullyConnected = self.__createOuputFullyConnected()

    def forward(self, lightProbe, disneyDescriptor):
        """
        :param lightProbe: light probe with an two angles and one offset vector
        :return: ouput of current block
        """

        # batchSize = disneyDescriptor.size()[0]
        # out = torch.zeros((batchSize, self.inputDimension))

        out = self.inputFullyConnected(lightProbe)

        for i, block in enumerate(self.blocks):
            out = block(out, disneyDescriptor.narrow(1, i, 1).squeeze(1))

        return self.outputFullyConnected(out)

    def __createInputFullyConnected(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.inputDimension, self.inputDimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.inputDimension, self.inputDimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.inputDimension, self.inputDimension),
            torch.nn.ReLU(),
        )

    def __createBlocks(self):
        return torch.nn.ModuleList([self.__createBlock() for i in range(self.blockCount)])

    def __createBlock(self):
        return DisneyBlock(
            self.inputDimension,
            self.DESCRIPTOR_LAYER_WITH_META_DIMENSION,
            self.inputDimension
        )

    def __createOuputFullyConnected(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.inputDimension, self.inputDimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.inputDimension, self.inputDimension),
            torch.nn.ReLU(),
            torch.nn.Linear(self.inputDimension, 1),
            torch.nn.LeakyReLU()
        )