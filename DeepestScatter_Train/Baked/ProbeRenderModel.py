import torch


class ProbeRedererModel(torch.nn.Module):
    META_INFO_DIMENSION = 5

    def __init__(self, lightProbeDimension):
        """
        :param lightProbeDimension: length of 1d vector, which will be representing the light probe
        """
        super(ProbeRedererModel, self).__init__()

        inputDimension = lightProbeDimension + self.META_INFO_DIMENSION

        self.fullyConnected = torch.nn.Sequential(
            torch.nn.Linear(inputDimension, inputDimension),
            torch.nn.ReLU(),
            torch.nn.Linear(inputDimension, inputDimension),
            torch.nn.ReLU(),
            torch.nn.Linear(inputDimension, 1),
            torch.nn.LeakyReLU()
        )

    def forward(self, lightProbeWithMetaInfo):
        """
        :param lightProbeWithMetaInfo: light probe concatenated with an two angles and one offset vector
        :return: ouput of current block
        """

        return self.fullyConnected(lightProbeWithMetaInfo)