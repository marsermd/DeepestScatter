import torch

class DisneyBlock(torch.nn.Module):
    def __init__(self, oD, zD, outD):
        """
        :param oD: Dimension of output of the previous block
        :param zD: Dimension of one layer of hierarchical descriptor concatenated with an angle between light and view directions
        :param outD: Dimension of ouput of current block
        """
        super(DisneyBlock, self).__init__()

        self.f1 = torch.nn.Linear(zD + oD, outD, bias=True)
        self.f2 = torch.nn.Linear(outD, outD, bias=True)

        self.reLU = torch.nn.ReLU()

    def forward(self, o, z):
        """
        :param o: output of the previous block
        :param z: one layer of hierarchical descriptor concatenated with an angle between light and view directions
        :param out: ouput of current block
        """
        out = torch.cat((o, z), dim=1)

        out = self.f1(out)
        out = self.reLU(out)

        out = self.f2(out)
        out = torch.add(out, o)
        out = self.reLU(out)

        return out