import torch

class DisneyBlock(torch.nn.Module):
    def __init__(self, oD, zD, outD):
        """
        :param oD: Dimension of output of the previous block
        :param zD: Dimension of one layer of hierarchical descriptor
        :param outD: Dimension of ouput of current block
        """
        super(DisneyBlock, self).__init__()

        self.f1z = torch.nn.Linear(zD, outD, bias=True)
        self.f1o = torch.nn.Linear(oD, outD, bias=True)
        self.f2 = torch.nn.Linear(outD, outD, bias=True)

        self.activation = torch.nn.ReLU()

    def forward(self, o, z):
        """
        :param o: output of the previous block
        :param z: one layer of hierarchical descriptor
        :param out: ouput of current block
        """

        out = self.f1o(o).add_(self.f1z(z))
        out = self.activation(out)

        out = self.f2(out)
        out = torch.add(out, o)
        out = self.activation(out)

        return out