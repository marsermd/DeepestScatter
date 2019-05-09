import abc
from abc import abstractmethod

from torch.utils import data

from BakedDescriptor_pb2 import BakedDescriptor
from PythonProtocols.SceneSetup_pb2 import SceneSetup
from PythonProtocols.ScatterSample_pb2 import ScatterSample
from PythonProtocols.DisneyDescriptor_pb2 import DisneyDescriptor
from PythonProtocols.Result_pb2 import Result
from GlobalSettings import BATCH_SIZE

class BaseDataset(data.Dataset):
    __metaclass__ = abc.ABCMeta

    def __init__(self, lmdbDataset, mainProtocolType):
        self.lmdbDataset = lmdbDataset
        self.length = self.lmdbDataset.getCountOf(mainProtocolType)
        self.cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.__sampleId = index
        self.__sceneId = index // BATCH_SIZE
        self.cache = {}
        return self.__doGetItem__()

    @abstractmethod
    def __doGetItem__(self):
        pass

    def getScene(self):
        return self.__getProtocol(SceneSetup, self.__sceneId)

    def getDisneyDescriptor(self):
        return self.__getProtocol(DisneyDescriptor)

    def getBakedDescriptor(self):
        return self.__getProtocol(BakedDescriptor)

    def getResult(self):
        return self.__getProtocol(Result)

    def getScatterSample(self):
        return self.__getProtocol(ScatterSample)

    def __getProtocol(self, protocolType, id=None):
        if id is None:
            id = self.__sampleId

        if protocolType in self.cache.keys():
            return self.cache[protocolType]

        res = self.lmdbDataset.get(protocolType, id)
        self.cache[protocolType] = res
        return res

    def __getstate__(self):
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['cache']         # remove cache entry, since protobuf objects can't be pickled
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)   # update attributes
        self.cache = {}            # reset cache
