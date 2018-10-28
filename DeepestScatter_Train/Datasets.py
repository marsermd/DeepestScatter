from lmdb import Environment

import os

from PythonProtocols.SceneSetup_pb2 import SceneSetup
from PythonProtocols.ScatterSample_pb2 import ScatterSample
from PythonProtocols.Result_pb2 import Result

TRAIN_NAME = "Train.lmdb"
VALIDATION_NAME = "Validation.lmdb"
TEST_NAME = "Test.lmdb"

TB_1 = 1099511627776

class Dataset:

    def __init__(self, databasePath, readonly=True):
        create = not readonly
        self.env = Environment(databasePath, map_size=TB_1, subdir=False, max_dbs=64, mode=0, create=create, readonly=readonly)
        self.descriptorToDb = {}
        self.nextIds = {}
        self.scenes_db = self.__addDb(SceneSetup, create=create)
        self.scatter_db = self.__addDb(ScatterSample, create=create)
        self.results_db = self.__addDb(Result, create=create)


    def __addDb(self, protocol, create):
        db = self.env.open_db(protocol.DESCRIPTOR.full_name.encode('ascii'), integerkey=True, create=create)
        self.descriptorToDb[protocol.DESCRIPTOR.full_name] = db
        self.nextIds[protocol.DESCRIPTOR.full_name] = 0
        return db

    def append(self, value):
        name = value.DESCRIPTOR.full_name
        db = self.env.open_db(value.DESCRIPTOR.full_name.encode('ascii'), integerkey=True, create=False)

        with self.env.begin(write=True) as transaction:
            transaction.put((self.nextIds[name]).to_bytes(4, 'little'), value.SerializeToString(), db=db)
            self.nextIds[name] += 1

class Datasets:

    def __init__(self, datasetRoot, readonly=True):
        self.train = Dataset(os.path.join(datasetRoot, TRAIN_NAME), readonly)
        self.validation = Dataset(os.path.join(datasetRoot, VALIDATION_NAME), readonly)
        self.test = Dataset(os.path.join(datasetRoot, TEST_NAME), readonly)
