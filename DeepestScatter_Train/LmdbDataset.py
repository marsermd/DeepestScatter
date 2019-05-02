from lmdb import Environment

import os

from PythonProtocols.SceneSetup_pb2 import SceneSetup
from PythonProtocols.ScatterSample_pb2 import ScatterSample
from PythonProtocols.DisneyDescriptor_pb2 import DisneyDescriptor
from PythonProtocols.Result_pb2 import Result


class LmdbDataset:
    def __init__(self, databasePath, readonly=True):
        self.databasePath = databasePath
        self.readonly = readonly
        self.__init()


    def __init(self):
        create = not self.readonly

        self.env = Environment(
            self.databasePath,
            map_size=3e9,
            subdir=False,
            max_dbs=64,
            mode=0,
            create=create,
            readonly=self.readonly
        )
        self.descriptorToDb = {}
        self.nextIds = {}
        self.scenes_db = self.__addDb(SceneSetup, create=create)
        self.scatter_db = self.__addDb(ScatterSample, create=create)
        self.scatter_db = self.__addDb(DisneyDescriptor, create=create)
        self.results_db = self.__addDb(Result, create=create)

    def __addDb(self, protocol, create):
        db = self.env.open_db(protocol.DESCRIPTOR.name.encode('ascii'), integerkey=True, create=create)
        self.descriptorToDb[protocol.DESCRIPTOR.name] = db
        self.nextIds[protocol.DESCRIPTOR.name] = 0
        return db

    def append(self, value):
        name, db = self.__getNameAndDb(value)

        with self.env.begin(write=True) as transaction:
            transaction.put((self.nextIds[name]).to_bytes(4, 'little'), value.SerializeToString(), db=db)
            self.nextIds[name] += 1

    def getCountOf(self, protocolType):
        name, db = self.__getNameAndDb(protocolType)

        with self.env.begin() as transaction:
            return transaction.stat(db)['entries']

    def get(self, protocolType, id):
        _, db = self.__getNameAndDb(protocolType)

        with self.env.begin() as transaction:
            serialized = transaction.get(id.to_bytes(4, 'little'), db=db)

            protocol = protocolType()

            protocol.ParseFromString(serialized)
            return protocol

    def __getNameAndDb(self, protocolType):
        name = protocolType.DESCRIPTOR.name
        db = self.env.open_db(name.encode('ascii'), integerkey=True, create=False)
        return (name, db)

    def __getstate__(self):
        state = {
            "databasePath": self.databasePath,
            "readonly": self.readonly
        }

        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        self.__init()

TRAIN_NAME = "Train.lmdb"
VALIDATION_NAME = "Validation.lmdb"
TEST_NAME = "Test.lmdb"
class LmdbDatasets:
    def __init__(self, datasetRoot, readonly=True):
        self.datasetRoot = datasetRoot
        self.readonly = readonly

        self.__initDatasets()

    def __initDatasets(self):
        self.train = self.__createDataset(TRAIN_NAME)
        #self.validation = self.__createDataset(VALIDATION_NAME)
        #self.test = self.__createDataset(TEST_NAME)

    def __createDataset(self, name):
        return LmdbDataset(os.path.join(self.datasetRoot, name), self.readonly)

    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove the unpicklable entries.
        del state['train']
        del state['validation']
        del state['test']

        return state


    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        self.__initDatasets()
