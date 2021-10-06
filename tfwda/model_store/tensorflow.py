import tensorflow as tf
import pymongo
import abc
from abc import ABCMeta, abstractmethod
from pymongo import MongoClient
from typing import List
from collections import OrderedDict

from tfwda.model.tensorflow    import Model
from tfwda.logger.standard     import Logger
from tfwda.serializer.standard import Serializer


class IF_ModelStore(metaclass = ABCMeta):
    pass


class ModelStore(IF_ModelStore):
    """
    The model store manages all the models and logs all operations which were 
    performed on them. The model store is implemented as a Singleton since it
    is represents the central point of model traffic. The fitting, plotting and 
    serializing are all controlled by the model store.
    """

    __instance = None
    __client   = None
    __database = None
    logger     = None
    serializer = None
    
    def __init__(self, db_connection_string : str, database_name : str):
        if ModelStore.__instance != None:
            raise Exception("An instance has been already created! Get it with cls.get_instance()!")
        else:
            ModelStore.__instance = self
        if self.__database == None:
            self.__setup_db_connection(db_connection_string, database_name)
            self.logger     = Logger()
            self.serializer = Serializer(self.logger)


    @staticmethod
    def get_instance(db_connection_string : str, database_name : str):
        if ModelStore.__instance == None:
            ModelStore(db_connection_string, database_name)
        return ModelStore.__instance


    def pipe_models(self, models : List[Model]):
        self.logger.log(f"{len(models)} are being processed now...", "Header")

        count = 0
        serialized_weights_per_model = {}
        for model in models:
            count += 1
            serialized_weights_per_model[model.name] = self.serializer.flatten(model)
            self.logger.log(f"{count} model serializations have been finished...", "Info")

        self.logger.log(f"{len(models)} have been processed now...", "Info")


    def __setup_db_connection(self, db_connection_string : str, database_name : str):
        try:
            self.__client   = MongoClient(db_connection_string)
            self.__database = self.__client[database_name]
        except:
            print("Connection String is either invalid or cannot connect to database!")