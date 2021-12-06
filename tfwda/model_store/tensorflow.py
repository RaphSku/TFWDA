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
from tfwda.plotter.standard    import Plotter
from tfwda.utils.errors        import NotCreatedInstanceError


class IFModelStore(metaclass = ABCMeta):
    @abstractmethod
    def pipe_models(self, models : List[Model]):
        pass

    
    @staticmethod
    @abstractmethod
    def get_instance(db_connection_string : str, database_name : str):
        pass


class ModelStore(IFModelStore):
    """
    The model store manages all the models and logs all operations which were 
    performed on them. The model store is implemented as a Singleton since it
    is represents the central point of model traffic. The fitting, plotting and 
    serializing are all controlled by the model store.

    Input:
        - db_connection_string : str  -> Database Connection String
        - database_name        : str  -> The name of the database of interest
        - path_to_dir          : str  -> The Path to the directory where the plots should be deposited
        - verbosity            : bool -> The verbosity of the information which are given to the user
    """

    __instance = None
    __client   = None
    __database = None
    logger     = None
    serializer = None
    plotter    = None
    

    def __init__(self, db_connection_string: str, database_name: str, path_to_dir: str, verbosity: bool) -> None:
        if ModelStore.__instance != None:
            raise NotCreatedInstanceError("An instance has been already created! Get it with ModelStore.get_instance()!")
        else:
            ModelStore.__instance = self
        if self.__database == None:
            self.__setup_db_connection(db_connection_string, database_name)
            self.logger     = Logger(verbosity)
            self.serializer = Serializer(self.logger)
            self.plotter    = Plotter(self.logger, path_to_dir)


    @staticmethod
    def get_instance(db_connection_string: str, database_name: str, path_to_dir: str, verbosity: bool) -> object:
        """ This class is a Singleton, thus the instance is returned through this method. """
        if ModelStore.__instance == None:
            ModelStore(db_connection_string, database_name, path_to_dir, verbosity)
        return ModelStore.__instance


    def pipe_models(self, models: List[Model]) -> None:
        """ 
        A list of models is processed, meaning that weights are extracted, given to the Serializer,
        metadata are written and the plots are generated 
        """
        self.logger.log(f"{len(models)} models are being processed now!", "Header")

        count = 0
        serialized_weights_per_model = OrderedDict()
        for model in models:
            count += 1
            flattened_weights, metadata = self.serializer.flatten(model)
            serialized_weights_per_model[model.name] = {'weights': flattened_weights, 'metadata': metadata}
            self.logger.log(f"{count} model serializations have been finished...", "Info")

        self.logger.log(f"{len(models)} models have been processed now...", "Info")
        self.logger.log("Plotting will start now!", "Header")

        for model_name, model_data in serialized_weights_per_model.items():
            self.plotter.plot(model_name, model_data["weights"], model_data["metadata"])


    def __setup_db_connection(self, db_connection_string: str, database_name: str) -> None:
        """ 
        This method setups the database connection - pay attention: this method does
        not tell you if a successful connection could be established. 
        """
        self.__client   = MongoClient(db_connection_string)
        self.__database = self.__client[database_name]