import abc
import tensorflow as tf
import pymongo
import collections
from abc import abstractmethod

import model.tensorflow 
import logger.standard    
import serializer.standard 
import plotter.standard    
import fitter.standard    
import utils.errors   


class IFModelStore(metaclass = abc.ABCMeta):
    """Interface for every model store, the model store
    coordinates all the communication between the components and
    will command logging and database storage of important information

    Methods (abstract)
    ------------------
        pipe_models(models list[model.tensorflow.IFModel])
            A list of models will be piped into the model store and the model
            store will delegate these models to the different components
        get_instance(db_connection_string str, database_name str) IFModelStore
            Instance of the model store is either initialised or fetched
    """
    @abstractmethod
    def pipe_models(self, models: list[model.tensorflow.IFModel]):
        pass

    
    @staticmethod
    @abstractmethod
    def get_instance(db_connection_string: str, database_name: str) -> object:
        pass


class ModelStore(IFModelStore):
    """
    The model store manages all the models and logs all operations which were 
    performed on them. The model store is implemented as a Singleton since it
    is represents the central point of model traffic. The fitting, plotting and 
    serializing are all controlled by the model store.

    Parameters
    ----------
        db_connection_string : str
            Database Connection String for MongoDB
        database_name        : str
            The name of the database of interest
        path_to_dir          : str
            The Path to the directory where the plots should be deposited
        verbosity            : bool
            The verbosity of the information which are given to the user over console

    Attributes
    ----------
        logger     : logger.standard.Logger
            Logging instance which logs all the information to the console
        serializer : serializer.standard.Serializer
            Serializer instance which is responsible for flattening the neural network model
        plotter    : plotter.standard.Plotter
            Plotting instance which will perform all the distribution plots
        fitter     : fitter.standard.Fitter
            Fitting instance which will fit different distribution if the user would like
            to perform a fit on given weight distributions

    Methods
    -------
        get_instance(db_connection_string str, database_name str, path_to_dir str, verbosity bool) ModelStore `staticmethod`
            Since the model store is a Singleton, this method must be use to fetch an instance, this will be
            the central instance, coordinating and speaking to all the other components
        pipe_models(models list[model.tensorflow.Model])
            A list of tensorflow models will be serialized, plotted and fitted
    """
    __instance = None
    __client   = None
    __database = None
    logger     = None
    serializer = None
    plotter    = None
    

    def __init__(self, db_connection_string: str, database_name: str, path_to_dir: str, verbosity: bool) -> None:
        if ModelStore.__instance != None:
            raise utils.errors.NotCreatedInstanceError("An instance has been already created! Get it with ModelStore.get_instance()!")
        ModelStore.__instance = self
        if self.__database == None:
            self.__setup_db_connection(db_connection_string, database_name)
            self.logger     = logger.standard.Logger(verbosity)
            self.serializer = serializer.standard.Serializer(self.logger)
            self.plotter    = plotter.standard.Plotter(self.logger, path_to_dir)
            self.fitter     = fitter.standard.Fitter(self.plotter, self.logger)


    @staticmethod
    def get_instance(db_connection_string: str, database_name: str, path_to_dir: str, verbosity: bool) -> object:
        """This class is a Singleton, thus the instance is returned through this method
        
        Parameters
        ----------
            db_connection_string : str
                MongoDB connection string
            database_name        : str
                Name of the target database
            path_to_dir          : str
                Path where the plots will be stored in
            verbosity            : bool
                Whether the info output to the console should be verbose or not

        Returns
        -------
            ModelStore
                Instance of the ModelStore class
        """
        if ModelStore.__instance == None:
            ModelStore(db_connection_string, database_name, path_to_dir, verbosity)
        return ModelStore.__instance


    def pipe_models(self, models: list[model.tensorflow.Model]) -> None:
        """A list of models is processed, meaning that weights are extracted, given to the serializer,
        metadata are written and the plots are generated 

        Parameters
        ----------
            models : list[model.tensorflow.Model]
                A list of tensorflow models which weights should be analysed
        """
        self.logger.log(f"{len(models)} models are being processed now!", "Header")
        serialized_weights_per_model = collections.OrderedDict()
        for count, model in enumerate(models):
            flattened_weights, metadata              = self.serializer.flatten(model)
            serialized_weights_per_model[model.name] = {'weights': flattened_weights, 'metadata': metadata}
            self.logger.log(f"{count} model serializations have been finished...", "Info")
        self.logger.log(f"{len(models)} models have been processed now...", "Info")

        self.logger.log("Plotting will start now!", "Header")
        for model_name, model_data in serialized_weights_per_model.items():
            self.plotter.plot(model_name, model_data["weights"], model_data["metadata"])
        self.logger.log("Plotting terminated successfully!", "Info")

        self.logger.log("Fitting process will start now!", "Header")
        for model_name, model_data in serialized_weights_per_model.items():
            self.fitter.fit(model_data)
        self.logger.log("Fitting process terminated successfully!", "Info")


    def __setup_db_connection(self, db_connection_string: str, database_name: str) -> None:
        """This method setups the database connection - pay attention: this method does
        not check whether a successful connection could be established!

        Parameters
        ----------
            db_connection_string : str
                MongoDB connection string
            database_name        : str
                Name of the target database
        """
        self.__client   = pymongo.MongoClient(db_connection_string)
        self.__database = self.__client[database_name]