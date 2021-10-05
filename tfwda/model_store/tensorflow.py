import tensorflow as tf
import abc
from abc import ABCMeta, abstractmethod

from model.tensorflow import Model


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
    
    def __init__(self):
        if ModelStore.__instance != None:
            raise Exception("An instance has been already created! Get it with cls.get_instance()!")
        else:
            ModelStore.__instance = self


    @staticmethod
    def get_instance():
        if ModelStore.__instance == None:
            ModelStore()
        return ModelStore.__instance