import tensorflow as tf
import abc
from abc import abstractmethod


class IFModel(metaclass=abc.ABCMeta):
    pass


class Model(IFModel):
    """
    Model holds all architecture relevant information, e.g. like summary,
    weights and the architecture itself
    Input:
        - name          : str                                          -> Name of the architecture
        - architecture  : tf.python.keras.engine.functional.Functional -> Keras/Tensorflow model
    """


    def __init__(self, name: str, architecture: tf.python.keras.engine.functional.Functional) -> None:
        self.name         = name
        if not isinstance(architecture, tf.python.keras.engine.functional.Functional):
            raise TypeError('The architecture variable has to be of type tf.python.keras.engine.functional.Functional!')
        self.architecture = architecture
        self.summary      = self.architecture.summary()
        self.weights      = self.architecture.weights