import numpy as np
import abc
from abc import ABCMeta, abstractmethod

from tfwda.model.tensorflow import Model
from tfwda.logger.standard  import Logger

class IF_Serializer(metaclass=ABCMeta):
    @abstractmethod
    def flatten(self, model: Model):
        pass


class Serializer(IF_Serializer):
    def __init__(self, logger: Logger):
        self.logger = logger


    def flatten(self, model: Model):
        serialized_weights = []
        for weight in model.weights:
            name  = weight.name
            shape = weight.shape
            dtype = np.dtype(weight.dtype.as_numpy_dtype).name
            value = weight.numpy()
            serialized_weights.append(value.flatten())
        return serialized_weights