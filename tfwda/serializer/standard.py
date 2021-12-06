import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Tuple
from collections import OrderedDict

from tfwda.model.tensorflow import Model
from tfwda.logger.standard  import Logger


class IFSerializer(metaclass=ABCMeta):
    @abstractmethod
    def flatten(self, model: Model):
        pass


class Serializer(IFSerializer):
    """
    The Serializer serialized the weights of the model and extracts some info as metadata.
    Input:
        - logger : Logger -> An instance of the logger class    
    """


    def __init__(self, logger: Logger) -> None:
        self.logger = logger


    def flatten(self, model: Model) -> Tuple[list, OrderedDict]:
        """ The weights of the model are extracted and serialized, also the metdata is extracted. """
        serialized_weights = []
        metadata           = OrderedDict({'names': [], 'shapes': [], 'dtypes': []})
        for weight in model.weights:
            name  = weight.name
            shape = weight.shape
            dtype = np.dtype(weight.dtype.as_numpy_dtype).name
            value = weight.numpy()
            self.logger.log(f"{name} of shape {shape} and type {dtype} is flattened now...", "Info")
            serialized_weights.append(value.flatten())
            metadata['names'].append(name)
            metadata['shapes'].append(shape)
            metadata['dtypes'].append(dtype)
        return serialized_weights, metadata