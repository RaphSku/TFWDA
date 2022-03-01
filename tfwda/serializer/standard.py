import abc
import numpy as np
import collections
from abc import abstractmethod
from typing import Tuple


import tfwda.model.tensorflow
import tfwda.logger.standard 


class IFSerializer(metaclass = abc.ABCMeta):
    """Interface for the serializer which flattens the weights of the
    model which is provided
    
    Methods (abstract)
    ------------------
        model : model.tensorflow.IFModel
            The model whose weights should be serialized
    """


    @abstractmethod
    def flatten(self, model: tfwda.model.tensorflow.IFModel):
        """Flattens the model weights
        
        Parameters
        ----------
            model : model.tensorflow.IFModel
                Model which represents a neural network architecture and holds the weights of it
        """
        pass


class Serializer(IFSerializer):
    """The Serializer flattens the weights of the model and extracts information as metadata

    Parameters
    ----------
        logger : logger.standard.Logger
            Logger instance which is responsible for logging    
    """


    def __init__(self, logger: tfwda.logger.standard.Logger):
        self.logger = logger


    def flatten(self, model: tfwda.model.tensorflow.Model) -> Tuple[list, collections.OrderedDict]:
        """The weights of the model are extracted and serialized, also the metdata is extracted

        Parameters
        ----------
            model : model.tensorflow.Model
                Model with weights which should be flattened

        Returns
        -------
            list, collections.OrderedDict
                A list which contains the serialized weights and an ordered dictionary with all the meta-information
                is returned
        """
        serialized_weights = []
        metadata           = collections.OrderedDict({'names': [], 'shapes': [], 'dtypes': []})
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