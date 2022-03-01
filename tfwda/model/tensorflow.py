import abc
import tensorflow.python.keras.engine.functional


class IFModel(metaclass=abc.ABCMeta):
    """Interface for Model which holds the relevant information on
    the neural network model/architecture
    """
    pass


class Model(IFModel):
    """Model holds all tensorflow architecture relevant information, e.g. like summary,
    weights and the architecture structure itself

    Parameters
    ----------
        name         : str
            Name of the architecture
        architecture : tf.python.keras.engine.functional.Functional
            Tensorflow/Keras model

    Attributes
    ----------
        summary : method
            Prints the relevant structure of the architecture
        weights : list
            List of all layer variables and weights
    """


    def __init__(self, name: str, architecture: tensorflow.python.keras.engine.functional.Functional) -> None:
        self.name         = name
        if not isinstance(architecture, tensorflow.python.keras.engine.functional.Functional):
            raise TypeError('The architecture variable has to be of type tf.python.keras.engine.functional.Functional!')
        self.architecture = architecture
        self.summary      = self.architecture.summary
        self.weights      = self.architecture.weights