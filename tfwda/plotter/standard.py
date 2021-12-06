import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
from abc import abstractmethod, ABCMeta
from typing import List
from collections import OrderedDict

from tfwda.logger.standard import Logger


class IFPlotter(metaclass=ABCMeta):
    @abstractmethod
    def plot(self, model_name: str, flattened_weight: List[np.ndarray], metadata: OrderedDict):
        pass


class Plotter(IFPlotter):
    """
    The Plotter takes the weights as input and generates the corresponding histograms.
    Input:
        - logger      : Logger -> A valid Logger instance
        - path_to_dir : str    -> Directory where the plots are stored to
    """


    def __init__(self, logger: Logger, path_to_dir: str):
        self.logger = logger
        if not os.path.exists(path_to_dir):
            self.logger.log(f"The folder {path_to_dir} does not exist, if you want to continue, press Y, if not, press N...", "Warning")
            confirmation = input()
            if not confirmation in ["Y", "Yes", "yes"]:
                exit()
            os.mkdir(path_to_dir)
        self.path_to_dir = path_to_dir
    
    
    def plot(self, model_name: str, flattened_weight: List[np.ndarray], metadata: OrderedDict):
        """ Plotting the weights and storing these to the corresponding location """
        for weight, name, shape, dtype in zip(flattened_weight, metadata['names'], metadata['shapes'], metadata['dtypes']):
            df  = pd.DataFrame({'weight': weight})
            fig = px.histogram(df, x = "weight", labels = {'x': "Weights", 'y': "Frequency"})
            
            name = str.replace("/", "_", name)
            name = str.replace(":", "_", name)
            fig.write_image(f"{self.path_to_dir}/{model_name}_{name}_{shape}_{dtype}.png")