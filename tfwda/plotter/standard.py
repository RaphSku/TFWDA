import abc
import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from abc import abstractmethod
from typing import Callable, Sequence


import tfwda.logger.standard


class IFPlotter(metaclass = abc.ABCMeta):
    """Interface for the Plotter
    
    Methods (abstract)
    ------------------
        plot(model_name str, flattened_weight list[np.ndarray], metadata collections.OrderedDict)
            Responsible for plotting
    """


    @abstractmethod
    def plot(self, model_name: str, flattened_weight: list[np.ndarray], metadata: collections.OrderedDict):
        """Plotting the flattened model weights
        
        Parameters
        ----------
            model_name       : str
                Name of the model
            flattened_weight : list[np.ndarray]
                Flattened weights of the model
            metadata         : collections.OrderedDict
                Metadata of the weights
        """
        pass


class Plotter(IFPlotter):
    """The Plotter takes the weights as input and generates the corresponding histograms.
    
    Parameters
    ----------
        logger      : logger.standard.Logger
            Logger instance
        path_to_dir : str
            Directory where the plots are stored to
    """


    def __init__(self, logger: tfwda.logger.standard.Logger, path_to_dir: str):
        self.logger = logger
        if not os.path.exists(path_to_dir):
            self.logger.log(f"The folder {path_to_dir} does not exist, if you want to continue, press Y, if not, press N...", "Warning")
            confirmation = input()
            if not confirmation in ["Y", "Yes", "yes"]:
                exit()
            os.mkdir(path_to_dir)
        self.path_to_dir = path_to_dir
    
    
    def plot(self, model_name: str, flattened_weight: list[np.ndarray], metadata: collections.OrderedDict) -> None:
        """Plotting the weights and storing these to the location given by `path_to_dir`
        
        Parameters
        ----------
            model_name : str
                Name of the model
            flattened_weight : list[np.ndarray]
                The flattened weights of the model
            metadata : collections.OrderedDict
                Metdata of the weights
        """
        for weight, name, shape, dtype in zip(flattened_weight, metadata['names'], metadata['shapes'], metadata['dtypes']):
            df  = pd.DataFrame({'weight': weight})
            fig = px.histogram(df, x = "weight", labels = {'x': "Weights", 'y': "Frequency"})
            
            name = name.replace("/", "_")
            name = name.replace(":", "_")
            fig.write_image(f"{self.path_to_dir}/{model_name}_{name}_{shape}_{dtype}.png")


    @staticmethod
    def display_hist(weight: list) -> None:
        """Is used for plotting a histogram weight plot, the plot is only display
        and not stored
        
        Parameters
        ----------
            weight : list
                A layer of a neural network contains N weights,
                these weights are given as a list of values
        """
        df  = pd.DataFrame({'weight': weight})
        fig = px.histogram(df, x = "weight", labels = {'x': "Weights", 'y': "Frequency"})
        fig.show()


    @staticmethod
    def display_fit(weight: list, func: Callable, number_of_bins: Sequence, popt: Sequence) -> None:
        """The weight distribution is fitted and the fit is plotted
        with the help of its optimized parameters

        Parameters
        ----------
            weight : list
                The respective weight which was fitted
            func   : Callable
                The fit function
            x      : Sequence
                The input or range of the fit
            popt   : Sequence
                The fit parameters
        """
        x       = np.linspace(np.min(weight), np.max(weight), 1000)
        fig, ax = plt.subplots(figsize = (10, 6))
        ax.hist(x = weight, bins = number_of_bins, label = "Weight Distribution")
        ax.plot(x, func(x, *popt), label = "Fit")
        fig.show()