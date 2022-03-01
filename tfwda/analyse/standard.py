import numpy as np
import abc
import collections
import scipy.stats
from abc import abstractmethod


import tfwda.logger.standard
import tfwda.plotter.standard 


class IFAnalyser(metaclass = abc.ABCMeta):
    pass


class Analyser(IFAnalyser):
    def __init__(self, plotter: tfwda.plotter.standard.Plotter, logger: tfwda.logger.standard.Logger) -> None:
        self.plotter     = plotter
        self.logger      = logger


    def process(self, data):
        weights  = data["weights"]
        metadata = data["metadata"]

        extracted_properties = collections.OrderedDict({'names': None, 'shapes': None, 'dtypes': None, 'min': [], 'max': [], 'mean': [], '25-quantile': [], 'median': [],
                                                        '75-quantile': [], 'IQR': [], 'mode': [], 'variance': [], 'skewness': [], 'kurtosis': [], 'MAD': []})
        for weight in weights:
            extracted_properties['min'].append(float(np.min(weight)))
            extracted_properties['max'].append(float(np.max(weight)))
            extracted_properties['mean'].append(float(np.mean(weight)))
            extracted_properties['25-quantile'].append(float(np.quantile(weight, q = 0.25)))
            extracted_properties['median'].append(float(np.median(weight)))
            extracted_properties['75-quantile'].append(float(np.quantile(weight, q = 0.75)))
            extracted_properties['IQR'].append(float(np.quantile(weight, q = 0.75) - np.quantile(weight, q = 0.25)))

            bin_frequencies, bins = np.histogram(weight, bins = 'auto')
            max_indices           = np.where(bin_frequencies == np.max(bin_frequencies))
            centralized_bins      = bins[:-1] + np.diff(bins) / 2
            mode                  = centralized_bins[max_indices]
            for mod in mode:
                extracted_properties['mode'].append(float(mod))
            extracted_properties['variance'].append(float(np.var(weight)))
            extracted_properties['skewness'].append(float(scipy.stats.skew(weight)))
            extracted_properties['kurtosis'].append(float(scipy.stats.kurtosis(weight)))
            extracted_properties['MAD'].append(float(scipy.stats.median_abs_deviation(weight, scale = "normal")))
        extracted_properties['names']  = metadata["names"]
        extracted_properties['shapes'] = []
        for tensor_shape in metadata["shapes"]:
            extracted_properties['shapes'].append(tensor_shape.as_list())
        extracted_properties['dtypes'] = []
        for numpy_dtype in metadata["dtypes"]:
            type_variable = np.dtype(numpy_dtype)
            extracted_properties['dtypes'].append(type_variable.name)

        return extracted_properties