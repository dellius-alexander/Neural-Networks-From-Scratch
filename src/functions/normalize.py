#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Annotated, List, Union

import numpy as np
import pandas as pd


Dataset = Annotated[Union[np.ndarray, pd.DataFrame, List], "A pandas DataFrame, numpy array, or list of data"]

# Normalize the dataset
def normalize_data(data: Dataset) -> Dataset:
    """
    Normalize the data using the mean and standard deviation.
    This is useful when the data has different scales.
    :param data: ``Dataset``: The data to normalize
    :return:
    """
    if isinstance(data, pd.DataFrame):
        return (data - data.mean()) / data.std()
    if isinstance(data, np.ndarray):
        return (data - data.mean()) / data.std()
    if isinstance(data, list):
        return (np.array(data) - np.mean(data)) / np.std(data)
