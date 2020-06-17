import os
import re
from typing import List

import pandas as pd
import numpy as np
import nltk
import pickle


def load_datasets(files: [List[str], str] = "__all__") -> dict:
    """
    Loads all, a subset or a specific file and return them to a dictionary with the file name as key and
    the loaded Dataframe
    :param files:
    :return:
    """
    pass