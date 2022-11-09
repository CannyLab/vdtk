import itertools
import logging
import time

import click
import nltk
import numpy as np
import rich
from mpire import WorkerPool
from rich.progress import track

from vdtk.data_utils import load_dataset
