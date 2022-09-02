import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from omegaconf import DictConfig
from joblib import Parallel, delayed
from sklearn import metrics
from torch.nn import functional as F
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
import math
import os
import random
from attrdict import AttrDict
import gc

from sklearn.metrics import mean_squared_error

import wandb

import time

from torch.utils.data import DataLoader, Dataset

from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs

from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout, ContextPooler

import warnings
warnings.filterwarnings("ignore")