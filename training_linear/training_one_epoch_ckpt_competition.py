import torch
from torch.nn import ModuleList
from utils.utils import AverageMeter,warmup_learning_rate, accuracy, save_model
import sys
import time
import numpy as np
from config.linear import parse_option
from utils.utils_competition import set_loader_competition, set_model_competition_first, set_optimizer, adjust_learning_rate, accuracy_multilabel
from sklearn.metrics import average_precision_score,roc_auc_score, classification_report
import pandas as pd