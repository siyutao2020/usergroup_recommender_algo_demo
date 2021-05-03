import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, config_enumerate, TraceEnum_ELBO, infer_discrete
import pyro.distributions as dist

import matplotlib.pyplot as plt

from pyro.poutine import block

torch.linalg.norm()