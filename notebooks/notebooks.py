%load_ext autoreload
%autoreload 2
import sys
if "../.." not in sys.path:
    sys.path.append("../..")

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import signature.tensor_algebra as ta
from simulation.diffusion import Diffusion

import jax.numpy as jnp
import numpy as np

