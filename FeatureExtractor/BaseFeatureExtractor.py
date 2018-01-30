from gulpio.loader import DataLoader
import numpy as np
import pandas as pd

class BaseFeatures:

  def computeFeatures(self, frames_batch):
    raise NotImplementedError


