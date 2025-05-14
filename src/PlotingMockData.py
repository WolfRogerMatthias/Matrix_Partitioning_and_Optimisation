import matplotlib.pyplot as plt
from src.MockDataGenerator import MockDataGenerator
from src.OptimizeAlgoApplied import OptimizeAlgoApplied
from src.GreedyAlgo import GreedyAlgo
from src.GreedyAlgoExtended import GreedyAlgoExtended
from src.GreedyAlgoDynamic import GreedyAlgoDynamic
from src.GreedyAlgoDynamicExtended import GreedyAlgoDynamicExtended
import numpy as np
import os
import time
import h5py

MockDataGenerator = MockDataGenerator()
OptimizeAlgoApplied = OptimizeAlgoApplied()
GreedyAlgo = GreedyAlgo(OptimizeAlgoApplied)
GreedyAlgoExtended = GreedyAlgoExtended(OptimizeAlgoApplied)
GreedyAlgoDynamic = GreedyAlgoDynamic(OptimizeAlgoApplied)
GreedyAlgoDynamicExtended = GreedyAlgoDynamicExtended(OptimizeAlgoApplied)


