#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:59:11 2019

@author: rbjv28
"""

import matplotlib.pyplot as plt  # Import library for direct plotting 
import numpy as np               # Import Numerical Python
from IPython.core.display import display, HTML #Import HTML for formatting output
from scipy.constants import h as C_h
from scipy.constants import e as C_e


# NOTE: Uncomment following lines ONLY if you are not using installation via pip
import sys, os
rootDir = '/home/homeblue01/8/rbjv28/My_Documents/ARC-Alkali-Rydberg-Calculator' # e.g. '/Users/Username/Desktop/ARC-Alkali-Rydberg-Calculator'

#rootDir  = '/home/erlizzard/Documents/Uni_Year_4/DARC/' # e.g. '/Users/Username/Desktop/ARC-Alkali-Rydberg-Calculator'
sys.path.insert(0,rootDir)

from arc import *
atom= StrontiumI()

calc = LevelPlot(atom)
calc.makeLevels(0,5,0,3,0)
calc.drawLevels()
calc.showPlot()
