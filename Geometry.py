import numpy as np
import tqdm
import json
import glob
import pyvista as pv
import os
from pathlib import Path
import math
from os.path import join
import itertools
from scipy.spatial import KDTree


def searchForClosestPoints(sourceVs, targetVs, tree=None, returnVId = False):
    if tree is None:
        tree = KDTree(targetVs)

    closestPts = []
    dis = []
    for sv in sourceVs:

        d, tvId = tree.query(sv)
        if returnVId:
            closestPts.append(tvId)
        else:
            closestPts.append(targetVs[tvId, :])
        dis.append(d)
    return np.array(closestPts), np.array(dis)