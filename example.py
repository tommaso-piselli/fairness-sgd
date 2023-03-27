## custom
from utils import utils, vis
# from utils import poly_point_isect as bo   ##bentley-ottmann sweep line
import criteria as C
import quality as Q
# import gd2
from gd2 import GD2
import utils.weight_schedule as ws

import urllib.request
import io
import zipfile

## third party
import networkx as nx
# from PIL import Image
from natsort import natsorted

### numeric
import numpy as np
# import scipy.io as io
import torch
from torch import nn, optim
import torch.nn.functional as F

### vis
import tqdm
import matplotlib.pyplot as plt

# import matplotlib.animation as animation
# from matplotlib.colors import LinearSegmentedColormap
# from mpl_toolkits import mplot3d
# from matplotlib import collections  as mc
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
plt.style.use('ggplot')
plt.style.use('seaborn-v0_8')

## sys
from collections import defaultdict
import random
import time
from glob import glob
import math
import os
from pathlib import Path
import itertools
import pickle as pkl

device = 'cpu'
seed = 2337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
max_iter = int(1e4)

# Karate Club
graph_name = 'karate_club_graph'
G = nx.karate_club_graph()

# Binomial tree
# G = nx.binomial_tree(7)

# Random Graph
# n = 30
# p = 0.05
# G = nx.fast_gnp_random_graph(n, p, seed, directed=False)


# graph_name = 'grid2_dual'
# max_iter = int(1e4)
# mat_dir = 'input_graphs/SuiteSparse Matrix Collection'
# G = utils.load_mat(f'{mat_dir}/{graph_name}.mat')

# print("Test with STRESS+UNFAIRNESS")
criteria = ['stress', 'unfairness']
criteria_weights = dict(
    stress=ws.SmoothSteps([max_iter / 4, max_iter], [1, 0.5]),
    unfairness=ws.SmoothSteps([max_iter / 4, max_iter], [1, 0.5]),
)

# Stress Only
# print("Test with STRESS")
# criteria = ['stress']
# criteria_weights = dict(
#     stress=ws.SmoothSteps([max_iter / 4, max_iter], [1, 0.5]),
# )

# Unfairness Only
# print("Test with UNFAIRNESS")
# criteria = ['unfairness']
# criteria_weights = dict(
#     unfairness=ws.SmoothSteps([max_iter / 4, max_iter], [1, 0.5]),
# )

criteria = list(criteria_weights.keys())

##plot_weight(criteria_weights, max_iter)
##plt.close()


sample_sizes = dict(
    stress=16,
    unfairness=16,
    ideal_edge_length=16,
    neighborhood_preservation=16,
    crossings=128,
    crossing_angle_maximization=64,
    aspect_ratio=max(128, int(len(G) ** 0.5)),
    angular_resolution=16,
    vertex_resolution=max(256, int(len(G) ** 0.5)),
    gabriel=64,
)
sample_sizes = {c: sample_sizes[c] for c in criteria}

gd = GD2(G)

# Initial Printing
stress = C.stress(
    gd.pos, gd.D, gd.W,
    sample=None, reduce='mean')
print("Initial stress: ", stress.item())

unf = C.unfairness(
    gd.pos, gd.D, gd.W, gd.node_dic,
    sample=None, reduce='mean')
print("Initial unfairness: ", unf.item())
print(f'Shape Pos: {gd.pos.shape}, D: {gd.D.shape}, W: {gd.W.shape}')

result = gd.optimize(
    criteria_weights=criteria_weights,
    sample_sizes=sample_sizes,

    # evaluate='all',
    evaluate=set(criteria),

    max_iter=max_iter,
    time_limit=3600,  ##sec

    evaluate_interval=max_iter, evaluate_interval_unit='iter',
    vis_interval=-1, vis_interval_unit='sec',

    clear_output=True,
    grad_clamp=20,
    criteria_kwargs=dict(
        aspect_ratio=dict(target=[1, 1]),
    ),
    #optimizer_kwargs = dict(mode='Adam', lr=1),
    optimizer_kwargs=dict(mode='SGD', lr=2),
    scheduler_kwargs=dict(verbose=True),
    node_dic=None,
)

## output 
pos = gd.pos.detach().numpy().tolist()
pos_G = {k: pos[gd.k2i[k]] for k in gd.G.nodes}

# print('nodes')
# for node_id, pos in pos_G.items():
#    print(f'{node_id}, {pos[0]}, {pos[1]}')

# print('edges')
# for e in gd.G.edges:
#    print(f'{e[0]}, {e[1]}')

# Final Printing
st = C.stress(
            gd.pos, gd.D, gd.W,
            sample=None, reduce='mean')
print('final stress: ', st.item())

unf = C.unfairness(
    gd.pos, gd.D, gd.W, gd.node_dic,
    sample=None, reduce='mean')
print("final unfairness: ", unf.item())

## vis
vis.plot(
    gd.G, pos_G,
    [gd.iters, gd.loss_curve],
    result['iter'], result['runtime'],
    criteria_weights, max_iter,
    # show=True, save=False,
    node_size=14,
    edge_width=1,
    node_color=gd.color_list,
)
plt.show()

