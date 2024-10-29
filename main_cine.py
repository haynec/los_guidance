from quadsim.ptr_mc import PTR_main as PTR_main_mc
from quadsim.plotting import plot_ral_mc_results, plot_ral_mc_timing_results
from quadsim.config import (
    SimConfig,
    ScpConfig,
    ObsConfig,
    VpConfig,
    RacingConfig,
    WarmConfig,
    Config,
)
from quadsim.params.cinema_vp import params
from quadsim.params.cinema_vp_nodal import params as params_nodal

params = Config.from_config(params, savedir="results/")
params_nodal = Config.from_config(params_nodal, savedir="results/")

import pickle
import warnings
import numpy as np
from copy import deepcopy
warnings.filterwarnings("ignore")
###############################
# Author: Chris Hayner
# Autonomous Controls Laboratory
################################
print('Starting Quadrotor Simulator')
nodes = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

CT_LoS_vio_seq = []
CT_LoS_vio_node_seq = []
CT_runtime_seq = []
CT_iters = []
CT_obj = []
CT_mc_results = []

for i in range(len(nodes)):
    # Print the current node
    print('Current number of Nodes: ', nodes[i])
    params_test = deepcopy(params)
    params_test.scp.n = nodes[i]

    _, LoS_vio, CT_LoS_vio_node, runtime, iters, obj, mc_results = PTR_main_mc(params_test, True)
    CT_iters.append(iters)
    CT_LoS_vio_seq.append(LoS_vio)
    CT_LoS_vio_node_seq.append(CT_LoS_vio_node)
    CT_runtime_seq.append(runtime)
    CT_obj.append(obj)
    CT_mc_results.append(mc_results)
    del params_test


DT_LoS_vio_seq = []
DT_LoS_vio_node_seq = []
DT_runtime_seq = []
DT_iters = []
DT_obj = []
DT_mc_results = []

for i in range(len(nodes)):
    # Print the current node
    print('Current number of Nodes: ', nodes[i])
    params_test_nodal = deepcopy(params_nodal)
    params_test_nodal.scp.n = nodes[i]

    _, LoS_vio, LoS_vio_node, runtime, iters, obj, mc_results = PTR_main_mc(params_test_nodal, True)
    DT_iters.append(iters)
    DT_LoS_vio_seq.append(LoS_vio)
    DT_LoS_vio_node_seq.append(LoS_vio_node)
    DT_runtime_seq.append(runtime)
    DT_obj.append(obj)
    DT_mc_results.append(mc_results)
    del params_test_nodal


# Save MC results
with open('results/CT_cine_mc_results.pickle', 'wb') as f:
    pickle.dump(CT_mc_results, f)
with open('results/DT_cine_mc_results.pickle', 'wb') as f:
    pickle.dump(DT_mc_results, f)
    

# Load the mc results
with open('results/CT_cine_mc_results.pickle', 'rb') as f:
    CT_mc_results = pickle.load(f)
with open('results/DT_cine_mc_results.pickle', 'rb') as f:
    DT_mc_results = pickle.load(f)

# Plot the results
plot_ral_mc_results(CT_mc_results, DT_mc_results, nodes, params, False, True)
plot_ral_mc_timing_results(CT_mc_results, DT_mc_results, nodes, params, showlegend=True)