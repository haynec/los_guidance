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
import pickle
import warnings
import numpy as np
from quadsim.params.dr_vp_nodal import params
warnings.filterwarnings("ignore")
###############################
# Author: Chris Hayner
# Autonomous Controls Laboratory
################################
print('Starting Quadrotor Simulator')
nodes = [22, 33, 44, 55, 66, 77, 88, 99, 110, 121, 132, 143]

gate_nodes_seq = []
gate_nodes_seq.append(np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]))
gate_nodes_seq.append(np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30]))
gate_nodes_seq.append(np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40]))
gate_nodes_seq.append(np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]))
gate_nodes_seq.append(np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 60]))
gate_nodes_seq.append(np.array([7, 14, 21, 28, 35, 42, 49, 56, 63, 70]))
gate_nodes_seq.append(np.array([8, 16, 24, 32, 40, 48, 56, 64, 72, 80]))
gate_nodes_seq.append(np.array([9, 18, 27, 36, 45, 54, 63, 72, 81, 90]))
gate_nodes_seq.append(np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
gate_nodes_seq.append(np.array([11, 22, 33, 44, 55, 66, 77, 88, 99, 110]))
gate_nodes_seq.append(np.array([12, 24, 36, 48, 60, 72, 84, 96, 108, 120]))
gate_nodes_seq.append(np.array([13, 26, 39, 52, 65, 78, 91, 104, 117, 130]))
gate_nodes_seq.append(np.array([14, 28, 42, 56, 70, 84, 98, 112, 126, 140]))

CT_LoS_vio_seq = []
CT_LoS_vio_node_seq = []
CT_runtime_seq = []
CT_iters = []
CT_obj = []
CT_mc_results = []

for i in range(len(nodes)):
    # Print the current node
    print('Current number of Nodes: ', nodes[i])

    total_time = 30.0  # Total time for the simulation
    sim = SimConfig(
        model="drone_6dof",
        initial_state=np.array([10, 0, 20, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # Initial State
        final_state=np.array([10, 0, 20, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # Final State
        initial_control=np.array([0, 0, 10, 0, 0, 0, 1]),  # Initial Control
        max_state=np.array(
            [200, 100, 50, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 1e-4]
        ),  # Upper Bound on the states
        min_state=np.array(
            [-200, -100, 15, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0]
        ),  # Lower Bound on the states
        max_control=np.array(
            [0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562, 3.0 * total_time]
        ),  # Upper Bound on the controls
        min_control=np.array(
            [0, 0, 0, -18.665, -18.665, -0.55562, 0.3 * total_time]
        ),  # Lower Bound on the controls
        max_dt=1e2,  # Maximum Time Step
        min_dt=1e-2,  # Minimum Time Step
        total_time=total_time,
    )
    scp = ScpConfig(
        n=nodes[i],  # Number of Nodes
        w_tr=2e0,  # Weight on the Trust Reigon
        lam_o=0e0,  # Weight on the Nodal Obstacle Avoidance (not used for CTCS)
        lam_vp=0e0,  # Weight on the Nodal Virtual Point Avoidance (not used for CTCS)
        lam_min=0e0,  # Weight on the Nodal Minimum Range Avoidance (not used for CTCS)
        lam_max=0e0,  # Weight on the Nodal Maximum Range Avoidance (not used for CTCS)
        lam_fuel=0e0,  # Weight on the Minimal Fuel Objective
        lam_t=2e-1,  # Weight on the Minimal Time Objective
        lam_vc=1e1,  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
        lam_vc_ctcs=1e1,  # Weight on the Virtual Control Objective for only CTCS
        ep_tr=1e-5,  # Trust Region Tolerance
        ep_vb=1e-4,  # Virtual Control Tolerance
        ep_vc=1e-8,  # Virtual Control Tolerance for CTCS
        ep_vc_ctcs=1e-8,  # Virtual Control Tolerance for CTCS
        free_final_drop=10,  # SCP iteration to relax minimal final time objective
        min_time_relax=0.8,  # Minimal Time Relaxation Factor
        w_tr_adapt=1.2,  # Trust Region Adaptation Factor
        w_tr_max_scaling_factor=1e2,  # Maximum Trust Region Weight
    )
    obs = ObsConfig()
    vp = VpConfig(
        n_subs=10,  # Number of subjects
        R_sb=np.array(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        ),  # Rotation Matrix from Sensor to Body Frame
        norm=2,  # Norm for the View cone
        alpha_x=6.0,  # Angle for the x-axis of Sensor Cone
        alpha_y=6.0,  # Angle for the y-axis of Sensor Cone
    )
    racing = RacingConfig(
        gate_centers=[
            np.array([59.436, 0.0000, 20.0000]),
            np.array([92.964, -23.750, 25.5240]),
            np.array([92.964, -29.274, 20.0000]),
            np.array([92.964, -23.750, 20.0000]),
            np.array([130.150, -23.750, 20.0000]),
            np.array([152.400, -73.152, 20.0000]),
            np.array([92.964, -75.080, 20.0000]),
            np.array([92.964, -68.556, 20.0000]),
            np.array([59.436, -81.358, 20.0000]),
            np.array([22.250, -42.672, 20.0000]),
        ],
        gate_nodes=gate_nodes_seq[i],
    )

    warm = WarmConfig()
    params = Config(sim=sim, scp=scp, obs=obs, vp=vp, racing=racing, warm=warm)

    _, LoS_vio, CT_LoS_vio_node, runtime, iters, obj, mc_results = PTR_main_mc(params, True)
    CT_LoS_vio_seq.append(LoS_vio)
    CT_LoS_vio_node_seq.append(CT_LoS_vio_node)
    CT_runtime_seq.append(runtime)
    CT_iters.append(iters)
    CT_obj.append(obj)
    CT_mc_results.append(mc_results)

# Save MC results
with open('results/dr_CT_mc_results.pickle', 'wb') as f:
    pickle.dump(CT_mc_results, f)

DT_LoS_vio_seq = []
DT_LoS_vio_node_seq = []
DT_runtime_seq = []
DT_iters = []
DT_obj = []
DT_mc_results = []

for i in range(len(nodes)):
    print('Current number of Nodes: ', nodes[i])
    total_time = 30.0  # Total time for the simulation
    sim = SimConfig(
        model="drone_6dof",
        initial_state=np.array([10, 0, 20, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # Initial State
        final_state=np.array([10, 0, 20, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # Final State
        initial_control=np.array([0, 0, 10, 0, 0, 0, 1]),  # Initial Control
        max_state=np.array(
            [200, 100, 200, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 1e-4]
        ),  # Upper Bound on the states
        min_state=np.array(
            [-200, -100, 15, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0]
        ),  # Lower Bound on the states
        max_control=np.array(
            [0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562, 3.0 * total_time]
        ),  # Upper Bound on the controls
        min_control=np.array(
            [0, 0, 0, -18.665, -18.665, -0.55562, 0.3 * total_time]
        ),  # Lower Bound on the controls
        max_dt=1e2,  # Maximum Time Step
        min_dt=1e-2,  # Minimum Time Step
        total_time=total_time,
    )
    scp = ScpConfig(
        n=nodes[i],  # Number of Nodes
        ctcs=False,  # Whether to use CTCS augmentation
        free_final_drop=10,  # SCP iteration to relax minimal final time objective
        min_time_relax=0.8,  # Minimal Time Relaxation Factor
        w_tr_adapt=1.05,  # Trust Region Adaptation Factor
        w_tr=8e1,  # Weight on the Trust Reigon
        lam_o=0e0,  # Weight on the Nodal Obstacle Avoidance (not used for CTCS)
        lam_vp=4e0,  # Weight on the Nodal Virtual Point Avoidance (not used for CTCS)
        lam_min=4e0,  # Weight on the Nodal Minimum Range Avoidance (not used for CTCS)
        lam_max=4e0,  # Weight on the Nodal Maximum Range Avoidance (not used for CTCS)
        lam_fuel=0e0,  # Weight on the Minimal Fuel Objective
        lam_t=2e1,  # Weight on the Minimal Time Objective
        lam_vc=1e2,  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
        lam_vc_ctcs=0e0,  # Weight on the Virtual Control Objective for only CTCS
        ep_tr=1e-5,  # Trust Region Tolerance
        ep_vb=1e-5,  # Virtual Control Tolerance, 2E3
        ep_vc=1e-8,  # Virtual Control Tolerance for CTCS
        ep_vc_ctcs=1e-8,  # Virtual Control Tolerance for CTCS
        w_tr_max_scaling_factor=1e2,  # Maximum Trust Region Weight
    )
    obs = ObsConfig()
    vp = VpConfig(
        n_subs=10,  # Number of subjects
        norm=2,  # Norm for the viewcone
        R_sb=np.array(
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        ),  # Rotation Matrix from Sensor to Body Frame
        alpha_x=6.0,  # Angle for the x-axis of Sensor Cone
        alpha_y=6.0,  # Angle for the y-axis of Sensor Cone
    )
    racing = RacingConfig(
        gate_centers=[
            np.array([59.436, 0.0000, 20.0000]),
            np.array([92.964, -23.750, 25.5240]),
            np.array([92.964, -29.274, 20.0000]),
            np.array([92.964, -23.750, 20.0000]),
            np.array([130.150, -23.750, 20.0000]),
            np.array([152.400, -73.152, 20.0000]),
            np.array([92.964, -75.080, 20.0000]),
            np.array([92.964, -68.556, 20.0000]),
            np.array([59.436, -81.358, 20.0000]),
            np.array([22.250, -42.672, 20.0000]),
        ],
        gate_nodes=gate_nodes_seq[i],
    )

    warm = WarmConfig()
    params = Config(sim=sim, scp=scp, obs=obs, vp=vp, racing=racing, warm=warm)

    _, LoS_vio, LoS_vio_node, runtime, iters, obj, mc_results = PTR_main_mc(params, True)
    DT_iters.append(iters)
    DT_LoS_vio_seq.append(LoS_vio)
    DT_LoS_vio_node_seq.append(LoS_vio_node)
    DT_runtime_seq.append(runtime)
    DT_obj.append(obj)
    DT_mc_results.append(mc_results)

# Save MC results
with open('results/dr_DT_mc_results.pickle', 'wb') as f:
    pickle.dump(DT_mc_results, f)

# Load the MC results
with open('results/dr_CT_mc_results.pickle', 'rb') as f:
    CT_mc_results = pickle.load(f)

with open('results/dr_DT_mc_results.pickle', 'rb') as f:
    DT_mc_results = pickle.load(f)

# Plot the results
plot_ral_mc_results(CT_mc_results, DT_mc_results, nodes, params, True, showlegend=True)
plot_ral_mc_timing_results(CT_mc_results, DT_mc_results, nodes, params, showlegend=True)