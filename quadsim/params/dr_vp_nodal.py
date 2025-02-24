import numpy as np
from quadsim.config import (
    SimConfig,
    ScpConfig,
    ObsConfig,
    VpConfig,
    RacingConfig,
    WarmConfig,
    Config,
)

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
    n=22,  # Number of Nodes
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
    ep_vb=1e-5,  # Virtual Control Tolerance
    ep_vc=1e-8,  # Virtual Control Tolerance for CTCS
    ep_vc_ctcs=1e-8,  # Virtual Control Tolerance for CTCS'
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
    gate_nodes = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]),
    # gate_nodes = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30]),
    # gate_nodes = np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 40]),
    # gate_nodes = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]),
    # gate_nodes = np.array([8, 16, 24, 32, 40, 48, 56, 64, 72, 80])
    # gate_nodes = np.array([12, 24, 36, 48, 60, 72, 84, 96, 108, 120])
)

warm = WarmConfig()
params = Config(sim=sim, scp=scp, obs=obs, vp=vp, racing=racing, warm=warm)
