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

total_time = 40.0
sim = SimConfig(
    model="drone_6dof",
    initial_state=np.array([10, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  # Initial State
    final_state=np.array([-10, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  # Final State
    initial_control=np.array([0, 0, 10, 0, 0, 0, 1]),  # Initial Control
    max_state=np.array(
        [200, 100, 50, 100, 100, 100, 1, 1, 1, 1, 10, 10, 10, 2000, 1E-4]
    ),  # Upper Bound on the states
    min_state=np.array(
        [-100, -100, -10, -100, -100, -100, -1, -1, -1, -1, -10, -10, -10, 0, 0]
    ),  # Lower Bound on the states
    max_control=np.array(
        [0, 0, 4.179446268 * 9.81, 18.665, 18.665, 0.55562, 3.0 * total_time]
    ),  # Upper Bound on the controls
    min_control=np.array(
        [0, 0, 0, -18.665, -18.665, -0.55562, 0.3 * total_time]
    ),  # Lower Bound on the controls
    total_time=total_time,  # Total time for the simulation
    n_states = 15,  # Number of States
)
scp = ScpConfig(
    n=10,  # Number of Nodes
    fixed_final_time=True,  # Whether to fix the final time
    min_fuel=True,  # Whether to minimize fuel
    w_tr=1E1,  # Weight on the Trust Reigon
    lam_fuel=1E0,  # Weight on the Minimal Fuel Objective
    lam_t=0e0,  # Weight on the Minimal Time Objective
    lam_vc=1E1,  # Weight on the Virtual Control Objective (not including CTCS Augmentation)
    lam_vc_ctcs=1E1,  # Weight on the Virtual Control Objective for only CTCS
    ep_tr=1e-4,  # Trust Region Tolerance
    ep_vb=1e-4,  # Virtual Control Tolerance
    ep_vc=1e-8,  # Virtual Control Tolerance for CTCS
    ep_vc_ctcs=1e-8,  # Virtual Control Tolerance for CTCS
    free_final_drop=3,  # SCP iteration to relax minimal final time objective
    min_time_relax=0.8,  # Minimal Time Relaxation Factor
    w_tr_adapt=1.3,  # Trust Region Adaptation Factor
    w_tr_max_scaling_factor=1e3,  # Maximum Trust Region Weight
)
obs = ObsConfig()
vp = VpConfig(
    n_subs=1,  # Number of subjects
    R_sb=np.array(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    ),  # Rotation Matrix from Sensor to Body Frame
    alpha_x=6,  # Angle for the x-axis of Sensor Cone
    alpha_y=8,  # Angle for the y-axis of Sensor Cone
    min_range=4,  # Minimum Range to the subjects
    max_range=12,  # Maximum Range to the subjects
    tracking=True,  # Whether to use tracking
)
racing = RacingConfig()
warm = WarmConfig()
params = Config(sim=sim, scp=scp, obs=obs, vp=vp, racing=racing, warm=warm)