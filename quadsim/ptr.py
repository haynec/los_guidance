import numpy as np
import scipy.linalg as sla
import numpy.linalg as la
import cvxpy as cp
import pickle
import time
import jax.numpy as jnp

import sys
from termcolor import colored

from quadsim.discretization import calculate_discretization, dVdt, s_to_t
from quadsim.propagation import simulate_nonlinear, interpolate_control, nonlinear_constraint, full_subject_traj, subject_traj, full_time
from quadsim.dynamics.vehicles import auto_jacobians_vec, Quad_JAX
from quadsim.dynamics.constraints import Ellipsoidal_Obstacle, Planar_Rectangular_Gate, Subject, Subject_JAX
from quadsim.ocp import OCP
from quadsim.plotting import plot_initial_guess, save_gate_parameters
from quadsim.config import Config

import warnings
# warnings.filterwarnings("ignore")

def PTR_main(params):
    J_vb = 1E2
    J_vc = 1E2
    J_tr = 1E2
    
    sub_jax = Subject_JAX(params)
    quad_jax = Quad_JAX(params)

    obstacles = []
    axes = None if not params.obs.obstacle_axes else params.obs.obstacle_axes
    radius = None if not params.obs.obstacle_radius else params.obs.obstacle_radius
    
    if axes is not None and len(axes) != params.obs.n_obs:
        raise ValueError(f"Expected {params.obs.n_obs} axes, but got {len(axes)}.")

    if radius is not None and len(radius) != params.obs.n_obs:
        raise ValueError(f"Expected {params.obs.n_obs} radii, but got {len(radius)}.")

    for i in range(params.obs.n_obs):
        ax = axes[i] if axes is not None else None
        rad = radius[i] if radius is not None else None
        obstacles.append(Ellipsoidal_Obstacle(params.obs.obstacle_centers[i], ax, rad))

    gates = []
    for i in range(params.racing.n_gates):
        gates.append(Planar_Rectangular_Gate(params.racing.gate_centers[i], params.racing.gate_nodes))
    
    save_gate_parameters(gates, params)

    subs = []
    np.random.seed(5)
    for i in range(params.vp.n_subs):
        init_pose = np.array([100.0, -70.0, 20.0])
        init_pose[:2] = init_pose[:2] + np.random.random(2)*20.0
        subs.append(Subject(init_pose, params))
        
    x_bar, u_bar = scp_init(params.sim.initial_state, params.sim.initial_control, params.sim.final_state, params.scp.n, gates, subs, params)

    if params.vp.tracking:
        params.sim.initial_state[:3] = x_bar[:3,0]

    prob = OCP(params) # Initialize the problem
    A, B = auto_jacobians_vec(params) # Automatic Jacobians

    gate_centers = np.zeros((3, params.racing.n_gates))
    A_gates = np.zeros((3*params.racing.n_gates, 3))
    mult_A_cen = np.zeros((3, params.racing.n_gates))
    norm_gates = np.zeros((3, params.racing.n_gates))

    for k in range(params.racing.n_gates):
        gate_centers[:,k] = gates[k].center
        A_gates[3*k:3*(k+1),:] = gates[k].A
        mult_A_cen[:,k] = gates[k].A @ gates[k].center
        norm_gates[:,k] = gates[k].normal

    if params.racing.n_gates > 0:
        prob.param_dict['A_gates'].value = A_gates
        prob.param_dict['mult_A_cen'].value = mult_A_cen

    scp_trajs = [x_bar]
    scp_controls = [u_bar]

    # Define colors for printing
    col_main = "blue"
    col_pos = "green"
    col_neg = "red"

    print("{:^4} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^14}".format(
            "Iter", "J_total", "J_tr", "J_vb", "J_vc", "J_vc_ctcs", "TOF", "Solver Status"))
    print(colored("-----------------------------------------------------------------------------------"))

    k = 1
    cpg_solve = None

    # Solve a dumb problem to intilize DPP and JAX jacobians
    _ = PTR_subproblem(cpg_solve, x_bar, u_bar, A, B, obstacles, subs, prob, sub_jax, quad_jax, params)

    t_0_while = time.time()
    while k <= params.scp.k_max and ((J_tr >= params.scp.ep_tr) or (J_vb >= params.scp.ep_vb) or (J_vc >= params.scp.ep_vc) or (J_vc_ctcs >= params.scp.ep_vc_ctcs)):
        x, u, t, J_total, J_vb_vec, J_vc_vec, J_tr_vec, J_vc_ctcs_vec, prob_stat = PTR_subproblem(cpg_solve, x_bar, u_bar, A, B, obstacles, subs, prob, sub_jax, quad_jax, params)

        x_bar = x
        u_bar = u

        J_tr = np.sum(np.array(J_tr_vec))
        J_vb = np.sum(np.array(J_vb_vec))
        J_vc = np.sum(np.array(J_vc_vec))
        J_vc_ctcs = np.sum(np.array(J_vc_ctcs_vec))
        scp_trajs.append(x)
        scp_controls.append(u)

        params.scp.w_tr = min(params.scp.w_tr * params.scp.w_tr_adapt, params.scp.w_tr_max)
        if k > params.scp.free_final_drop:
            params.scp.lam_t = params.scp.lam_t * params.scp.min_time_relax
        
        # remove bottom labels and line
        if not k == 0:
            sys.stdout.write('\x1b[1A\x1b[2K\x1b[1A\x1b[2K')

        # Determine color for each value
        iter_colored = colored("{:4d}".format(k))
        J_tot_colored = colored("{:.1e}".format(J_total))
        J_tr_colored = colored("{:.1e}".format(J_tr), col_pos if J_tr <= params.scp.ep_tr else col_neg)
        J_vb_colored = colored("{:.1e}".format(J_vb), col_pos if J_vb <= params.scp.ep_vb else col_neg)
        J_vc_colored = colored("{:.1e}".format(J_vc), col_pos if J_vc <= params.scp.ep_vc else col_neg)
        J_vc_ctcs_colored = colored("{:.1e}".format(J_vc_ctcs), col_pos if J_vc_ctcs <= params.scp.ep_vc_ctcs else col_neg)
        tof_colored = colored("{:.1e}".format(t[-1]))
        prob_stat_colored = colored(prob_stat, col_pos if prob_stat == 'optimal' else col_neg)

        # Print with colors
        print("{:^4} | {:^7} | {:^7} | {:^7} | {:^7} |  {:^7}  | {:^7} | {:^14}".format(
            iter_colored, J_tot_colored, J_tr_colored, J_vb_colored, J_vc_colored, J_vc_ctcs_colored, tof_colored, prob_stat_colored))

        print(colored("-----------------------------------------------------------------------------------"))
        print("{:^4} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^7} | {:^14}".format(
            "Iter", "J_total", "J_tr", "J_vb", "J_vc", "J_vc_ctcs", "TOF", "Solver Status"))
        k += 1

    t_f_while = time.time()
    
    print(colored("-----------------------------------------------------------------------------------"))
    # Define ANSI color codes
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # Print with bold text
    print("------------------------------------- " + BOLD + "RESULTS" + RESET + " -------------------------------------")
    print("Total Computation Time: ", t_f_while - t_0_while)
    x_full = simulate_nonlinear(x[:,0], u, subs, sub_jax, params)
    # print("Time Grid: ", t)
    # print('Initial Time Dilation Constants: ', u_org[-1])
    # print('Final Time Dilation Constants: ', u[-1])
    # x_full_node = simulate_nonlinear(x[:,0], u, subs, sub_jax, params, True).T
    u_full = interpolate_control(u, params)

    x_sub_full = []
    if len(subs) != 0:
        x_sub_full, _, x_sub_sen = full_subject_traj(u, x_full, u_full, subs, params, False)

        x_sub_sen_node = subject_traj(x, u, subs, params)
    else:
        x_sub_sen = []
        x_sub_sen_node = []
    t_full = full_time(u, params)

    obs_vio, sub_vp_vio, sub_vp_vio_node, sub_min_vio, sub_max_vio, sub_direc_vio, state_bound_vio = nonlinear_constraint(x_full, x, t_full, t, obstacles, subs, sub_jax, params)

    if params.vp.n_subs > 0:
        LoS_vio = sub_vp_vio.sum()/sub_vp_vio.shape[1]
        LoS_vio_node = sub_vp_vio_node.sum()/sub_vp_vio_node.shape[1]
    else:
        LoS_vio = 0
        LoS_vio_node = 0

    print("Total LoS CTCS Constraint Violation:", LoS_vio)
    print("Total LoS Node Constraint Violation:", LoS_vio_node)
    if params.vp.tracking:
        # Minimal Fuel Consumption
        obj = x[-2, -1]
        print("Original Objective Value: ", obj)
    else:
        obj = t[-1]
        print("Time-of-Flight: ", t[-1])

    params.sim.total_time = t[-1]

    scp_trajs_interp = scp_traj_interp(scp_trajs, params)

    result = dict(
        tof = t[-1],
        drone_state = x_full,
        drone_positions = x_full[:,:3],
        drone_velocities = x_full[:,3:6],
        drone_attitudes = x_full[:,6:10],
        drone_forces = u_full[:,:3],
        drone_controls = u,
        drone_state_scp = x,
        sub_positions = x_sub_full,
        sub_positions_sen = x_sub_sen,
        sub_positions_sen_node = x_sub_sen_node,
        scp_trajs = scp_trajs,
        scp_controls = scp_controls,
        obstacles = obstacles,
        gates = gates,
        scp_interp = scp_trajs_interp,
        J_tr_vec = J_tr_vec,
        J_vb_vec = J_vb_vec,
        J_vc_vec = J_vc_vec,
        J_vc_ctcs_vec = J_vc_ctcs_vec,
        obs_vio = obs_vio,
        sub_vp_vio = sub_vp_vio,
        sub_min_vio = sub_min_vio,
        sub_max_vio = sub_max_vio,
        sub_direc_vio = sub_direc_vio,
        state_bound_vio = state_bound_vio
    )

    # Save Drone State and Control Trajectories
    airsim_results = dict(
        drone_state = x_full,
        drone_controls = u_full,
        x_sub_full = x_sub_full
    )
    with open('results/drone_state.pkl', 'wb') as f:
        pickle.dump(airsim_results, f)

    return result, LoS_vio, LoS_vio_node, t_f_while - t_0_while, k, obj

def unit_test_jacobian(A, B, A_jax, B_jax, x, u, params):
    # Verify that the analuytical and jax jacobians are the same
    # Extract the state and control input
    
    # Extract a single state and control input
    x = x[:13,4]
    u = u[:6,4]

    # Generate random x and u
    x = np.random.rand(13)
    # Normalize the quaternion component
    x[6:10] = x[6:10] / la.norm(x[6:10])
    u = np.random.rand(6)

    p, v, q, w, f, torque = np.expand_dims(x[:3], 1), np.expand_dims(x[3:6],1), np.expand_dims(x[6:10],1), np.expand_dims(x[10:13],1), np.expand_dims(u[:3],1), np.expand_dims(u[3:6],1)

    # Evaluate the State Jacobian
    dx = A(p, v, q, w, f, torque)
    du = B(p, v, q, w, f, torque)

    x = jnp.array(x)
    u = jnp.array(u.astype(float))
    dx_jax = A_jax(x, u)
    du_jax = B_jax(x, u)
    np.array_equal(dx, dx_jax)
    np.array_equal(du, du_jax)
    return

def PTR_subproblem(cpg_solve, x_bar, u_bar, A, B, obstacles, subs, prob, sub_jax, quad_jax, params):
    J_vb_vec = []
    J_vc_vec = []
    J_vc_ctcs_vec = []
    J_tr_vec = []
    
    prob.param_dict['x_bar'].value = x_bar
    prob.param_dict['u_bar'].value = u_bar
    
    A_bar, B_bar, C_bar, z_bar = calculate_discretization(x_bar, u_bar, A, B, obstacles, subs, sub_jax, quad_jax, params)
    prob.param_dict['A_d'].value = A_bar
    prob.param_dict['B_d'].value = B_bar
    prob.param_dict['C_d'].value = C_bar
    prob.param_dict['z_d'].value = z_bar

    t = s_to_t(u_bar, params)

    if not params.scp.ctcs:
        i_obs = 0
        for obs in obstacles:
            prob.param_dict['g_bar_obs_' + str(i_obs)].value = obs.g_bar_obs(t, x_bar[:3,:])
            prob.param_dict['grad_g_bar_obs_' + str(i_obs)].value = obs.grad_g_bar_obs(t, x_bar[:3, :])
            i_obs += 1
        
        i_sub = 0
        for sub in subs:
            prob.param_dict['g_bar_vp_' + str(i_sub)].value = sub.g_bar_vp_ctcs_org_vec(np.array(t), x_bar.T, params)
            prob.param_dict['grad_g_bar_vp_' + str(i_sub)].value = sub.grad_g_bar_vp_ctcs_org_vec(np.array(t), x_bar.T, params)
            if params.vp.tracking:
                prob.param_dict['g_bar_min_' + str(i_sub)].value = sub.g_bar_sub_min_ctcs(np.array(t), x_bar.T, params, 'Full')
                prob.param_dict['grad_g_bar_min_' + str(i_sub)].value = sub.grad_g_bar_sub_min_nodal(np.array(t), x_bar.T, params, 'Full').T
                prob.param_dict['g_bar_max_' + str(i_sub)].value = sub.g_bar_sub_max_ctcs(np.array(t), x_bar.T, params, 'Full')
                prob.param_dict['grad_g_bar_max_' + str(i_sub)].value = sub.grad_g_bar_sub_max_nodal(np.array(t), x_bar.T, params, 'Full').T
            i_sub += 1

    prob.param_dict['w_tr'].value = params.scp.w_tr

    if not params.scp.fixed_final_time:
        prob.param_dict['lam_t'].value = params.scp.lam_t
    
    if params.scp.min_fuel:
        prob.param_dict['lam_fuel'].value = params.scp.lam_fuel

    prob.solve(solver = cp.CLARABEL, enforce_dpp = True)

    x = params.sim.S_x @ prob.var_dict['x'].value + np.expand_dims(params.sim.c_x, axis = 1)
    u = params.sim.S_u @ prob.var_dict['u'].value + np.expand_dims(params.sim.c_u, axis = 1)

    t = [0]
    tau = np.linspace(0, 1, params.scp.n)
    for k in range(1, params.scp.n):
        if params.scp.dis_type == 'ZOH':
            t.append(t[k-1] + (tau[k] - tau[k-1])*(u[-1,k-1]))
        else:
            t.append(t[k-1] + 0.5 * (u[-1,k] + u[-1,k-1]) * (tau[k] - tau[k-1]))

    J_tr_vec.append(la.norm(la.inv(params.sim.S_x) @ (x[:,0] - x_bar[:,0])))
    for k in range(params.scp.n):
        J_tr_vec.append(la.norm(la.inv(sla.block_diag(params.sim.S_x, params.sim.S_u)) @ (np.hstack((x[:,k], u[:,k-1])) - np.hstack((x_bar[:,k], u_bar[:,k-1]))))**2)

        J_vb = 0
        if not params.scp.ctcs:
            for sub in subs:
                J_vb += np.maximum(0, (sub.g_bar_vp_ctcs_org_vec(np.array(t), x.T, params) + np.squeeze((sub.grad_g_bar_vp_ctcs_org_vec(np.array(t), x.T, params) @ (x[:,k] - x_bar[:,k])[:,None])))).sum()
                if params.vp.tracking:
                    J_vb += np.maximum(0, (sub.g_bar_sub_min_ctcs(np.array(t), x.T, params, 'Full')[:,None] + sub.grad_g_bar_sub_min_ctcs(np.array(t), x.T, params, 'Full') @ (x[:3,k]  - x_bar[:3,k])[:,None])).sum()
                    J_vb += np.maximum(0, (sub.g_bar_sub_max_ctcs(np.array(t), x.T, params, 'Full')[:,None] + sub.grad_g_bar_sub_max_ctcs(np.array(t), x.T, params, 'Full') @ (x[:3,k]  - x_bar[:3,k])[:,None])).sum()
        J_vb_vec.append(J_vb)
        J_vc_vec.append(np.sum(np.abs(prob.var_dict['nu'].value[:-1,k-1])))
        J_vc_ctcs_vec.append(np.sum(np.abs(prob.var_dict['nu'].value[-1,k-1])))
    return x, u, np.array(t), prob.value, J_vb_vec, J_vc_vec, J_tr_vec, J_vc_ctcs_vec, prob.status

def scp_init(x : np.ndarray,
                       u : np.ndarray,
                       x_f : np.ndarray,
                       n : int,
                       gates : list,
                       subs : list,
                       params : dict
                       ) -> np.ndarray:
    u_traj = np.repeat(np.expand_dims(u, axis = 1), n, axis = 1)
    
    # Linear interpolation between x, gate centers, and x_f with n/number of gate nodes
    x_traj = np.zeros((params.sim.n_states,n))
    x_traj[-1] = 0.0
    x_traj[6] = 1

    # Iterate through gates
    if not params.vp.tracking:
        if params.racing.n_gates != 0:
            i = 0
            origins = [x[:3]]
            ends = []
            for gate in gates:
                origins.append(gate.center)
                ends.append(gate.center)
            ends.append(x_f[:3])
            gate_idx = 0
            for j in range(params.racing.n_gates + 1):
                for k in range(n//(params.racing.n_gates + 1)):
                    x_traj[:3,i] = origins[gate_idx] + (k/(n//(params.racing.n_gates + 1))) * (ends[gate_idx] - origins[gate_idx])
                    i += 1
                gate_idx += 1
        else:
            for k in range(n):
                x_traj[:6,k] = x[:6] + (k/(n-1)) * (x_f[:6] - x[:6])
    else:
        s = params.sim.total_time
        u_traj[-1] = np.repeat(s, n)
        t = s_to_t(u_traj, params)
        for k in range(n):
            if params.vp.tracking:
                key_point = subs[0].get_position(t[k], True)
            else:
                key_point = subs[0].get_position(t[k], False)
            x_traj[:3,k] = key_point + np.array([-5, 0.2, 0.2])

    s = params.sim.total_time
    u_traj[-1] = np.repeat(s, n)
    
    # Point the attitude towards the keypoint
    if params.vp.n_subs != 0:   
        R_sb = params.vp.R_sb # Sensor to body frame
        b = R_sb @ np.array([0, 1, 0])
        t = s_to_t(u_traj, params)
        for k in range(n):
            kp = []
            for sub in subs:
                kp.append(sub.get_position(t[k], True).squeeze())
            mean_kp = np.mean(kp, axis = 0)
            a = mean_kp - x_traj[:3,k]
            # Determine the direction cosine matrix that aligns the z-axis of the sensor frame with the relative position vector
            q_xyz = np.cross(b, a)
            q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a,b)
            q_no_norm = np.hstack((q_w, q_xyz))
            q = q_no_norm / la.norm(q_no_norm)
            x_traj[6:10,k] = q

    if params.vp.n_subs != 0:
        x_sub_full, _, _ = full_subject_traj(u_traj, x_traj, u_traj, subs, params, True)
    else:
        x_sub_full = []

    result = dict(
        x = x_traj,
        u = u_traj,
        sub_positions = x_sub_full
    )
    # plot_initial_guess(result, params)
    return x_traj, u_traj

def scp_traj_interp(scp_trajs, params):
    scp_prop_trajs = []
    for traj in scp_trajs:
        states = []
        for k in range(params.scp.n):
            traj_temp = np.repeat(np.expand_dims(traj[:,k], axis = 1), params.sim.inter_sample - 1, axis = 1)
            for i in range(1, params.sim.inter_sample - 1):
                states.append(traj_temp[:,i])
        scp_prop_trajs.append(np.array(states))
    return scp_prop_trajs


def scp_init_warm(
    x_3dof: np.ndarray,
    u_3dof: np.ndarray,
    n: int,
    gates: list,
    subs: list,
    params: dict,
):

    x_traj = x_3dof
    x_fill = np.tile(np.array([1, 0, 0, 0, 0, 0, 0]), (x_3dof.shape[1], 1)).T
    x_traj = np.insert(x_traj, 6, x_fill, axis=0)

    u_traj = u_3dof
    u_fill = np.tile(np.array([0, 0, 0]), (u_3dof.shape[1], 1)).T
    u_traj = np.insert(u_traj, 3, u_fill, axis=0)

    # Point the attitude towards the keypoint
    if params.vp.n_subs != 0:
        R_sb = params.vp.R_sb  # Sensor to body frame
        b = R_sb @ np.array([0, 1, 0])
        t = s_to_t(u_traj, params)
        for k in range(n):
            kp = []
            for sub in subs:
                kp.append(sub.get_position(t[k], True))
            mean_kp = np.mean(kp, axis=0)
            a = mean_kp - x_traj[:3, k]
            # Determine the direction cosine matrix that aligns the z-axis of the sensor frame with the relative position vector
            q_xyz = np.cross(b, a)
            q_w = np.sqrt(la.norm(a) ** 2 + la.norm(b) ** 2) + np.dot(a, b)
            q_no_norm = np.hstack((q_w, q_xyz))
            q = q_no_norm / la.norm(q_no_norm)
            x_traj[6:10, k] = q

    return x_traj, u_traj