
import numpy as np
import scipy.integrate as itg

from quadsim.dynamics.utils import qdcm
from quadsim.dynamics.vehicles import augmented_dynamics_vec
from quadsim.discretization import s_to_t
from quadsim.config import Config

def nonlinear_constraint(x_ctcs, x_node, t, t_node, obstacles, subs, sub_jax, params):
    obs_vio = []
    
    sub_vp_vio = []
    sub_min_vio = []
    sub_max_vio = []
    sub_direc_vio = []

    state_bound_vio = []
    i = 0
    for obs in obstacles:
        obs_vio.append(np.maximum(0,obs.g_bar_ctcs(x_ctcs[:,:3])))
    sub_vp_vio = np.maximum(0, sub_jax.g_bar_vp_helper(t, x_ctcs, subs, False, params))
    sub_vp_vio_node = np.maximum(0, sub_jax.g_bar_vp_helper(t_node, x_node.T, subs, False, params))
    for sub in subs:
        if params.vp.tracking:
            sub_min_vio.append(np.maximum(0, sub.g_bar_sub_min_ctcs(t, x_ctcs, params, "Full")))
            sub_max_vio.append(np.maximum(0, sub.g_bar_sub_max_ctcs(t, x_ctcs, params, "Full")))
    for i_state in range(params.sim.n_states-1):
        state_bound_vio.append(np.maximum(0, params.sim.min_state[i_state]-x_ctcs[:, i_state]) + np.maximum(0, x_ctcs[:, i_state]-params.sim.max_state[i_state]))
    state_bound_vio = np.array(state_bound_vio)
    obs_vio = np.array(obs_vio)
    sub_max_vio = np.array(sub_max_vio)
    sub_min_vio = np.array(sub_min_vio)
    return obs_vio, sub_vp_vio, sub_vp_vio_node, sub_min_vio, sub_max_vio, sub_direc_vio, state_bound_vio


def simulate_nonlinear(x_0, u, subs, sub_jax, params, node_only = False):
    states = [x_0]
    controls_next = None

    t_vals = s_to_t(u, params)
    tau = np.linspace(0, 1, params.scp.n)

    if node_only:
        for k in range(params.scp.n-1):
            controls_current = u[:,k]
            if params.scp.dis_type == 'FOH':
                controls_next = u[:,k+1]
            x = itg.solve_ivp(augmented_dynamics_vec, (tau[k], tau[k+1]), x_0, args=(controls_current[None,:], controls_next[None,:], np.array([[tau[k]]]), True, np.array([[t_vals[k]]]), [], subs, sub_jax, params), method='DOP853').y
            states.append(x[:,-1])
    else:
        for k in range(params.scp.n-1):
            controls_current = u[:,k]
            t_eval = np.linspace(tau[k], tau[k+1], params.sim.inter_sample)
            if params.scp.dis_type == 'FOH':
                controls_next = u[:,k+1]
            x = itg.solve_ivp(augmented_dynamics_vec, (tau[k], tau[k+1]), x_0, args=(controls_current[None,:], controls_next[None,:], np.array([[tau[k]]]), True, np.array([[t_vals[k]]]), [], subs, sub_jax, params), method='DOP853', t_eval=t_eval).y
            for k in range(1, x.shape[1]):
                states.append(x[:,k])
            x_0 = x[:,-1]
    return np.array(states)

def interpolate_control(u, params):
    tau = np.linspace(0, 1, params.scp.n)
    u_full = []
    for i in range(params.scp.n-1):
        u_cur = u[:,i]
        u_next = u[:,i+1]
        tau_interp = np.linspace(tau[i], tau[i+1], params.sim.inter_sample)
        for inter_tau in tau_interp:
            if params.scp.dis_type == 'ZOH':
                beta = 0.
            elif params.scp.dis_type == 'FOH':
                beta = (inter_tau-tau[i]) * params.scp.n
            u_full.append(u_cur + beta*(u_next - u_cur))
    return np.array(u_full)

def full_subject_traj(u, x_full, u_full, subs, params, init):
    subs_traj = []
    subs_traj_sen = []
    t = s_to_t(u, params)
    t_full = []
    for i in range(params.scp.n-1):
        t_interp = np.linspace(t[i], t[i+1], params.sim.inter_sample)
        t_interp = t_interp[:-1]
        t_full.append(t_interp)
    t_full = np.array(t_full).flatten()
    # Add the last element of t
    t_full = np.append(t_full, t[-1])
    for sub in subs:
        subs_traj.append(sub.get_position(t_full, False))
    
    if not init:
        R_sb = params.vp.R_sb
        for sub_traj in subs_traj:
            sub_traj_sen = []
            for i in range(x_full.shape[0]):
                sub_pose = sub_traj[i]
                sub_traj_sen.append(R_sb @ qdcm(x_full[i, 6:10]).T @ (sub_pose - x_full[i, 0:3]))
            subs_traj_sen.append(sub_traj_sen)
    else:
        subs_traj_sen = None
    
    return subs_traj, np.array(t_full).flatten(), subs_traj_sen

def full_time(u, params):
    t = s_to_t(u, params)
    t_full = []
    for i in range(params.scp.n-1):
        t_interp = np.linspace(t[i], t[i+1], params.sim.inter_sample)
        t_interp = t_interp[:-1]
        t_full.append(t_interp)
    t_full = np.array(t_full).flatten()
    # Add the last element of t
    t_full = np.append(t_full, t[-1])
    return t_full

def subject_traj(x, u, subs, params):
    subs_traj = []
    subs_traj_sen = []
    t = np.array(s_to_t(u, params))
    for sub in subs:
        sub_traj = []
        sub_traj.append(sub.get_position(t, False))
        subs_traj.append(np.squeeze(np.array(sub_traj)))

    R_sb = params.vp.R_sb
    for sub_traj in subs_traj:
        sub_traj_sen = []
        for i in range(x.shape[1]):
            sub_pose = sub_traj[i]
            sub_traj_sen.append(R_sb @ qdcm(x[6:10,i]).T @ (sub_pose - x[0:3,i]))
        subs_traj_sen.append(sub_traj_sen)
    
    return subs_traj_sen