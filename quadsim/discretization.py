import jax.numpy as jnp
from jax import vmap, jit
from jax.experimental.ode import odeint
import numpy as np
from quadsim.dynamics.vehicles import augmented_dynamics_vec, Quad_JAX
from quadsim.dynamics.constraints import Subject_JAX
import scipy.integrate as itg
from quadsim.config import Config

def s_to_t(u, params):
    t = [0]
    tau = np.linspace(0, 1, params.scp.n)
    for k in range(1, params.scp.n):
        s_kp = u[-1,k-1]
        s_k = u[-1,k]
        if params.scp.dis_type == 'ZOH':
            t.append(t[k-1] + (tau[k] - tau[k-1])*(s_kp))
        else:
            t.append(t[k-1] + 0.5 * (s_k + s_kp) * (tau[k] - tau[k-1]))
    return t

def t_to_s(u, t, params):
    s = np.zeros(params.scp.n)
    tau = np.linspace(0, 1, params.scp.n)
    for k in range(1, params.scp.n):
        s_kp = u[-1,k-1]
        s_k = u[-1,k]
        if params.scp.dis_type == 'ZOH':
            s[k] = (t[k] - t[k-1]) / (tau[k] - tau[k-1])
        else:
            s[k] = 2 * (t[k] - t[k-1]) / (s_k + s_kp)
    return s

def calculate_discretization(x: np.ndarray,
                             u: np.ndarray,
                             A,
                             B,
                             obstacles,
                             subs,
                             sub_jax,
                             quad_jax,
                             params: dict):
    """
    Calculate discretization for given states, inputs and total time.
    X: Matrix of states for all time points
    U: Matrix of inputs for all time points
    return: A_k, B_k, C_k, z_k
    """

    # Extract the number of states and controls from the parameters
    n_x = params.sim.n_states
    n_u = params.sim.n_controls

    # Define indices for slicing the augmented state vector
    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u
    i5 = i4 + n_x

    # Initialize the augmented state vector
    V0 = np.zeros((x.shape[1]-1, i5))
    V0[:, i1:i2] = np.tile(np.eye(n_x).reshape(-1), (x.shape[1]-1, 1))

    # Initialize the augmented Jacobians
    A_bar = np.zeros((n_x * n_x, params.scp.n - 1))
    B_bar = np.zeros((n_x * n_u, params.scp.n - 1))
    C_bar = np.zeros((n_x * n_u, params.scp.n - 1))
    z_bar = np.zeros((n_x, params.scp.n - 1))

    # Propagate the system forward in time from 0 to T
    tau = np.linspace(0, 1, params.scp.n)
    t_vals = s_to_t(u, params)

    # Vectorized integration
    V0[:, i0:i1] = x[:, :-1].T
    int_result = itg.solve_ivp(dVdt, (tau[0], tau[1]), V0.flatten(), args=(u[:, :-1].T, u[:, 1:].T, A, B, obstacles, subs, tau[0], np.array(t_vals[:-1]), sub_jax, quad_jax, params), method='RK45', t_eval=np.linspace(tau[0], tau[1], 50))
    V = int_result.y[:,-1].reshape(-1, i5)

    V_multi_shoot = int_result.y

    # Flatten matrices in column-major (Fortran) order for cvxpy
    A_bar = V[:, i1:i2].reshape((params.scp.n - 1, n_x, n_x)).transpose(1, 2, 0).reshape(n_x * n_x, -1, order='F')
    B_bar = V[:, i2:i3].reshape((params.scp.n - 1, n_x, n_u)).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F')
    C_bar = V[:, i3:i4].reshape((params.scp.n - 1, n_x, n_u)).transpose(1, 2, 0).reshape(n_x * n_u, -1, order='F')
    z_bar = V[:, i4:i5].T

    return A_bar, B_bar, C_bar, z_bar, V_multi_shoot

def dVdt(tau: float,
             V_seq: np.ndarray,
             control_current_seq: np.ndarray,
             control_next_seq: np.ndarray,
             A,
             B,
             obstacles,
             subs,
             tau_init: float,
             t_init: float,
             sub_jax, 
             quad_jax,
             params: dict,
             debug = False) -> np.ndarray:
    """
    Computes the time derivative of the augmented state vector for the system for a sequence of states.

    Parameters:
    tau (float): Current time.
    V_seq (np.ndarray): Sequence of augmented state vectors.
    control_current_seq (np.ndarray): Sequence of current control inputs.
    control_next_seq (np.ndarray): Sequence of next control inputs.
    A: Function that computes the Jacobian of the system dynamics with respect to the state.
    B: Function that computes the Jacobian of the system dynamics with respect to the control input.
    obstacles: List of obstacles in the environment.
    params (dict): Parameters of the system.

    Returns:
    np.ndarray: Time derivatives of the augmented state vectors.
    """
    
    # Extract the number of states and controls from the parameters
    n_x = params.sim.n_states
    n_u = params.sim.n_controls

    # Define indices for slicing the augmented state vector
    i0 = 0
    i1 = n_x
    i2 = i1 + n_x * n_x
    i3 = i2 + n_x * n_u
    i4 = i3 + n_x * n_u
    i5 = i4 + n_x

    # Unflatten V
    V_seq = V_seq.reshape(-1, i5)

    if not debug:
        tau_init = np.linspace(0, 1, params.scp.n)[:-1]

    # Compute the interpolation factor based on the discretization type
    if params.scp.dis_type == 'ZOH':
        beta = 0.
    elif params.scp.dis_type == 'FOH':
        beta = (tau) * params.scp.n
    alpha = 1 - beta

    # Interpolate the control input
    u_seq = control_current_seq + beta * (control_next_seq - control_current_seq)
    s_seq = u_seq[:,-1]

    # Generate a sequence of tau values
    if debug:
        tau_seq = tau
    else:
        tau_seq = np.linspace(0, 1, params.scp.n)[:-1] + tau

    # Compute t_seq based on the discretization type
    if params.scp.dis_type == 'ZOH':
        t_seq = t_init[:, None] + (tau_seq - tau_init) * s_seq[:, None]
    elif params.scp.dis_type == 'FOH':
        t_seq = t_init + 0.5 * np.array(s_seq + control_current_seq[:,-1]) * (tau_seq - tau_init)

    # Initialize the augmented Jacobians
    dfdx_seq = np.zeros((V_seq.shape[0], n_x, n_x))
    B_aug_seq = np.zeros((V_seq.shape[0], n_x, n_u))

    # Ensure x_seq and u_seq have the same batch size
    x_seq = V_seq[:,:params.sim.n_states]
    u_seq = u_seq[:x_seq.shape[0]]

    # Evaluate the State Jacobian
    if params.scp.min_fuel:
        dfdx_veh_seq = A(x_seq[:,:-2], u_seq[:,:-1])
        dfdx_seq[:, :-2, :-2] = dfdx_veh_seq
    else:
        dfdx_veh_seq = A(x_seq[:,:-1], u_seq[:,:-1])
        dfdx_seq[:, :-1, :-1] = dfdx_veh_seq

    # Copmute the CTCS Gradient
    if params.scp.ctcs:
        if params.obs.n_obs != 0:
            dfdx_seq[:, -1, :3] += np.sum([obs.grad_g_bar_ctcs(x_seq[:,:3]) for obs in obstacles], axis=0)

            # check = np.sum([obs.grad_g_bar_obs(x_seq[:,:3]) for obs in obstacles], axis=0).T
        dfdx_seq[:, -1, :-1] += -2 * np.maximum(0, params.sim.min_state[:-1] - x_seq[:,:-1])
        dfdx_seq[:, -1, :-1] +=  2 * np.maximum(0, x_seq[:,:-1] - params.sim.max_state[:-1])
        if params.vp.n_subs != 0:
            for sub in subs:
                if params.vp.tracking:
                    dfdx_seq[:, -1, :3] += sub.grad_g_bar_sub_min_ctcs(t_seq, x_seq, params, 'Full')
                    dfdx_seq[:, -1, :3] += sub.grad_g_bar_sub_max_ctcs(t_seq, x_seq, params, 'Full')
            dfdx_seq[:, -1, :] += sub_jax.grad_g_bar_vp_helper(t_seq, x_seq, subs, params, None, None)
    sdfdx_seq = s_seq[:, None, None] * dfdx_seq

    # Compute the nonlinear propagation term
    if params.sim.model == "drone_3dof":
        F_seq = augmented_dynamics_vec_di(tau, x_seq, u_seq, None, None, False, t_seq, obstacles, subs, sub_jax, params)
    elif params.sim.model == "drone_6dof":
        F_seq = augmented_dynamics_vec(tau, x_seq, u_seq, None, None, False, t_seq, obstacles, subs, sub_jax, params)
    f_subs_seq = s_seq[:, None] * F_seq

    # Evaluate the Control Jacobian
    if params.scp.min_fuel:
        dfdu_veh_seq = B(x_seq[:,:-2], u_seq[:,:-1])
        B_aug_seq[:, :-2, :-1] = s_seq[:, None, None] * dfdu_veh_seq
        B_aug_seq[:, 13, :-1] = quad_jax.fuel_jacobian(u_seq[:,:-1], params)
    else:
        dfdu_veh_seq = B(x_seq[:,:-1], u_seq[:,:-1])
        B_aug_seq[:, :-1, :-1] = s_seq[:, None, None] * dfdu_veh_seq
    B_aug_seq[:, :, -1] = F_seq

    # Compute the error from nonlinear propagation and linearization
    z_t_seq = f_subs_seq - np.einsum('ijk,ik->ij', sdfdx_seq, x_seq) - np.einsum('ijk,ik->ij', B_aug_seq, u_seq)

    # Stack up the results into the augmented state vector
    dVdt_seq = np.zeros_like(V_seq)
    dVdt_seq[:, i0:i1] = f_subs_seq
    dVdt_seq[:, i1:i2] = np.matmul(sdfdx_seq, V_seq[:, i1:i2].reshape(-1, n_x, n_x)).reshape(-1, n_x * n_x)
    dVdt_seq[:, i2:i3] = (np.matmul(sdfdx_seq, V_seq[:, i2:i3].reshape(-1, n_x, n_u)) + B_aug_seq * alpha).reshape(-1, n_x * n_u)
    dVdt_seq[:, i3:i4] = (np.matmul(sdfdx_seq, V_seq[:, i3:i4].reshape(-1, n_x, n_u)) + B_aug_seq * beta).reshape(-1, n_x * n_u)
    dVdt_seq[:, i4:i5] = (np.matmul(sdfdx_seq, V_seq[:, i4:i5].reshape(-1, n_x)[..., None]).squeeze(-1) + z_t_seq).reshape(-1, n_x)
    return dVdt_seq.flatten()