import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import cvxpy as cp

def OCP(params):
    ########################
    # VARIABLES & PARAMETERS
    ########################

    # Parameters
    w_tr = cp.Parameter(nonneg = True, name='w_tr')
    lam_t = cp.Parameter(nonneg=True, name='lam_t')
    if params.scp.min_fuel:
        lam_fuel = cp.Parameter(nonneg=True, name='lam_fuel')

    # State
    x = cp.Variable((params.sim.n_states, params.scp.n), name='x') # Current State
    dx = cp.Variable((params.sim.n_states, params.scp.n), name='dx') # State Error
    x_bar = cp.Parameter((params.sim.n_states, params.scp.n), name='x_bar') # Previous SCP State

    # Affine Scaling for State
    S_x = params.sim.S_x
    c_x = params.sim.c_x

    # Control
    u = cp.Variable((params.sim.n_controls, params.scp.n), name='u') # Current Control
    du = cp.Variable((params.sim.n_controls, params.scp.n), name='du') # Control Error
    u_bar = cp.Parameter((params.sim.n_controls, params.scp.n), name='u_bar') # Previous SCP Control

    # Normalized Time
    tau = np.linspace(0, 1, params.scp.n)

    # Affine Scaling for Control
    S_u = params.sim.S_u
    c_u = params.sim.c_u

    # Linearized Jacobians for Nodal Constraints Avoidance
    if not params.scp.ctcs:
        g_bar_obs = []
        grad_g_bar_obs = []

        g_bar_vp = []
        grad_g_bar_vp = []

        g_bar_min = []
        grad_g_bar_min = []

        g_bar_max = []
        grad_g_bar_max = []
        
        for k in range(params.obs.n_obs):
            g_bar_obs.append(cp.Parameter(params.scp.n - 1, name='g_bar_obs_' + str(k)))
            grad_g_bar_obs.append(cp.Parameter((3, params.scp.n - 1), name='grad_g_bar_obs_' + str(k)))

        for k in range(params.vp.n_subs):
            g_bar_vp.append(cp.Parameter(params.scp.n, name='g_bar_vp_' + str(k)))
            grad_g_bar_vp.append(cp.Parameter((params.scp.n, params.sim.n_states), name='grad_g_bar_vp_' + str(k)))

            if params.vp.tracking:
                g_bar_min.append(cp.Parameter(params.scp.n, name='g_bar_min_' + str(k)))
                grad_g_bar_min.append(cp.Parameter((3, params.scp.n), name='grad_g_bar_min_' + str(k)))

                g_bar_max.append(cp.Parameter(params.scp.n, name='g_bar_max_' + str(k)))
                grad_g_bar_max.append(cp.Parameter((3, params.scp.n), name='grad_g_bar_max_' + str(k)))

    # Linearized Augmented Dynamics Constraints for CTCS
    A_prop = cp.Parameter(((params.sim.n_states)*(params.sim.n_states), params.scp.n - 1), name='A_d')
    B_prop = cp.Parameter((params.sim.n_states*params.sim.n_controls, params.scp.n - 1), name='B_d')
    C_prop = cp.Parameter((params.sim.n_states*params.sim.n_controls, params.scp.n - 1), name='C_d')
    z_prop = cp.Parameter((params.sim.n_states, params.scp.n - 1), name='z_d')
    nu = cp.Variable((params.sim.n_states, params.scp.n - 1), name='nu') # Virtual Control

    # Applying the affine scaling to state and control
    x_nonscaled = []
    u_nonscaled = []
    t = [0]
    for k in range(params.scp.n):
        x_nonscaled.append(S_x @ x[:,k] + c_x)
        u_nonscaled.append(S_u @ u[:,k] + c_u)
        if k > 0:
            s_cur = u_nonscaled[k][-1]
            s_prev = u_nonscaled[k-1][-1]
            if params.scp.dis_type == 'ZOH':
                t.append(t[k-1] + (tau[k] - tau[k-1]) * s_prev)
            else:
                t.append(t[k-1] + 0.5 * (s_cur + s_prev) * (tau[k] - tau[k-1]))

    # Gate Parameters
    if params.racing.n_gates > 0:
        A_gates = cp.Parameter((3 * params.racing.n_gates, 3), name='A_gates')
        mult_A_cen = cp.Parameter((3, params.racing.n_gates), name='mult_A_cen')
        gate_nodes = params.racing.gate_nodes

    
    #############
    # CONSTRAINTS
    #############
    
    # Gate Constraints
    i = 0
    constr = []
    if params.racing.n_gates > 0:
        for node in gate_nodes:
            constr += [cp.norm(A_gates[(3*i):3*(i+1), :] @ x_nonscaled[node][:3] - mult_A_cen[:,i], "inf") <= 1]
            i += 1
    
    constr += [t[-1] >= 1E-1, t[-1] <= 1E3] # Time Boundary Constraint
    constr += [x_nonscaled[0][:6] == params.sim.initial_state[:6]] # Initial Boundary Conditions
    if not params.vp.tracking:
        constr += [x_nonscaled[-1][:3] == params.sim.final_state[:3]] # Final Boundary Conditions
    constr += [t[0] == 0] # Initial Time    
    if params.scp.uniform_time_grid:
        constr += [t[i] - t[i-1] == t[i-1] - t[i-2] for i in range(2, params.scp.n)] # Uniform Time Step

    if params.scp.fixed_final_time:
        constr += [t[-1] == params.sim.total_time]

    constr += [0 == la.inv(S_x) @ (x_nonscaled[i] - x_bar[:,i] - dx[:,i]) for i in range(params.scp.n)] # State Error
    constr += [0 == la.inv(S_u) @ (u_nonscaled[i] - u_bar[:,i] - du[:,i]) for i in range(params.scp.n)] # Control Error

    constr += [t[i] - t[i-1] <= params.sim.max_dt for i in range(1, params.scp.n)] # Maximum Time Step
    constr += [t[i] - t[i-1] >= params.sim.min_dt for i in range(1, params.scp.n)] # Minimum Time Step

    constr += [x_nonscaled[i] == \
                    cp.reshape(A_prop[:,i-1], (params.sim.n_states, params.sim.n_states)) @ x_nonscaled[i-1] \
                    + cp.reshape(B_prop[:,i-1], (params.sim.n_states, params.sim.n_controls)) @ u_nonscaled[i-1] \
                    + cp.reshape(C_prop[:,i-1], (params.sim.n_states, params.sim.n_controls)) @ u_nonscaled[i] \
                    + z_prop[:,i-1] \
                    + nu[:,i-1] for i in range(1, params.scp.n)] # Dynamics Constraint
    
    constr += [u_nonscaled[i] <= params.sim.max_control for i in range(params.scp.n)]
    constr += [u_nonscaled[i] >= params.sim.min_control for i in range(params.scp.n)] # Control Constraints

    constr += [x_nonscaled[i][:-1] <= params.sim.max_state[:-1] for i in range(params.scp.n)]
    constr += [x_nonscaled[i][:-1] >= params.sim.min_state[:-1] for i in range(params.scp.n)] # State Constraints (Also implemented in CTCS but included for numerical stability)

    ########
    # COSTS
    ########
    cost = 0
    
    if not params.scp.fixed_final_time:
        cost = lam_t * t[-1] # Minimal Time Cost

    cost += sum(w_tr * cp.sum_squares(sla.block_diag(la.inv(S_x), la.inv(S_u)) @ cp.hstack((dx[:,i], du[:,i]))) for i in range(params.scp.n)) # Trust Region Cost

    if params.scp.min_fuel:
        cost += lam_fuel * x_nonscaled[-1][-2] # Minimal Fuel Cost

    cost += sum(params.scp.lam_vc * cp.sum(cp.abs(nu[:-1, i-1])) for i in range(1, params.scp.n)) # Virtual Control Slack
    
    if params.scp.ctcs:
        cost += sum(params.scp.lam_vc_ctcs * cp.sum(cp.abs(nu[-1, i-1])) for i in range(1, params.scp.n)) # Virtual Control CTCS Slack
        constr += [cp.abs(x_nonscaled[i][-1] - x_nonscaled[i-1][-1]) <= params.sim.max_state[-1] for i in range(1, params.scp.n)] # LICQ Constraint
        constr += [x_nonscaled[0][-1] == 0]

    else:
        for k in range(params.obs.n_obs):
            cost += sum(params.scp.lam_o * cp.pos(g_bar_obs[k][i-1] + grad_g_bar_obs[k][:,i-1].T @ dx[:3,i-1]) for i in range(params.scp.n)) # Nodal Obstacle Avoidance Cost
        
        for k in range(params.vp.n_subs):
            cost += sum(params.scp.lam_vp * cp.pos(g_bar_vp[k][i-1] + grad_g_bar_vp[k][i-1].T @ dx[:,i-1]) for i in range(params.scp.n)) # Viewplanning Virtual Buffer Cost
            if params.vp.tracking:
                cost += sum(params.scp.lam_min * cp.pos(g_bar_min[k][i-1] + grad_g_bar_min[k][:,i-1].T @ dx[:3,i-1]) for i in range(params.scp.n)) # Minimal Range Virtual Buffer Cost
                cost += sum(params.scp.lam_max * cp.pos(g_bar_max[k][i-1] + grad_g_bar_max[k][:,i-1].T @ dx[:3,i-1]) for i in range(params.scp.n)) # Maximal Range Virtual Buffer Cost

    #########
    # PROBLEM
    #########
    prob = cp.Problem(cp.Minimize(cost), constr)
    return prob