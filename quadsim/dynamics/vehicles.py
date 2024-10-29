import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jit, vmap

from quadsim.config import Config
from .utils import (
    qdcm,
    qdcm_vec,
    jax_qdcm,
    jax_skew_symetric_matrix,
    jax_skew_symetric_matrix_quat,
    skew_symetric_matrix_quat_vec,
    skew_symetric_matrix_vec,
)
from .constraints import Subject_JAX


class Quad_JAX:
    def __init__(self, params):
        self.aug_dy = vmap(
            jit(self.augmented_dynamics_jax), in_axes=(0, 0, 0, None, None, None)
        )
        self.sub = Subject_JAX(params)
    
    def fuel_jacobian(self, u, params):
        return u / np.linalg.norm(u, axis = 1)[:, None]

    def aug_dy_jax_helper(
        self,
        tau,
        x: jnp.ndarray,
        u: jnp.ndarray,
        time: float,
        obstacles: list,
        subs: list,
        sub_jax,
        params: dict,
    ) -> jnp.ndarray:
        """
        Helper function to run augmented_dynamics_jax with the correct input arguments.
        """
        # Subject Constraints
        p_s_I = []
        for sub in subs:
            p_s_I.append(sub.get_position(time, False).T)
        p_s_I = jnp.array(p_s_I)

        if params.scp.ctcs:
            y_dot = np.zeros(x.shape[0])
            for obs in obstacles:
                y_dot += np.maximum(0, obs.g_bar_ctcs(x[:, :3])) ** 2
            y_dot += sub_jax.g_bar_vp_helper(time, x, subs, params)

            for sub in subs:
                if params.vp.tracking:
                    y_dot += (
                        np.maximum(0, sub.g_bar_sub_min_ctcs(time, x, params, "Full"))
                        ** 2
                    )
                    y_dot += (
                        np.maximum(0, sub.g_bar_sub_max_ctcs(time, x, params, "Full"))
                        ** 2
                    )

            # Subtract max_state and min_state from state_seq along axis 1
            diff_max = (x - params.sim.max_state)[:, :-1]
            diff_min = (params.sim.min_state - x)[:, :-1]

            # Apply maximum operation with 0 and square the result
            sq_max = np.maximum(0, diff_max) ** 2
            sq_min = np.maximum(0, diff_min) ** 2

            # Sum the results along axis 1
            y_dot += np.sum(sq_max, axis=1)
            y_dot += np.sum(sq_min, axis=1)

        # Call augmented_dynamics_jax using vmap
        return self.aug_dy(
            x, u, y_dot[:, None], params.sim.m, params.sim.g, params.sim.J_b
        )

    def augmented_dynamics_jax(
        self,
        x: jnp.ndarray,
        u: jnp.ndarray,
        y_dot: jnp.ndarray,
        m: float,
        g: float,
        J_b: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes the augmented dynamics of a system with obstacles. Not to be used when propogating the system, use nonlinear_dynamics for this.

        Parameters:
        t (float): Current time.
        state (np.ndarray): Current state of the system, including position, velocity, attitude (quaternion), and angular rate.
        control_current (np.ndarray): Current control input.
        control_slope (np.ndarray): Slope of the control input for first-order hold (FOH) discretization.
        t_start (float): Start time of the current control interval.
        prop (bool): Whether to propagate the dynamics forward in time.
        obstacles: List of obstacles in the environment.
        params (dict): Parameters of the system, including discretization type ('dis_type'), whether to use exact discretization ('dis_exact'), mass ('m'), gravity ('g'), inertia tensor ('J_b'), and whether to use CTCS augmentation ('ctcs').

        Returns:
        np.ndarray: Time derivative of the augmented state, including the state of the system and the CTCS augmentation for each obstacle.
        """

        # Extract the State Variables
        r_i = x[:3]  # Position
        v_i = x[3:6]  # Velocity
        q_bi = x[6:10]  # Attitude (Quaternion)
        w_b = x[10:13]  # Angular Rate

        # Extract the control inputs
        u_mB = u[:3]  # Applied Force
        tau_i = u[3:6]  # Applied Torque

        # Ensure that the quaternion is normalized
        q_norm = jnp.linalg.norm(q_bi)
        q_bi = q_bi / q_norm

        # Compute the time derivatives of the state variables
        p_i_dot = v_i
        v_i_dot = (1 / m) * jax_qdcm(q_bi) @ (u_mB) + jnp.array([0, 0, g])

        q_bi_dot = 0.5 * jax_skew_symetric_matrix_quat(w_b) @ q_bi
        w_b_dot = jnp.diag(1 / J_b) @ (
            tau_i - jax_skew_symetric_matrix(w_b) @ jnp.diag(J_b) @ w_b
        )

        # Combine the time derivatives of the state variables and the CTCS augmentations into a single array
        state_dot = jnp.hstack([p_i_dot, v_i_dot, q_bi_dot, w_b_dot])

        augmented_state_dot = jnp.hstack(
            [state_dot, y_dot]
        )  # Augment the state with the CTCS augmentation

        return augmented_state_dot

def augmented_dynamics_vec(
    tau: float,
    state_seq: np.ndarray,
    control_current_seq: np.ndarray,
    control_next_seq: np.ndarray,
    tau_init: float,
    prop: bool,
    time: float,
    obstacles,
    subs,
    sub_jax,
    params: dict,
) -> np.ndarray:
    """
    Computes the augmented dynamics of a system with obstacles for a sequence of states.

    Parameters:
    t (float): Current time.
    state_seq (np.ndarray): Sequence of states of the system.
    control_current_seq (np.ndarray): Sequence of current control inputs.
    control_slope_seq (np.ndarray): Sequence of slopes of the control inputs for first-order hold (FOH) discretization.
    t_start (float): Start time of the current control interval.
    prop (bool): Whether to propagate the dynamics forward in time.
    obstacles: List of obstacles in the environment.
    params (dict): Parameters of the system.

    Returns:
    np.ndarray: Time derivatives of the augmented states.
    """
    if not prop:
        control_seq = control_current_seq
        single = False
        t = time
    else:
        state_seq = state_seq[None, :]
        if params.scp.dis_type == "ZOH":
            beta = 0.0
        elif params.scp.dis_type == "FOH":
            beta = (tau - tau_init) * params.scp.n
        control_seq = control_current_seq + beta * (
            control_next_seq - control_current_seq
        )
        single = True

        if params.scp.dis_type == 'ZOH':
            t = time + (tau - tau_init) * control_seq[:,-1]
        elif params.scp.dis_type == 'FOH':
            t = time + 0.5 * np.array(control_seq[:,-1] + control_current_seq[:,-1]) * (tau - tau_init)
            

    # Extract the State Variables
    r_i_seq = state_seq[:, :3]  # Position
    v_i_seq = state_seq[:, 3:6]  # Velocity
    q_bi_seq = state_seq[:, 6:10]  # Attitude (Quaternion)
    w_b_seq = state_seq[:, 10:13]  # Angular Rate

    # Compute the CTCS augmentation
    y_dot_seq = np.zeros(state_seq.shape[0])

    if params.scp.ctcs:
        for obs in obstacles:
            y_dot_seq += np.maximum(0, obs.g_bar_ctcs(r_i_seq)) ** 2
        y_dot_seq += np.sum(
            np.maximum(
                0, sub_jax.g_bar_vp_helper(t, state_seq, subs, single, params)
            )
            ** 2,
            axis=0,
        )
        for sub in subs:
            if params.vp.tracking:
                if single:
                    y_dot_seq += (
                        np.maximum(
                            0, sub.g_bar_sub_min_ctcs(t, state_seq, params, "Single")
                        )
                        ** 2
                    )
                    y_dot_seq += (
                        np.maximum(
                            0, sub.g_bar_sub_max_ctcs(t, state_seq, params, "Single")
                        )
                        ** 2
                    )
                else:
                    y_dot_seq += (
                        np.maximum(
                            0, sub.g_bar_sub_min_ctcs(t, state_seq, params, "Full")
                        )
                        ** 2
                    )
                    y_dot_seq += (
                        np.maximum(
                            0, sub.g_bar_sub_max_ctcs(t, state_seq, params, "Full")
                        )
                        ** 2
                    )

        # Subtract max_state and min_state from state_seq along axis 1
        diff_max = (state_seq - params.sim.max_state)[:, :-1]
        diff_min = (params.sim.min_state - state_seq)[:, :-1]

        # Apply maximum operation with 0 and square the result
        sq_max = np.maximum(0, diff_max) ** 2
        sq_min = np.maximum(0, diff_min) ** 2

        # Sum the results along axis 1
        y_dot_seq += np.sum(sq_max, axis=1)
        y_dot_seq += np.sum(sq_min, axis=1)

    # Extract the control inputs
    u_mB_seq = control_seq[:, :3]  # Applied Force
    tau_i_seq = control_seq[:, 3:6]  # Applied Torque

    # Ensure that the quaternion is normalized
    q_norm_seq = np.linalg.norm(q_bi_seq, axis=1, keepdims=True)
    q_bi_seq = q_bi_seq / q_norm_seq

    # Compute the time derivatives of the state variables
    p_i_dot_seq = v_i_seq
    v_i_dot_seq = (1 / params.sim.m) * np.einsum(
        "ijk,ikl->ij", qdcm_vec(q_bi_seq), u_mB_seq[:, :, None]
    ) + np.array([0, 0, params.sim.g])

    q_bi_dot_seq = 0.5 * np.einsum(
        "ijk,ikl->ij", skew_symetric_matrix_quat_vec(w_b_seq), q_bi_seq[:, :, None]
    )
    w_b_dot_seq = np.einsum(
        "ij,kj->ki",
        np.diag(1 / params.sim.J_b),
        (
            tau_i_seq
            - np.einsum(
                "ijk,ikl->ij",
                skew_symetric_matrix_vec(w_b_seq),
                np.einsum("ij,kj->ki", np.diag(params.sim.J_b), w_b_seq)[:, :, None],
            )
        ),
    )


    # Combine the time derivatives of the state variables and the CTCS augmentations into a single array
    state_dot_seq = np.hstack([p_i_dot_seq, v_i_dot_seq, q_bi_dot_seq, w_b_dot_seq])

    # 2 Norm of fuel
    if params.scp.min_fuel:
        fuel = np.linalg.norm(np.hstack((u_mB_seq, tau_i_seq)), axis = 1)
        # Concatenate this to the state
        state_dot_seq = np.hstack([state_dot_seq, fuel[:, None]])

    augmented_state_dot_seq = np.hstack(
        [state_dot_seq, y_dot_seq[:, None]]
    )  # Augment the state with the CTCS augmentation

    if single:
        augmented_state_dot_seq = control_seq[:, 6] * augmented_state_dot_seq.squeeze()
    return augmented_state_dot_seq


def auto_jacobians_vec(params: dict):
    """
    Computes the analytical Jacobians of the system dynamics.

    Parameters:
    params (dict): Parameters of the system, including mass ('m'), gravity ('g'), and inertia tensor ('J_b').

    Returns:
    A, B: Functions that compute the Jacobians of the system dynamics with respect to the state and the control input, respectively.
    """

    @jit
    def jax_skew_symmetric_matrix(w: jnp.ndarray):
        # Convert an angular rate to a 3 x 3 skew symmetric matrix
        x, y, z = w
        return jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    @jit
    def jax_skew_symmetric_matrix_quat(w: jnp.ndarray):
        # Convert an angular rate to a 4 x 4 skew symmetric matrix
        x, y, z = w
        return jnp.array([[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]])

    def state_dot_func(x, u):
        # Ensure that the inputs are JAX arrays
        x, u = map(jnp.asarray, [x, u])

        # Unpack the state and control vectors
        _, v, q, w = jnp.split(x, indices_or_sections=[3, 6, 10])
        f, tau = jnp.split(u, indices_or_sections=[3])

        # Compute the time derivatives of the state variables
        r_dot = v
        v_dot = (1 / params.sim.m) * jax_qdcm(q) @ f + jnp.array([0, 0, params.sim.g])
        q_dot = 0.5 * jax_skew_symmetric_matrix_quat(w) @ q
        w_dot = jnp.linalg.inv(jnp.diag(params.sim.J_b)) @ (
            tau - jax_skew_symmetric_matrix(w) @ jnp.diag(params.sim.J_b) @ w
        )

        # Combine the time derivatives of the state variables into a single array
        return jnp.concatenate([r_dot, v_dot, q_dot, w_dot])

    # Compute the Jacobians of the system dynamics with respect to the state and the control input
    A = jit(vmap(jacfwd(state_dot_func, argnums=0)))
    B = jit(vmap(jacfwd(state_dot_func, argnums=1)))
    return A, B


def auto_jacobians(params: dict):
    """
    Computes the analytical Jacobians of the system dynamics.

    Parameters:
    params (dict): Parameters of the system, including mass ('m'), gravity ('g'), and inertia tensor ('J_b').

    Returns:
    A, B: Functions that compute the Jacobians of the system dynamics with respect to the state and the control input, respectively.
    """

    def jax_skew_symmetric_matrix(w: np.ndarray):
        # Convert an angular rate to a 3 x 3 skew symmetric matrix
        x, y, z = w
        return jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

    def jax_skew_symmetric_matrix_quat(w: np.ndarray):
        # Convert an angular rate to a 4 x 4 skew symmetric matrix
        x, y, z = w
        return jnp.array([[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]])

    def state_dot_func(x, u):
        # Ensure that the inputs are JAX arrays
        x, u = map(jnp.asarray, [x, u])

        # Unpack the state and control vectors
        _, v, q, w = jnp.split(x, indices_or_sections=[3, 6, 10])
        f, tau = jnp.split(u, indices_or_sections=[3])

        # Compute the time derivatives of the state variables
        r_dot = v
        v_dot = (1 / params.sim.m) * jax_qdcm(q) @ f + jnp.array([0, 0, params.sim.g])
        q_dot = 0.5 * jax_skew_symmetric_matrix_quat(w) @ q
        w_dot = jnp.linalg.inv(jnp.diag(params.sim.J_b)) @ (
            tau - jax_skew_symmetric_matrix(w) @ jnp.diag(params.sim.J_b) @ w
        )

        # Combine the time derivatives of the state variables into a single array
        return jnp.concatenate([r_dot, v_dot, q_dot, w_dot])

    # Compute the Jacobians of the system dynamics with respect to the state and the control input
    A = jit(jacfwd(state_dot_func, argnums=0))
    B = jit(jacfwd(state_dot_func, argnums=1))
    return A, B