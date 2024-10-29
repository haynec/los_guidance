import numpy as np
import numpy.linalg as la
import jax.numpy as jnp
from jax import jacfwd, jit, vmap

from quadsim.config import Config
from .utils import qdcm, qdcm_vec, jax_qdcm


class Planar_Rectangular_Gate:
    def __init__(self, center, node):
        # center = np.zeros(3)
        # center[:2] = 20 * np.random.rand(2) - 10 * np.ones(2)
        center[0] = center[0] + 2.5
        center[2] = center[2] + 2.5
        self.center = center
        rot = np.array(
            [
                [np.cos(np.pi / 2), np.sin(np.pi / 2), 0],
                [-np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
                [0, 0, 1],
            ]
        )
        self.axes = rot  # self.generate_orthogonal_unit_vectors()
        rad = np.random.rand(2) + np.ones(2)
        radii = np.zeros(3)
        radii[0] = np.sqrt(2.5)  # rad[0]
        radii[1] = 1e-2
        radii[2] = np.sqrt(2.5)  # rad[1]
        self.radius = radii**2
        A = np.diag(1 / self.radius)
        self.A = self.axes @ A @ self.axes.T
        self.normal = self.normal_vector()
        self.node = node
        self.vertices = self.gen_vertices()

    def generate_orthogonal_unit_vectors(self):
        """
        Generates 3 orthogonal unit vectors to model the axis of the ellipsoid.

        Returns:
        np.ndarray: A 3x3 matrix where each column is a unit vector.
        """
        vectors = np.random.rand(3, 3)
        Q, _ = np.linalg.qr(vectors)
        return Q

    def normal_vector(self):
        """
        Obtains the vector normal to the plane of the gate.
        """
        return self.axes[:, 1]

    def gen_vertices(self):
        """
        Obtains the vertices of the gate.
        """
        vertices = []
        vertices.append(self.center + self.axes @ [self.radius[0], 0, self.radius[2]])
        vertices.append(self.center + self.axes @ [-self.radius[0], 0, self.radius[2]])
        vertices.append(self.center + self.axes @ [-self.radius[0], 0, -self.radius[2]])
        vertices.append(self.center + self.axes @ [self.radius[0], 0, -self.radius[2]])
        return vertices


class Ellipsoidal_Obstacle:
    def __init__(self, center, axes=None, radius=None):
        raise NotImplementedError("Ellipsoidal Obstacles are not implemented.")

class Subject:
    def __init__(self, init_pose, params):
        self.state = None
        self.traj = None
        self.full_traj = None

        self.init_pose = init_pose

        self.init_pose_jax = jnp.array(init_pose)

        self.tracking = params.vp.tracking

        self.auto_grad_p_s_s(params)

    def get_position(self, t: float, single) -> np.ndarray:
        self.loop_time = 40.0
        self.loop_radius = 20.0

        t_angle = t / self.loop_time * (2 * np.pi)
        x = self.loop_radius * np.sin(t_angle)
        y = x * np.cos(t_angle)
        z = 0.5 * x * np.sin(t_angle)
        if self.tracking:
            if single:
                return (
                    np.array([x.squeeze(), y.squeeze(), z.squeeze()])
                    + np.array([13, 0, 2])
                )[None, :]
            else:
                return (
                    np.repeat(
                        np.expand_dims(np.array([13, 0, 2]), axis=1), len(t), axis=1
                    )
                    + np.array([x, y, z])
                ).T
        else:
            if not single:
                return np.repeat(
                    np.expand_dims(self.init_pose, axis=1), len(t), axis=1
                ).T
            else:
                return self.init_pose[None, :]

    def get_position_jax(self, t: jnp.ndarray) -> jnp.ndarray:
        self.loop_time = 40.0
        self.loop_radius = 20.0

        t_angle = t / self.loop_time * (2 * jnp.pi)
        x = self.loop_radius * jnp.sin(t_angle)
        y = x * jnp.cos(t_angle)
        z = 0.5 * x * jnp.sin(t_angle)
        if self.tracking:
            return jnp.repeat(
                jnp.expand_dims(jnp.array([13, 0, 2]), axis=1), len(t), axis=1
            ) + jnp.array([x, y, z])
        else:
            return jnp.repeat(
                jnp.expand_dims(self.init_pose_jax, axis=1), len(t), axis=1
            )

    def get_velocity(self, t: float, dt: float) -> np.ndarray:
        return (self.get_position(t + dt) - self.get_position(t)) / dt

    def get_state(self, t: float, dt: float) -> np.ndarray:
        return np.hstack([self.get_position(t), self.get_velocity(t, dt)])

    def g_bar_sub_min_ctcs(self, t, x, params, type):
        # Minimum Range Constraint to Subject
        if type == "Single":
            p = x[:, :3]
            return params.vp.min_range - np.linalg.norm(p - self.get_position(t, True))
        else:
            t = np.array(t)
            p = x[:, :3]
            return params.vp.min_range - np.linalg.norm(
                p - self.get_position(t, False), axis=1
            )

    def grad_g_bar_sub_min_ctcs(self, t, x, params, type):
        # Jacobian of the Minimum Range Constraint to Subject
        if type == "Single":
            p = x[:3]
            return (
                -2
                * max(0, self.g_bar_sub_min_ctcs(t, x, params, type))
                * (p - self.get_position(t, True))
                / np.linalg.norm(p - self.get_position(t, True))
            )
        else:
            t = np.array(t)
            p = x[:, :3]
            p_s_I = self.get_position(t, False)
            return (
                -2
                * np.maximum(0, self.g_bar_sub_min_ctcs(t, x, params, type))[:, None]
                * (p - p_s_I)
                / np.linalg.norm(p - p_s_I, axis=0)
            )
    
    def grad_g_bar_sub_min_nodal(self, t, x, params, type):
        # Jacobian of the Minimum Range Constraint to Subject
        if type == "Single":
            p = x[:3]
            return (
                - (p - self.get_position(t, True))
                / np.linalg.norm(p - self.get_position(t, True))
            )
        else:
            t = np.array(t)
            p = x[:, :3]
            p_s_I = self.get_position(t, False)
            return (
                - (p - p_s_I)
                / np.linalg.norm(p - p_s_I, axis=0)
            )

    def g_bar_sub_max_ctcs(self, t, x, params, type):
        # Maximum Range Constraint to Subject
        if type == "Single":
            p = x[:, :3]
            return np.linalg.norm(p - self.get_position(t, True)) - params.vp.max_range
        else:
            t = np.array(t)
            p = x[:, :3]
            return (
                np.linalg.norm(p - self.get_position(t, False), axis=1)
                - params.vp.max_range
            )

    def grad_g_bar_sub_max_ctcs(self, t, x, params, type):
        # Jacobian of the Maximum Range Constraint to Subject
        if type == "Single":
            p = x[:3]
            return (
                2
                * max(0, self.g_bar_sub_max_ctcs(t, x, params, type))
                * (p - self.get_position(t, True))
                / np.linalg.norm(p - self.get_position(t, True))
            )
        else:
            t = np.array(t)
            p = x[:, :3]
            p_s_I = self.get_position(t, False)
            return (
                2
                * np.maximum(0, self.g_bar_sub_max_ctcs(t, x, params, type))[:, None]
                * (p - p_s_I)
                / np.linalg.norm(p - p_s_I, axis=0)
            )
        
    def grad_g_bar_sub_max_nodal(self, t, x, params, type):
        # Jacobian of the Maximum Range Constraint to Subject
        if type == "Single":
            p = x[:3]
            return (
                (p - self.get_position(t, True))
                / np.linalg.norm(p - self.get_position(t, True))
            )
        else:
            t = np.array(t)
            p = x[:, :3]
            p_s_I = self.get_position(t, False)
            return (
                (p - p_s_I)
                / np.linalg.norm(p - p_s_I, axis=0)
            )

    def auto_vp_jacobian(self, params):
        # Analytical Jacobian of Rotational Constraint
        R_sb = params.vp.R_sb  # Rotation Matrix from Sensor to Body Frame

        def p_s_s_func(x, p_s_I):
            return jnp.dot(
                jnp.dot(R_sb, jax_qdcm(x[6:10]).T), (p_s_I - x[0:3])
            )  # Subject Position in Sensor Frame

        self.auto_grad_p_s_s_func = vmap(jacfwd(p_s_s_func))

    def g_bar_vp_ctcs(self, t, x, params, type):
        # Viewplanning Constraint to Subject
        # Jacobian of the Viewplanning Constraint

        if type == "Full":
            t = np.array(t)

        A = np.diag(
            [
                1 / np.tan(np.pi / params.vp.alpha_x),
                1 / np.tan(np.pi / params.vp.alpha_y),
                0,
            ]
        )  # Conic Matrix
        c = np.array([0, 0, 1])  # Boresight Vector in Sensor Frame

        if type == "Single":
            p_s_I = self.get_position(t, True)
        else:
            p_s_I = self.get_position(t, False).T  # Subject Position in Inertial Frame

        R_sb = params.vp.R_sb  # Rotation Matrix from Sensor to Body Frame

        if type == "Full":
            p_s_s = np.squeeze(
                R_sb
                @ qdcm_vec(x[:, 6:10]).transpose((0, 2, 1))
                @ (p_s_I[:, :, None] - x[:, 0:3, None])
            )
            if params.vp.norm == "inf":
                g_vp = la.norm(
                    np.einsum("ij,kj->ki", A.T, p_s_s), ord=np.inf
                ) - np.squeeze(c[None, :] @ p_s_s.T)
            else:
                g_vp = (
                    la.norm(
                        np.einsum("ij,kj->ki", A.T, p_s_s), ord=params.vp.norm, axis=1
                    )
                    - np.matmul(c[:, None].T, p_s_s.T)
                ).T
        else:
            p_s_s = R_sb @ qdcm(x[6:10]).T @ (p_s_I - x[0:3]).T
            if params.vp.norm == "inf":
                g_vp = la.norm(A @ p_s_s, ord=np.inf) - (c.T @ p_s_s)
            else:
                g_vp = la.norm(A @ p_s_s, ord=params.vp.norm) - (c.T @ p_s_s)
        return g_vp

    def grad_g_bar_vp_ctcs(self, t, x, params, type):
        # Jacobian of the Viewplanning Constraint

        if type == "Full":
            t = np.array(t)

        A = np.diag(
            [
                1 / np.tan(np.pi / params.vp.alpha_x),
                1 / np.tan(np.pi / params.vp.alpha_y),
                0,
            ]
        )  # Conic Matrix
        c = np.array([0, 0, 1])  # Boresight Vector in Sensor Frame

        if type == "Single":
            p_s_I = self.get_position(t, True)
        else:
            p_s_I = self.get_position(t, False).T  # Subject Position in Inertial Frame

        R_sb = params.vp.R_sb  # Rotation Matrix from Sensor to Body Frame

        if type == "Full":
            p_s_s = np.squeeze(
                R_sb
                @ qdcm_vec(x[:, 6:10]).transpose((0, 2, 1))
                @ (p_s_I[:, :, None] - x[:, 0:3, None])
            )
            grad_p_s_s = self.auto_grad_p_s_s_func(x, p_s_I)
        else:
            p_s_s = R_sb @ qdcm(x[6:10]).T @ (p_s_I - x[0:3])
            # grad_p_s_s = self.grad_p_s_s_func(np.expand_dims(x, axis = 1), np.expand_dims(p_s_I, axis = 1))
            grad_p_s_s = self.auto_grad_p_s_s_func(x[None, :], p_s_I[None, :])

        if type == "Single":
            if params.vp.norm == "inf":
                grad_vp = (
                    (A.T @ A @ p_s_s) / (la.norm(A @ p_s_s, ord=np.inf)) - c.T
                ) @ grad_p_s_s
            else:
                grad_vp = (
                    (A.T @ A @ p_s_s) / (la.norm(A @ p_s_s, ord=params.vp.norm)) - c.T
                ) @ grad_p_s_s
        else:
            if params.vp.norm == "inf":
                grad_vp = np.einsum(
                    "ij,jik->jk",
                    (
                        (A.T @ A @ p_s_s.T)
                        / (la.norm(A @ p_s_s.T, ord=np.inf, axis=0))[None, :]
                        - c[:, None]
                    ),
                    grad_p_s_s,
                )
            else:
                grad_vp = np.zeros((21, 14))
                for k in range(params.scp.n - 1):
                    grad_vp[k] = (
                        (A.T @ A @ p_s_s[k])
                        / (la.norm(A @ p_s_s[k], ord=params.vp.norm))
                        - c.T
                    ) @ grad_p_s_s[k]
                # grad_vp = np.einsum('ijk,ikl->ijl', (np.einsum('ij,kj->ki', A.T @ A , p_s_s)/(la.norm(np.einsum('ij,kj->ki', A , p_s_s), ord = params.vp.norm, axis = 1))[:,None] - c)[:,None, :], grad_p_s_s)

        if type == "Single":
            return np.squeeze(
                2 * max(0, self.g_bar_vp_ctcs(t, x, params, type)) * grad_vp
            )
        else:
            grad_vp_full = (
                2 * np.maximum(0, self.g_bar_vp_ctcs(t, x, params, type)) * grad_vp
            )
            # for i in range(len(grad_vp)):
            #     grad_vp_full.append(2 * max(0, self.g_bar_vp_ctcs(t, x, params, type)[i]) * grad_vp[i])
            return grad_vp_full  # np.array(grad_vp_full)

    def g_bar_vp_ctcs_org_vec(self, t, x, params):
        # Viewplanning Constraint to Subject
        # Jacobian of the Viewplanning Constraint

        A = np.diag(
            [
                1 / np.tan(np.pi / params.vp.alpha_x),
                1 / np.tan(np.pi / params.vp.alpha_y),
                0,
            ]
        )  # Conic Matrix
        c = np.array([0, 0, 1])  # Boresight Vector in Sensor Frame

        p_s_I = self.get_position(t, False)  # Subject Position in Inertial Frame

        R_sb = params.vp.R_sb  # Rotation Matrix from Sensor to Body Frame

        p_s_s = np.einsum(
            "ijk,ik->ij",
            np.matmul(R_sb, np.transpose(qdcm_vec(x[:, 6:10]), (0, 2, 1))),
            (p_s_I - x[:, 0:3]),
        )

        if params.vp.norm == "inf":
            g_vp = la.norm(A @ p_s_s.T, ord=np.inf, axis=0) - np.matmul(c.T, p_s_s.T)
        else:
            g_vp = la.norm(A @ p_s_s.T, ord=params.vp.norm, axis=0) - np.matmul(
                c.T, p_s_s.T
            )
        return g_vp

    def auto_grad_p_s_s(self, params):
        R_sb = params.vp.R_sb

        def compute(x, p_s_I):
            return jnp.dot(
                jnp.dot(R_sb, jax_qdcm(x[6:10]).T), (p_s_I - x[0:3])
            )  # Subject Position in Sensor Frame

        self.auto_grad_p_s_s_func = jit(jacfwd(compute))

    def grad_g_bar_vp_ctcs_org_vec(self, t, x, params: dict):
        # Jacobian of the Viewplanning Constraint

        A = np.diag(
            [
                1 / np.tan(np.pi / params.vp.alpha_x),
                1 / np.tan(np.pi / params.vp.alpha_y),
                0,
            ]
        )  # Conic Matrix
        c = np.array([0, 0, 1])  # Boresight Vector in Sensor Frame

        p_s_I = self.get_position(t, False)  # Subject Position in Inertial Frame

        R_sb = params.vp.R_sb  # Rotation Matrix from Sensor to Body Frame
        p_s_s = np.einsum(
            "ijk,ik->ij",
            np.matmul(R_sb, np.transpose(qdcm_vec(x[:, 6:10]), (0, 2, 1))),
            (p_s_I - x[:, 0:3]),
        )

        grad_p_s_s = []
        i = 0
        for p_s in p_s_I:
            grad_p_s_s.append(self.auto_grad_p_s_s_func(x[i, :], p_s))
            i += 1
        grad_p_s_s = np.array(grad_p_s_s)

        if params.vp.norm == "inf":
            # Compute the norm for each row in p_s_s
            norms = la.norm(A @ p_s_s.T, ord=np.inf, axis=0)

            # Compute the term (A.T @ A @ p_s_s) / norms
            term = (A.T @ A @ p_s_s.T) / norms

            # Subtract c.T from each column of term
            term = term.T - c

            # Compute grad_vp
            grad_vp = np.einsum("ij,ijk->ik", term, grad_p_s_s)
        else:
            # Compute the norm for each row in p_s_s
            norms = la.norm(A @ p_s_s.T, ord=params.vp.norm, axis=0)

            # Compute the term (A.T @ A @ p_s_s) / norms
            term = (A.T @ A @ p_s_s.T) / norms

            # Subtract c.T from each column of term
            term = term.T - c

            # Compute grad_vp
            grad_vp = np.einsum("ij,ijk->ik", term, grad_p_s_s)

        grad_vp_full = (
            grad_vp
        )
        return grad_vp_full


class Subject_JAX:
    def __init__(self, params):
        self.g_bar_func = jit(vmap(self.g_bar_vp_jax, in_axes=(0, 0, None, None, None)))
        self.grad_g_func = jit(
            vmap(self.grad_g_bar_vp_jax, in_axes=(0, 0, None, None, None, 0))
        )
        self.norm_type = np.inf if params.vp.norm == "inf" else params.vp.norm

    def g_bar_vp_helper(self, t, x, subs, single, params):
        # Extract params from the dictionary
        A_cone = np.diag(
            [
                1 / np.tan(np.pi / params.vp.alpha_x),
                1 / np.tan(np.pi / params.vp.alpha_y),
                0,
            ]
        )  # Conic Matrix
        norm_type = np.inf if params.vp.norm == "inf" else params.vp.norm
        R_sb = params.vp.R_sb  # Rotation Matrix from Sensor to Body Frame

        # g_bar = 0
        g_bar = []
        for i in range(params.vp.n_subs):
            p_s_I = subs[i].get_position(t, single)
            # g_bar += jnp.maximum(0, self.g_bar_func(p_s_I, x, A_cone, norm_type, R_sb)) ** 2
            g_bar.append(self.g_bar_func(p_s_I, x, A_cone, norm_type, R_sb))
        return np.array(g_bar)

    def g_bar_vp_jax(self, p_s_I, x, A, norm_type, R_sb) -> jnp.ndarray:
        # Viewplanning Constraint to Subject
        # Jacobian of the Viewplanning Constraint
        c = jnp.array([0, 0, 1])  # Boresight Vector in Sensor Frame

        p_s_s = R_sb @ jax_qdcm(x[6:10]).T @ (p_s_I - x[0:3])
        # norm_type = jnp.inf if params.vp.norm == 'inf' else params.vp.norm
        g_vp = jnp.linalg.norm(A @ p_s_s, ord=self.norm_type) - (c.T @ p_s_s)
        return g_vp

    def grad_g_bar_vp_helper(self, t, x, subs, params, g_bar_vp, p_s_I_vec):
        # Extract params from the dictionary
        A_cone = np.diag(
            [
                1 / np.tan(np.pi / params.vp.alpha_x),
                1 / np.tan(np.pi / params.vp.alpha_y),
                0,
            ]
        )  # Conic Matrix
        norm_type = np.inf if params.vp.norm == "inf" else params.vp.norm
        R_sb = params.vp.R_sb  # Rotation Matrix from Sensor to Body Frame

        grad_g_bar = 0
        for i in range(params.vp.n_subs):
            p_s_I = subs[i].get_position(t, False)
            grad_g_bar += self.grad_g_func(p_s_I, x, A_cone, norm_type, R_sb, g_bar_vp)
        return grad_g_bar

    def grad_g_bar_vp_jax(
        self,
        p_s_I,
        x: jnp.ndarray,
        A: jnp.ndarray,
        norm_type,
        R_sb: jnp.ndarray,
        g_bar_vp: jnp.ndarray,
    ) -> jnp.ndarray:
        # Jacobian of the Viewplanning Constraint
        c = jnp.array([0, 0, 1])  # Boresight Vector in Sensor Frame

        def grad_p_s_s_func(x_int, p_s_I):
            return jnp.dot(
                jnp.dot(R_sb, jax_qdcm(x_int[6:10]).T), (p_s_I - x_int[0:3])
            )  # Subject Position in Sensor Frame

        p_s_s = R_sb @ jax_qdcm(x[6:10]).T @ (p_s_I - x[0:3])
        grad_p_s_s = jacfwd(grad_p_s_s_func)(x, p_s_I)

        grad_vp = (
            (A.T @ A @ p_s_s) / jnp.linalg.norm(A @ p_s_s, ord=self.norm_type) - c.T
        ) @ grad_p_s_s
        return jnp.squeeze(
            2
            * jnp.maximum(0, self.g_bar_vp_jax(p_s_I, x, A, norm_type, R_sb))
            * grad_vp
        )
