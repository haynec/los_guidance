import numpy as np
import jax.numpy as jnp
from jax import jit


def qdcm(q: np.ndarray) -> np.ndarray:
    # Convert a quaternion to a direction cosine matrix
    q_norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
    w, x, y, z = q / q_norm
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def qdcm_vec(q: np.ndarray) -> np.ndarray:
    # Convert a quaternion to a direction cosine matrix with a vectorized input
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / q_norm
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    ).transpose((2, 0, 1))


@jit
def jax_qdcm(q: jnp.ndarray) -> jnp.ndarray:
    # Convert a quaternion to a direction cosine matrix
    q_norm = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2) ** 0.5
    w, x, y, z = q / q_norm
    # x, y, z, w = q
    return jnp.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def skew_symetric_matrix_quat(w: np.ndarray):
    # Convert an angular rate to a 4 x 4 skew symetric matrix
    x, y, z = w
    return np.array([[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]])


@jit
def jax_skew_symetric_matrix_quat(w: jnp.ndarray):
    # Convert an angular rate to a 4 x 4 skew symetric matrix
    x, y, z = w
    return jnp.array([[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]])


def skew_symetric_matrix_quat_vec(w: np.ndarray):
    # Convert an angular rate to a 4 x 4 skew symetric matrix with a vectorized input
    x, y, z = w[:, 0], w[:, 1], w[:, 2]
    zeros = np.zeros_like(x)
    return np.array(
        [[zeros, -x, -y, -z], [x, zeros, z, -y], [y, -z, zeros, x], [z, y, -x, zeros]]
    ).transpose((2, 0, 1))


def skew_symetric_matrix(w: np.ndarray):
    # Convert an angular rate to a 3 x 3 skew symetric matrix
    x, y, z = w
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


@jit
def jax_skew_symetric_matrix(w: jnp.ndarray):
    # Convert an angular rate to a 3 x 3 skew symetric matrix
    x, y, z = w
    return jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def skew_symetric_matrix_vec(w: np.ndarray):
    # Convert an angular rate to a 3 x 3 skew symetric matrix with a vectorized input
    x, y, z = w[:, 0], w[:, 1], w[:, 2]
    zeros = np.zeros_like(x)
    return np.array([[zeros, -z, y], [z, zeros, -x], [-y, x, zeros]]).transpose(
        (2, 0, 1)
    )
