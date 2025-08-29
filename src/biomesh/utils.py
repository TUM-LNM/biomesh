# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Some small utilities."""

import numpy as np
import scipy.spatial as sp


def bislerp(
    Q_A: sp.transform.Rotation, Q_B: sp.transform.Rotation, t: np.ndarray
) -> sp.transform.Rotation:
    """Bidirectional-spherical linear interpolation between two rotation
    matrices.

    Args:
        Q_A: The first set of rotation matrices.
        Q_B: The second set of rotation matrices.
        t: The interpolation parameter(s).

    Returns:
        The interpolated rotation matrices.
    """

    q_a = Q_A.as_quat()
    q_b = Q_B.as_quat()

    norm_max = np.zeros((len(Q_A)))
    max_rotations = q_a

    for i in range(4):
        for pm in [-1, 1]:
            q_help = sp.transform.Rotation.from_quat(pm * np.eye(4)[i])

            q_m = (q_help * Q_A).as_quat()
            norm = np.abs(np.sum(q_m * q_b, axis=1))

            max_rotations[norm > norm_max] = q_m[norm > norm_max]
            norm_max[norm > norm_max] = norm[norm > norm_max]

    r_a_bar = sp.transform.Rotation.from_quat(max_rotations)

    interp = sp.transform.Rotation.concatenate(
        [
            sp.transform.Slerp(
                [0.0, 1.0], sp.transform.Rotation.concatenate([q_a_i, q_b_i])
            )(t_i)
            for q_a_i, q_b_i, t_i in zip(r_a_bar, Q_B, t)
        ]
    )
    return interp
