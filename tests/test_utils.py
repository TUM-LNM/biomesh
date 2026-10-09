# This file is part of biomesh licensed under the MIT License.
#
# See the LICENSE file in the top-level for license information.
#
# SPDX-License-Identifier: MIT
"""Test small utilities."""

import numpy as np
import scipy.spatial as sp

import biomesh


def test_bislerp():
    """Test bislerp operation for rotations."""

    random_rotations_1 = sp.transform.Rotation.from_quat(
        np.array(
            [
                [0.10377005, 0.42208032, 0.06739728, 0.20368327],
                [0.42899313, 0.38149584, 0.25978406, 0.18940957],
                [0.44195921, 0.29562431, 0.3516844, 0.2188311],
                [0.17815841, 0.29713605, 0.31167307, 0.14250813],
                [0.44414069, 0.16290792, 0.16563452, 0.4993757],
                [0.26088402, 0.0641587, 0.27387306, 0.11629013],
                [0.37962083, 0.52176654, 0.28088421, 0.34948843],
                [0.02467308, 0.29008909, 0.10773388, 0.46944871],
                [0.17777532, 0.21832771, 0.62441441, 0.21813217],
                [0.36943482, 0.25673186, 0.35257723, 0.44881636],
            ]
        )
    )
    random_rotations_2 = sp.transform.Rotation.from_quat(
        np.array(
            [
                [0.27087052, 0.14185837, 0.32699565, 0.32047239],
                [0.1100865, 0.41477543, 0.4821505, 0.42659205],
                [0.18470968, 0.06425801, 0.4048248, 0.2020325],
                [0.4595524, 0.36257552, 0.01671866, 0.10775836],
                [0.42378484, 0.07148258, 0.09983605, 0.22052701],
                [0.12910392, 0.55275514, 0.09413712, 0.17272525],
                [0.04430284, 0.29633272, 0.13404752, 0.27092465],
                [0.50856929, 0.50803009, 0.36302255, 0.02316184],
                [0.28272705, 0.07699512, 0.14529143, 0.56637091],
                [0.36387167, 0.09881067, 0.55384959, 0.43555581],
            ]
        )
    )
    t = np.linspace(0, 1, 10)

    rotations = biomesh.utils.bislerp(random_rotations_1, random_rotations_2, t)

    expected_quats = np.array(
        [
            [0.2140843693, 0.8707791803, 0.1390449766, 0.4202118471],
            [0.6047190917, 0.5863829232, 0.4299440984, 0.3249891686],
            [0.6065957100, 0.3778037221, 0.6042147202, 0.3524635642],
            [0.5383747573, 0.6511053404, 0.4607679939, 0.2718589932],
            [0.7426270487, 0.1954047071, 0.2235768478, 0.6002794850],
            [0.0509440272, 0.8626629746, 0.0465043663, 0.5010535324],
            [0.2361934245, 0.6998940036, 0.3361674524, 0.5842537924],
            [0.6436627137, 0.5037631482, 0.5756160599, 0.0242312389],
            [0.4285218230, 0.1471075151, 0.3156109589, 0.8337374580],
            [0.4553291970, 0.1236462927, 0.6930572229, 0.5450308270],
        ]
    )

    np.testing.assert_allclose(np.abs(rotations.as_quat()), expected_quats)


def _fiber_angle_deg(Q_1: np.ndarray, Q_2: np.ndarray) -> np.ndarray:
    """Angle between the first axes of two frame sets, ignoring their signs."""
    cos = np.abs(np.einsum("ij,ij->i", Q_1[:, :, 0], Q_2[:, :, 0]))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def test_bislerp_returns_endpoints():
    """At t=0 the result is Q_A, at t=1 it is Q_B (as frames, up to axis
    signs)."""
    rotation_1 = sp.transform.Rotation.random(200, random_state=1)
    rotation_2 = sp.transform.Rotation.random(200, random_state=2)

    at_start = biomesh.utils.bislerp(rotation_1, rotation_2, np.zeros(200))  # t=0
    at_end = biomesh.utils.bislerp(rotation_1, rotation_2, np.ones(200))  # t=1

    np.testing.assert_allclose(
        _fiber_angle_deg(at_start.as_matrix(), rotation_1.as_matrix()), 0.0, atol=1e-3
    )
    np.testing.assert_allclose(
        _fiber_angle_deg(at_end.as_matrix(), rotation_2.as_matrix()), 0.0, atol=1e-3
    )
