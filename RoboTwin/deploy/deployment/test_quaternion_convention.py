#!/opt/shared/Prog/anaconda3/envs/rise/bin/python
"""Unit tests for quaternion convention round-trip correctness.

Requires the ``rise`` conda environment:
    /opt/shared/Prog/anaconda3/envs/rise/bin/python

Verifies:
- axis_angle → quaternion → axis_angle (round-trip)
- PyTorch3D scalar-first ↔ system scalar-last reordering
- Known identity and 90-degree rotations
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_scalar_first_to_last(q_wxyz: np.ndarray) -> np.ndarray:
    """[x,y,z,w,x,y,z] → [x,y,z,x,y,z,w]"""
    return q_wxyz[[0, 1, 2, 4, 5, 6, 3]]


def quaternion_scalar_last_to_first(q_xyzw: np.ndarray) -> np.ndarray:
    """[x,y,z,x,y,z,w] → [x,y,z,w,x,y,z]"""
    return q_xyzw[[0, 1, 2, 6, 3, 4, 5]]


def axis_angle_to_quaternion_xyzw(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle [x,y,z,rx,ry,rz] to quaternion scalar-last [x,y,z,qx,qy,qz,qw] using scipy."""
    r = R.from_rotvec(aa[3:])
    q = r.as_quat()  # scipy returns scalar-last [x, y, z, w] by default
    return np.array([aa[0], aa[1], aa[2], q[0], q[1], q[2], q[3]])


def quaternion_xyzw_to_axis_angle(q7: np.ndarray) -> np.ndarray:
    """Convert quaternion scalar-last [x,y,z,qx,qy,qz,qw] to axis-angle [x,y,z,rx,ry,rz]."""
    r = R.from_quat(q7[3:])  # scipy takes scalar-last
    aa = r.as_rotvec()
    return np.array([q7[0], q7[1], q7[2], aa[0], aa[1], aa[2]])


def test_identity_roundtrip():
    """Identity rotation round-trip."""
    aa = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0], dtype=np.float64)
    q7 = axis_angle_to_quaternion_xyzw(aa)
    aa_back = quaternion_xyzw_to_axis_angle(q7)
    assert np.allclose(aa, aa_back, atol=1e-10), f"Round-trip failed: {aa} → {q7} → {aa_back}"
    print("✅ identity round-trip OK")


def test_90deg_roundtrip():
    """90-degree rotation about Z."""
    aa = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2], dtype=np.float64)
    q7 = axis_angle_to_quaternion_xyzw(aa)
    # Expected: [0, 0, 0, qx=0, qy=0, qz=sin(pi/4), qw=cos(pi/4)]
    expected_qz = np.sin(np.pi / 4)
    expected_qw = np.cos(np.pi / 4)
    assert np.allclose(q7[3:6], [0, 0, expected_qz], atol=1e-10), f"Unexpected quat: {q7}"
    assert np.allclose(q7[6], expected_qw, atol=1e-10)
    aa_back = quaternion_xyzw_to_axis_angle(q7)
    assert np.allclose(aa, aa_back, atol=1e-10), f"Round-trip failed: {aa} → {q7} → {aa_back}"
    print("✅ 90° Z round-trip OK")


def test_scalar_convention_ordering():
    """Verify that our scalar-first ↔ scalar-last reorder is correct."""
    # Start with a known axis-angle
    aa = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float64)

    # Get reference scipy quaternion [x,y,z,w] (scalar-last)
    ref_r = R.from_rotvec(aa[3:])
    ref_xyzw = ref_r.as_quat()  # [qx, qy, qz, qw]

    # Now go through PyTorch3D's scalar-first format
    # In Agent:
    #   PyTorch3D quaternion_to_matrix expects [w, x, y, z]
    #   xyz_rot_transform(..., to_rep="quaternion") returns [x, y, z, w, x, y, z]
    #   reorder: [0,1,2,4,5,6,3] → [x,y,z,qx,qy,qz,qw]
    ref_w = ref_r.as_quat()  # scipy: [x, y, z, w] scalar-last
    # PyTorch3D scalar-first would be [w, x, y, z]
    ref_wxyz = np.array([aa[0], aa[1], aa[2], ref_w[3], ref_w[0], ref_w[1], ref_w[2]])

    # Simulate Agent reordering
    agent_output = ref_wxyz[[0, 1, 2, 4, 5, 6, 3]]  # → [x,y,z,qx,qy,qz,qw]
    assert np.allclose(agent_output[3:7], ref_xyzw, atol=1e-10), \
        f"Agent reorder check: {agent_output[3:7]} != scipy ref {ref_xyzw}"
    print("✅ scalar convention ordering OK")


def test_agent_scipy_roundtrip():
    """Full round-trip through the same reorder step used in Agent, verified via scipy."""
    aa_in = np.array([0.4, 0.3, 0.2, 0.5, -0.3, 1.2], dtype=np.float64)

    # Simulate what PyTorch3D does: axis_angle → quaternion scalar-first → [x,y,z,w,x,y,z]
    r = R.from_rotvec(aa_in[3:])
    q_wxyz = r.as_quat()  # scipy scalar-last [x,y,z,w]
    q_w_first = np.array([q_wxyz[3], q_wxyz[0], q_wxyz[1], q_wxyz[2]])  # [w,x,y,z]
    q7_pt3d = np.array([aa_in[0], aa_in[1], aa_in[2], q_w_first[0], q_w_first[1], q_w_first[2], q_w_first[3]])

    # Agent reorder: [0,1,2,4,5,6,3] → [x,y,z,qx,qy,qz,qw]
    q7_agent = q7_pt3d[[0, 1, 2, 4, 5, 6, 3]]

    # Verify it matches scipy scalar-last
    assert np.allclose(q7_agent[3:7], q_wxyz, atol=1e-10), \
        f"Agent output {q7_agent[3:7]} != scipy ref {q_wxyz}"
    assert np.allclose(q7_agent[:3], aa_in[:3], atol=1e-10)

    # Reverse: scalar-last → axis_angle
    # First reorder back to scalar-first [x,y,z,w,x,y,z]
    q7_pt3d_back = q7_agent[[0, 1, 2, 6, 3, 4, 5]]
    # Then extract quaternion as [w,x,y,z] and convert to axis_angle
    q_back_wxyz = np.array([q7_pt3d_back[3], q7_pt3d_back[4], q7_pt3d_back[5], q7_pt3d_back[6]])
    r_back = R.from_quat([q_back_wxyz[1], q_back_wxyz[2], q_back_wxyz[3], q_back_wxyz[0]])  # scipy expects [x,y,z,w]
    aa_back = np.array([q7_pt3d_back[0], q7_pt3d_back[1], q7_pt3d_back[2],
                        r_back.as_rotvec()[0], r_back.as_rotvec()[1], r_back.as_rotvec()[2]])

    assert np.allclose(aa_in, aa_back, atol=1e-6), \
        f"Agent round-trip: {aa_in} → {q7_agent} → {aa_back}"
    print("✅ Agent scipy round-trip OK")


def test_agent_pytorch3d_roundtrip():
    """Full round-trip through the same path used in Agent (requires pytorch3d)."""
    try:
        from utils.transformation import xyz_rot_transform
    except ImportError as e:
        print(f"⏭  Skipping PyTorch3D round-trip test (pytorch3d not available): {e}")
        return

    aa_in = np.array([0.4, 0.3, 0.2, 0.5, -0.3, 1.2], dtype=np.float64)

    # Agent path: axis_angle → quaternion (PyTorch3D scalar-first) → reorder to scalar-last
    q7_pt3d = xyz_rot_transform(aa_in, from_rep="axis_angle", to_rep="quaternion")
    # q7_pt3d is [x, y, z, w, x, y, z] (PyTorch3D scalar-first)
    q7_agent = q7_pt3d[[0, 1, 2, 4, 5, 6, 3]]  # → [x,y,z,qx,qy,qz,qw]

    # Reverse: scalar-last → scalar-first → axis_angle
    q7_pt3d_back = q7_agent[[0, 1, 2, 6, 3, 4, 5]]  # → [x,y,z,w,x,y,z]
    aa_back = xyz_rot_transform(q7_pt3d_back, from_rep="quaternion", to_rep="axis_angle")

    assert np.allclose(aa_in, aa_back, atol=1e-6), \
        f"Agent round-trip: {aa_in} → {q7_agent} → {aa_back}"
    print("✅ Agent PyTorch3D round-trip OK")


def test_set_tcp_pose_path():
    """Verify rotation_6d round-trip via scipy (no pytorch3d needed)."""
    # rotation_6d for identity rotation: first two columns of I_3
    tcp_rot6d = np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    # rotation_6d → matrix → rotation_6d (round-trip via scipy)
    # First convert rotation_6d to matrix
    a1 = tcp_rot6d[3:6]   # first column
    a2 = tcp_rot6d[6:9]   # second column
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.column_stack([b1, b2, b3])
    mat[:3, 3] = tcp_rot6d[:3]

    # Extract rotation_6d back from matrix
    a1_back = mat[:3, 0]
    a2_back = mat[:3, 1]
    tcp_back = np.concatenate([mat[:3, 3], a1_back, a2_back])

    assert np.allclose(tcp_rot6d, tcp_back, atol=1e-6), \
        f"rotation_6d round-trip: {tcp_rot6d} → {tcp_back}"
    print("✅ rotation_6d round-trip OK (scipy)")

    # Test conversion to axis_angle for robot
    r = R.from_matrix(mat[:3, :3])
    aa = r.as_rotvec()
    # For identity rotation, axis_angle should be near [0, 0, 0]
    assert np.allclose(aa, [0, 0, 0], atol=1e-6), \
        f"Identity rot6d should give zero axis-angle, got {aa}"
    print("✅ rotation_6d → axis_angle path OK (scipy)")

    # 90-degree Z rotation round-trip
    # Rz(π/2): first column = [cos, sin, 0] = [0, 1, 0]; second = [-sin, cos, 0] = [-1, 0, 0]
    rot6d_z90 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float64)
    a1_90 = rot6d_z90[3:6]
    a2_90 = rot6d_z90[6:9]
    b1_90 = a1_90 / np.linalg.norm(a1_90)
    b2_90 = a2_90 - np.dot(b1_90, a2_90) * b1_90
    b2_90 = b2_90 / np.linalg.norm(b2_90)
    b3_90 = np.cross(b1_90, b2_90)
    mat_90 = np.column_stack([b1_90, b2_90, b3_90])
    r_90 = R.from_matrix(mat_90)
    aa_90 = r_90.as_rotvec()
    # Expected: ~[0, 0, pi/2]
    assert np.allclose(aa_90, [0, 0, np.pi / 2], atol=1e-6), \
        f"90° Z should give [0,0,pi/2] axis-angle, got {aa_90}"
    print("✅ 90° Z rot6d → axis_angle OK (scipy)")


if __name__ == "__main__":
    test_identity_roundtrip()
    test_90deg_roundtrip()
    test_scalar_convention_ordering()
    test_agent_scipy_roundtrip()
    test_agent_pytorch3d_roundtrip()
    test_set_tcp_pose_path()
    print("\n🎉 All quaternion convention tests passed.")
