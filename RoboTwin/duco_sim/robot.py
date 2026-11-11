"""Recorded joint convention; source URDF and meshes remain read-only."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


class DucoKinematics:
    def __init__(self, urdf):
        self.urdf = Path(urdf).resolve(strict=True)
        self.tree = ET.parse(self.urdf)
        joints = self.tree.getroot().findall('joint')
        self.arm = [j for j in joints if j.get('type') == 'revolute']
        self.fingers = [j for j in joints if j.get('type') == 'prismatic']
        if len(self.arm) != 6 or len(self.fingers) != 2:
            raise ValueError('Expected six revolute joints and two prismatic fingers')
        self.names = [j.get('name') for j in self.arm + self.fingers]
        self.lower = np.array([float(j.find('limit').get('lower')) for j in self.arm + self.fingers])
        self.upper = np.array([float(j.find('limit').get('upper')) for j in self.arm + self.fingers])

    def flange(self, q):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        for joint, angle in zip(self.arm, np.asarray(q)[:6]):
            origin = joint.find('origin')
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            result = result @ transform
            result[:3, :3] = result[:3, :3] @ Rotation.from_rotvec(axis * angle).as_matrix()
        return result

    def qpos(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('Duco action must be finite [q1..q6, opening]')
        opening = np.clip(action[6], 0, 1)
        return np.r_[action[:6], self.lower[6:] + opening * (self.upper[6:] - self.lower[6:])]


def estimate_mount(fk, recording, profile):
    """ONE fixed provisional transform from demo 50, not per-demo retargeting.

    Translation aligns the pad center to the detected eraser upper half at
    grasp. This can absorb scene/FK error; it is NOT hardware calibration.
    """
    from duco_wipe_task import gripper_cycle
    from duco_wipe_scene import plane_z
    _, close, _ = gripper_cycle(recording)
    flange = fk.flange(recording.joints[close])
    eraser = profile['eraser']
    half = np.asarray(eraser['size_xyz_m']) / 2
    xy = np.asarray(eraser['xy_center'])
    target = np.r_[xy, plane_z(profile['table']['plane'], *xy) + half[2] + min(.010, half[2] * .6)]
    rotation = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    translation = flange[:3, :3].T @ (target - flange[:3, 3]) - rotation @ np.array([0., 0., .125])
    return dict(xyz_m=translation.tolist(), rpy_deg=[0., 0., 90.], verified=False,
                source='fixed estimate from demo 50 first grasp and 90deg jaw-axis mismatch',
                reference_demo=50, reference_frame=int(close), pad_offset_in_mount_m=[0., 0., .125],
                caveat='Not hardware calibration. Scene and URDF kinematic errors can be absorbed into this estimate.')


def mount_matrix(calibration):
    values = np.r_[calibration['xyz_m'], calibration['rpy_deg']]
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError('Invalid tool mounting xyz_m / rpy_deg')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', calibration['rpy_deg'], degrees=True).as_matrix()
    result[:3, 3] = calibration['xyz_m']
    return result


def corrected_urdf(fk, calibration, directory):
    """Move palm visual+collision AND finger joints; never edit source assets."""
    root = copy.deepcopy(fk.tree.getroot())
    root.set('name', 'duco_gcr3_robotwin')
    flange = root.find("link[@name='link_6']")
    tool = ET.SubElement(root, 'link', name='duco_tool_mount')
    for element in list(flange):
        mesh = element.find('geometry/mesh')
        if element.tag in ('visual', 'collision') and mesh is not None and 'gripper_palm' in mesh.get('filename', ''):
            flange.remove(element)
            tool.append(element)
    joint = ET.SubElement(root, 'joint', name='duco_tool_mount_fixed', type='fixed')
    ET.SubElement(joint, 'parent', link='link_6')
    ET.SubElement(joint, 'child', link='duco_tool_mount')
    radians = np.deg2rad(calibration['rpy_deg'])
    ET.SubElement(joint, 'origin', xyz=' '.join(map(str, calibration['xyz_m'])), rpy=' '.join(map(str, radians)))
    for finger in root.findall('joint'):
        if finger.get('type') == 'prismatic':
            finger.find('parent').set('link', 'duco_tool_mount')
    for mesh in root.findall('.//mesh'):
        filename = mesh.get('filename')
        if filename.startswith('package://'):
            from sapien.wrapper.urdf_loader import _try_very_hard_to_find_file
            filename = _try_very_hard_to_find_file(filename, fk.urdf.parent)
        else:
            filename = str((fk.urdf.parent / filename).resolve(strict=True))
        mesh.set('filename', filename)
    path = Path(directory) / 'duco_gcr3_robotwin.urdf'
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return path
