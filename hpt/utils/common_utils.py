import matplotlib as mpl

mpl.use("Agg")

from typing import Tuple
import os
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import tabulate

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0


def euler2mat_batch(ai, aj, ak, axes="sxyz"):
    """Return rotation matrix from Euler angles and axis sequence.

    Parameters
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    batches = np.asarray(ai).shape
    M = np.zeros((*batches, 3, 3))
    if repetition:
        M[..., i, i] = cj
        M[..., i, j] = sj * si
        M[..., i, k] = sj * ci
        M[..., j, i] = sj * sk
        M[..., j, j] = -cj * ss + cc
        M[..., j, k] = -cj * cs - sc
        M[..., k, i] = -sj * ck
        M[..., k, j] = cj * sc + cs
        M[..., k, k] = cj * cc - ss
    else:
        M[..., i, i] = cj * ck
        M[..., i, j] = sj * sc - cs
        M[..., i, k] = sj * cc + ss
        M[..., j, i] = cj * sk
        M[..., j, j] = sj * ss + cc
        M[..., j, k] = sj * cs - sc
        M[..., k, i] = -sj
        M[..., k, j] = cj * si
        M[..., k, k] = cj * ci
    return M


def mat2euler_batch(mat, axes="sxyz"):
    """Return Euler angles from rotation matrix for specified axis sequence."""
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(mat, dtype=np.float64, copy=False)[..., :3, :3]
    if repetition:
        sy = np.sqrt(M[..., i, j] * M[..., i, j] + M[..., i, k] * M[..., i, k])
        if_sy = sy > _EPS4
        ax = np.where(if_sy, np.arctan2(M[..., i, j], M[..., i, k]), np.arctan2(-M[..., j, k], M[..., j, j]))
        ay = np.where(if_sy, np.arctan2(sy, M[..., i, i]), np.arctan2(sy, M[..., i, i]))
        az = np.where(if_sy, np.arctan2(M[..., j, i], -M[..., k, i]), 0.0)
    else:
        cy = np.sqrt(M[..., i, i] * M[..., i, i] + M[..., j, i] * M[..., j, i])
        if_cy = cy > _EPS4
        ax = np.where(if_cy, np.arctan2(M[..., k, j], M[..., k, k]), np.arctan2(-M[..., j, k], M[..., j, j]))
        ay = np.where(if_cy, np.arctan2(-M[..., k, i], cy), np.arctan2(-M[..., k, i], cy))
        az = np.where(if_cy, np.arctan2(M[..., j, i], M[..., i, i]), 0.0)

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax

    return ax, ay, az


def euler2axangle_batch(eulers):
    from transforms3d.euler import euler2axangle

    torch_input = type(eulers) is not np.ndarray
    if torch_input:
        eulers_torch = eulers
        eulers = eulers.detach().cpu().numpy()

    shape = eulers.shape
    eulers = eulers.reshape(-1, 3)

    axangle = []
    for euler in eulers:
        axis, angle = euler2axangle(euler[0], euler[1], euler[2])
        axangle.append(axis * angle)

    axangle = np.array(axangle)
    if torch_input:
        return torch.from_numpy(axangle).to(eulers_torch).reshape(shape)
    return axangle.reshape(shape)


def axangle2euler_batch(axangles):
    from transforms3d.euler import axangle2euler

    torch_input = type(axangles) is not np.ndarray
    if torch_input:
        axangles_torch = axangles
        axangles = axangles.detach().cpu().numpy()

    shape = axangles.shape
    axangles = axangles.reshape(-1, 3)
    eulers = []
    for axanlge in axangles:
        angle = np.linalg.norm(axanlge)
        axis = axanlge / angle

        euler = axangle2euler(axis, angle)
        eulers.append(np.array(euler))

    eulers = np.array(eulers)
    if torch_input:
        return torch.from_numpy(eulers).to(axangles_torch).reshape(shape)
    return eulers.reshape(shape)


def euler2rotmat_batch(eulers):
    from transforms3d.euler import euler2mat

    torch_input = type(eulers) is not np.ndarray
    if torch_input:
        eulers_torch = eulers
        eulers = eulers.detach().cpu().numpy()

    shape = list(eulers.shape)
    shape[-1] = 6  # more dimensions
    eulers = eulers.reshape(-1, 3)

    rotmats = euler2mat_batch(eulers[:, 0], eulers[:, 1], eulers[:, 2])[:, :2]

    # rotmats = []
    # for euler in eulers:
    #     mat = euler2mat(euler[0], euler[1], euler[2])
    #     rotmats.append(mat[:2])

    # rotmats = np.array(rotmats)
    if torch_input:
        return torch.from_numpy(rotmats).to(eulers_torch).reshape(shape)
    return rotmats.reshape(shape)


def rotmat2euler_batch(rotmats):
    from transforms3d.euler import mat2euler

    torch_input = type(rotmats) is not np.ndarray
    if torch_input:
        rotmats_torch = rotmats
        rotmats = rotmats.detach().cpu().numpy()

    shape = list(rotmats.shape)
    shape[-1] = 3
    rotmats = rotmats.reshape(-1, 2, 3)
    eulers = []
    for rotmat in rotmats:
        # normalize first two rows and then expand to full matrix
        rotmat[0] /= np.linalg.norm(rotmat[0] + 1e-8)
        rotmat[1] /= np.linalg.norm(rotmat[1] + 1e-8)
        # rot_lastrow = np.array([rotmat[0, 2], rotmat[1, 2], np.sqrt(1 - rotmat[0, 2] ** 2 - rotmat[1, 2] ** 2)])
        rot_lastrow = np.cross(rotmat[0], rotmat[1])
        rotmat = np.concatenate((rotmat, rot_lastrow[None]), axis=0)
        euler = mat2euler(rotmat)
        eulers.append(np.array(euler))

    eulers = np.array(eulers)
    if torch_input:
        return torch.from_numpy(eulers).to(rotmats_torch).reshape(shape)
    return eulers.reshape(shape)


def expand_dict_dim(d, sample_num=1, dim=0):
    """
    Recursively expand a dictionary of PyTorch tensors along one of their dimensions.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = expand_dict_dim(v, sample_num=sample_num, dim=dim)
        elif isinstance(v, torch.Tensor):
            d[k] = torch.cat([v] * sample_num, dim=dim)
    return d


def print_and_write(file_handle, text):
    print(text)

    if file_handle is not None:
        if type(file_handle) is list:
            for f in file_handle:
                f.write(text + "\n")
        else:
            file_handle.write(text + "\n")
    return text


def tabulate_print_state_and_write(state_dict, extra_dict, output_dir, train_epochs):
    dt_string = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    file_name = "output_stat.txt"
    output_stat_file = os.path.join(f"outputs/{file_name}")
    file_handle = open(output_stat_file, "a+")

    # also write in local folder
    local_file_handle = open(os.path.join(f"{output_dir}/{file_name}.txt"), "a+")
    output_text = ""
    output_text += print_and_write([file_handle, local_file_handle], "\n")
    output_text += print_and_write(
        [file_handle, local_file_handle],
        "------------------------------------------------------------------",
    )

    for k, v in extra_dict.items():
        output_text += print_and_write([file_handle, local_file_handle], f"{k}: {v}")

    output_text += print_and_write(
        [file_handle, local_file_handle],
        "Test Time: {} Data Root: {} Output Dir: {} Epochs: {}".format(
            dt_string,
            extra_dict["logdir"],
            output_dir,
            train_epochs,
        ),
    )

    text = tabulate_print_state(state_dict)
    if file_handle is not None:
        file_handle.write(text + "\n")
    output_text += print_and_write(
        file_handle,
        "------------------------------------------------------------------",
    )
    if local_file_handle is not None:
        local_file_handle.write(text + "\n")
    output_text += print_and_write(
        local_file_handle,
        "------------------------------------------------------------------",
    )

    return text


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def tabulate_print_state(result_dict):
    """print state dict"""
    result_dict = sorted(result_dict.items())
    headers = ["task", "reward"]
    data = [[kv[0], kv[1]] for kv in result_dict]

    # np.mean([kv[1] for kv in result_dict])]
    data.append(["avg", [np.mean([kv[1][i] for kv in result_dict]) for i in range(3)]])
    str = tabulate.tabulate(data, headers, tablefmt="psql", floatfmt=".2f")
    print(str)
    return str


def to_rotation_matrix(angles):
    az, el, th = angles[..., 0], angles[..., 1], angles[..., 2]
    batch_dims = list(az.shape)

    cx = torch.cos(az)
    cy = torch.cos(el)
    cz = torch.cos(th)
    sx = torch.sin(az)
    sy = torch.sin(el)
    sz = torch.sin(th)

    ones = torch.ones_like(cx)
    zeros = torch.zeros_like(cx)

    rx = torch.stack([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx], dim=-1)
    ry = torch.stack([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy], dim=-1)
    rz = torch.stack([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones], dim=-1)

    rot_shape = batch_dims + [3, 3]
    rx = torch.reshape(rx, rot_shape)
    ry = torch.reshape(ry, rot_shape)
    rz = torch.reshape(rz, rot_shape)

    return torch.matmul(rz, torch.matmul(ry, rx))


def se3_inverse_numpy_batch(RT):
    R = RT[:, :3, :3]
    T = RT[:, :3, 3].reshape((-1, 3, 1))
    RT_new = np.tile(np.eye(4), (len(R), 1, 1))
    RT_new[:, :3, :3] = R.transpose(0, 2, 1)
    RT_new[:, :3, 3] = -1 * np.matmul(R.transpose(0, 2, 1), T).reshape(-1, 3)
    return RT_new


def se3_inverse_tensor(RT):
    R = RT[:, :3, :3]
    T = RT[:, :3, 3].view((-1, 3, 1))
    RT_new = torch.eye(4).repeat((len(R), 1, 1))
    RT_new[:, :3, :3] = R.permute(0, 2, 1)
    RT_new[:, :3, 3] = -1 * torch.matmul(R.permute(0, 2, 1), T).view(-1, 3)
    return RT_new


def unpack_action_numpy(xyzrpy):
    """
    Create 4x4 homogeneous transform matrix from pos and rpy
    """
    xyz = xyzrpy[0:3]
    rpy = xyzrpy[3:6]

    T = np.zeros([4, 4], dtype=xyzrpy.dtype)
    T[0:3, 0:3] = rpy_to_rotation_matrix(rpy)
    T[3, 3] = 1
    T[0:3, 3] = xyz
    return T


def rpy_to_rotation_matrix(rpy):
    """
    Creates 3x3 rotation matrix from rpy
    See http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
    """
    u = rpy[0]
    v = rpy[1]
    w = rpy[2]

    R = np.zeros([3, 3], dtype=rpy.dtype)

    # first row
    R[0, 0] = np.cos(v) * np.cos(w)
    R[0, 1] = np.sin(u) * np.sin(v) * np.cos(w) - np.cos(u) * np.sin(w)
    R[0, 2] = np.sin(u) * np.sin(w) + np.cos(u) * np.sin(v) * np.cos(w)

    # second row
    R[1, 0] = np.cos(v) * np.sin(w)
    R[1, 1] = np.cos(u) * np.cos(w) + np.sin(u) * np.sin(v) * np.sin(w)
    R[1, 2] = np.cos(u) * np.sin(v) * np.sin(w) - np.sin(u) * np.cos(w)

    # third row
    R[2, 0] = -np.sin(v)
    R[2, 1] = np.sin(u) * np.cos(v)
    R[2, 2] = np.cos(u) * np.cos(v)

    return R


def mat2euler_batch(rot_mat):
    """
    convert rotation matrix to euler angle
    :param rot_mat: rotation matrix rx*ry*rz [B, 3, 3]
    :param seq: seq is xyz(rotate along z first) or zyx
    :return: three angles, x, y, z
    """
    r11 = rot_mat[:, 0, 0]
    r12 = rot_mat[:, 0, 1]
    r13 = rot_mat[:, 0, 2]
    r21 = rot_mat[:, 1, 0]
    r22 = rot_mat[:, 1, 1]
    r23 = rot_mat[:, 1, 2]
    r31 = rot_mat[:, 2, 0]
    r32 = rot_mat[:, 2, 1]
    r33 = rot_mat[:, 2, 2]

    yaw = np.arctan2(r21, r11)
    pitch = -np.arcsin(r31)
    roll = np.arctan2(r32, r33)
    return np.stack((roll, pitch, yaw), axis=1)


def pack_action_batch(pose_mat):
    pose = np.zeros((len(pose_mat), 6))
    pose[:, :3] = pose_mat[:, :3, 3]
    pose[:, 3:6] = mat2euler_batch(pose_mat[:, :3, :3])
    return pose


def unpack_action_numpy_batch(xyzrpy):
    """
    Create 4x4 homogeneous transform matrix from pos and rpy
    """
    xyz = xyzrpy[..., 0:3]
    rpy = xyzrpy[..., 3:6]

    T = np.zeros([len(xyzrpy), 4, 4])
    T[..., 0:3, 0:3] = rpy_to_rotation_matrix_batch(rpy)
    T[..., 3, 3] = 1
    T[..., 0:3, 3] = xyz
    return T


def rpy_to_rotation_matrix_batch(rpy):
    """
    Creates 3x3 rotation matrix from rpy
    See http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
    """
    u = rpy[..., 0]
    v = rpy[..., 1]
    w = rpy[..., 2]

    R = np.zeros([len(u), 3, 3], dtype=rpy.dtype)

    # first row
    R[..., 0, 0] = np.cos(v) * np.cos(w)
    R[..., 0, 1] = np.sin(u) * np.sin(v) * np.cos(w) - np.cos(u) * np.sin(w)
    R[..., 0, 2] = np.sin(u) * np.sin(w) + np.cos(u) * np.sin(v) * np.cos(w)

    # second row
    R[..., 1, 0] = np.cos(v) * np.sin(w)
    R[..., 1, 1] = np.cos(u) * np.cos(w) + np.sin(u) * np.sin(v) * np.sin(w)
    R[..., 1, 2] = np.cos(u) * np.sin(v) * np.sin(w) - np.sin(u) * np.cos(w)

    # third row
    R[..., 2, 0] = -np.sin(v)
    R[..., 2, 1] = np.sin(u) * np.cos(v)
    R[..., 2, 2] = np.cos(u) * np.cos(v)

    return R


class PoseBCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        control_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.053, -0.0, 0.075],
                [-0.053, 0.0, 0.075],
                [0.053, -0.0, 0.105],
                [-0.053, 0.0, 0.105],
            ],
            dtype=np.float32,
        )
        self.register_buffer("control_points", torch.from_numpy(control_points))

    def cp_from_action(self, action):
        # action: (N, 6)
        action = action.contiguous().view(-1, 6)
        eulers, trans = action[:, 3:], action[:, :3]
        rot = to_rotation_matrix(eulers)
        grasp_pc = torch.matmul(self.control_points[None], rot.permute(0, 2, 1))
        grasp_pc += trans.unsqueeze(1).expand(-1, grasp_pc.shape[1], -1)
        return grasp_pc

    def forward(self, x, y):
        cp_x = self.cp_from_action(x.contiguous().view(-1, 6))
        cp_y = self.cp_from_action(y.contiguous().view(-1, 6))
        return torch.abs(cp_x - cp_y).mean()
