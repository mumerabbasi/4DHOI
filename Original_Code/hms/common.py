import json
import os
import os.path as osp
import re
import random
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time
import uuid
import roma
import scipy
import scipy.sparse
import trimesh
import open3d as o3d
import imageio
import hashlib
from datetime import datetime
from pathlib import Path
from scipy.spatial import cKDTree


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0


class KNNSearch(object):
    DTYPE = np.float32
    WORKERS = 4

    def __init__(self, data):
        # data: (N, C)
        self.data = np.asarray(data, dtype=self.DTYPE)
        self.kdtree = cKDTree(self.data)

    def query(self, kpts, k, return_dists=False):
        # kpts: (K, C)
        kpts = np.asarray(kpts, dtype=self.DTYPE)
        nndists, nnindices = self.kdtree.query(kpts, k=k, workers=self.WORKERS)
        if return_dists:
            return nnindices, nndists
        else:
            return nnindices  # (K, k): k=1 -> 1D nnindices

    def query_ball(self, kpt, radius):
        # kpt: (3, )
        kpt = np.asarray(kpt, dtype=self.DTYPE)
        assert kpt.ndim == 1
        nnindices = self.kdtree.query_ball_point(kpt, radius, workers=self.WORKERS)  # list
        return nnindices


class LearnableParams(nn.Module):
    def __init__(self, init_val=None, shape=None, dtype=torch.float32, func=None):
        super().__init__()
        self.func = func
        if init_val is not None:
            self.param = nn.Parameter(init_val).to(dtype=dtype)
        elif shape is not None:
            self.param = nn.Parameter(torch.randn(*shape)).to(dtype=dtype)
        else:
            raise RuntimeError("[!] init_val and shape cannot be both None!")

    def forward(self):
        if self.func is not None:
            return self.func(self.param)
        else:
            return self.param


def may_create_folder(folder_path):
    if not osp.exists(folder_path):
        oldmask = os.umask(000)
        os.makedirs(folder_path, mode=0o777)
        os.umask(oldmask)
        return True
    return False


def make_clean_folder(folder_path):
    success = may_create_folder(folder_path)
    if not success:
        shutil.rmtree(folder_path)
        may_create_folder(folder_path)


def parent_folder(file_path):
    return str(Path(file_path).parent)


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key) if len(c) > 0]
    return sorted(file_list_ordered, key=alphanum_key)


def list_files(folder_path, name_filter, alphanum_sort=False, full_path=False):
    file_list = [p.name for p in list(Path(folder_path).glob(name_filter)) if p.is_file()]
    if alphanum_sort:
        file_list = sorted_alphanum(file_list)
    else:
        file_list = sorted(file_list)
    if full_path:
        file_list = [osp.join(folder_path, fn) for fn in file_list]
    return file_list


def list_folders(folder_path, name_filter="*", alphanum_sort=False, full_path=False):
    folder_list = [p.name for p in list(Path(folder_path).glob(name_filter)) if p.is_dir()]
    if alphanum_sort:
        folder_list = sorted_alphanum(folder_list)
    else:
        folder_list = sorted(folder_list)
    if full_path:
        folder_list = [osp.join(folder_path, fn) for fn in folder_list]
    return folder_list


def read_lines(file_path):
    with open(file_path, "r") as fin:
        lines = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
    return lines


def read_strings(file_path):
    with open(file_path, "r") as fin:
        ret = fin.readlines()
    return "".join(ret)


def read_json(filepath):
    with open(filepath, "r") as fh:
        ret = json.load(fh)
    return ret


def write_json(filepath, data):
    assert isinstance(data, (dict, tuple, list))
    with open(filepath, "w") as fh:
        fh.write(json.dumps(data, indent=4))


def read_yaml(filepath):
    with open(filepath, "r") as fh:
        ret = yaml.safe_load(fh)
    return ret


def write_yaml(filepath, data, flow_style=False):
    assert isinstance(data, (dict, tuple, list))
    with open(filepath, "w") as fh:
        yaml.dump(data, fh, default_flow_style=flow_style)


def get_time(fmt="%y%m%d-%H%M%S"):
    return datetime.now().strftime(fmt)


def get_unique_id():
    return get_time() + "_" + str(uuid.uuid4())


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def valid_str(x):
    return x is not None and isinstance(x, str) and x != ""


def strip_str(x):
    assert isinstance(x, str)
    res = list()
    for i in x.strip():
        if i.isalnum():
            res.append(i)
        else:
            if len(res) != 0 and res[-1] != "_":
                res.append("_")
    return "".join(res)


def hash_str(x):
    assert isinstance(x, str)
    x = x.encode("utf-8")
    sha256_hash = hashlib.sha256()
    sha256_hash.update(x)
    return sha256_hash.hexdigest()


def clean_text(t):
    return re.sub("[^A-Za-z0-9]+", "_", t)


def join_texts(sep, texts):
    return sep.join([t.strip() for t in texts if len(t.strip()) > 0])


def cosine_weights(w_start, w_end, steps, output_type="list", device="cpu"):
    w_min = min(w_start, w_end)
    w_max = max(w_start, w_end)
    curr_steps = np.arange(steps)
    weights = w_min + 0.5 * (w_max - w_min) * (1 + np.cos(curr_steps.astype(np.float32) / steps * np.pi))
    weights = weights if w_start >= w_end else weights[::-1]
    if output_type == "pt":
        return torch.as_tensor(weights).float().to(device)
    elif output_type == "np":
        return weights.astype(np.float32)
    elif output_type == "list":
        return weights.tolist()
    else:
        raise RuntimeError(f"[!] {output_type} is not supported!")


def linear_weights(w_start, w_end, steps, output_type="list", device="cpu"):
    weights = np.linspace(w_start, w_end, steps + 1)
    weights = weights[:-1]
    if output_type == "pt":
        return torch.as_tensor(weights).float().to(device)
    elif output_type == "np":
        return weights.astype(np.float32)
    elif output_type == "list":
        return weights.tolist()
    else:
        raise RuntimeError(f"[!] {output_type} is not supported!")


def normalize(x, axis, eps=1e-6):
    if isinstance(x, np.ndarray):
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        norm = np.clip(norm, a_min=eps, a_max=None)
        return x / norm
    elif isinstance(x, torch.Tensor):
        return F.normalize(x, dim=axis, eps=eps)
    else:
        raise RuntimeError(f"[!] input type is not supported!")


def up_to_homogeneous(vectors):
    if vectors.shape[-1] == 4:
        return vectors
    return torch.cat([vectors, torch.ones_like(vectors[..., 0:1])], dim=-1)


def down_from_homogeneous(homogeneous_vectors):
    return homogeneous_vectors[..., :-1] / homogeneous_vectors[..., -1:]


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:] / sin_half_angles_over_angles


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def index_vertices_by_faces(vertices_features, faces):
    """Index vertex features to convert per vertex tensor to per vertex per face tensor."""
    assert vertices_features.ndim == 3, "vertices_features must have 3 dimensions of shape (batch_size, num_points, knum)"
    assert faces.ndim == 2, "faces must have 2 dimensions of shape (num_faces, num_vertices)"
    input = vertices_features.unsqueeze(2).expand(-1, -1, faces.shape[-1], -1)
    indices = faces[None, ..., None].expand(vertices_features.shape[0], -1, -1, vertices_features.shape[-1])
    return torch.gather(input=input, index=indices, dim=1)


def get_translate_matrix(*args):
    if len(args) == 1:
        if isinstance(args[0], float):
            tx, ty, tz = args[0], args[0], args[0]
        elif isinstance(args[0], (np.ndarray, list, tuple)):
            assert len(args[0]) == 3
            tx, ty, tz = args[0][0], args[0][1], args[0][2]
        else:
            raise RuntimeError("[!] Wrong input arguments!")
    elif len(args) == 3:
        assert isinstance(args[0], float)
        tx, ty, tz = args
    else:
        raise RuntimeError("[!] Wrong input arguments!")
    res = np.identity(4, dtype=np.float32)
    res[0, 3] = tx
    res[1, 3] = ty
    res[2, 3] = tz
    return res


def get_scale_matrix(*args):
    if len(args) == 1:
        if isinstance(args[0], float):
            sx, sy, sz = args[0], args[0], args[0]
        elif isinstance(args[0], (np.ndarray, list, tuple)):
            assert len(args[0]) == 3
            sx, sy, sz = args[0][0], args[0][1], args[0][2]
        else:
            raise RuntimeError("[!] Wrong input arguments!")
    elif len(args) == 3:
        assert isinstance(args[0], float)
        sx, sy, sz = args
    else:
        raise RuntimeError("[!] Wrong input arguments!")
    res = np.identity(4, dtype=np.float32)
    res[0, 0] = sx
    res[1, 1] = sy
    res[2, 2] = sz
    return res


def get_rotation_matrix(axis, theta):
    theta = theta * math.pi / 180.0
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    if axis == "x":
        rot = np.asarray(
            [
                [1, 0, 0],
                [0, costheta, -sintheta],
                [0, sintheta, costheta],
            ]
        )
    elif axis == "y":
        rot = np.asarray(
            [
                [costheta, 0, sintheta],
                [0, 1, 0],
                [-sintheta, 0, costheta],
            ]
        )
    elif axis == "z":
        rot = np.asarray(
            [
                [costheta, -sintheta, 0],
                [sintheta, costheta, 0],
                [0, 0, 1],
            ]
        )
    else:
        raise RuntimeError(f"[!] axis {axis} is not supported!")
    res = np.identity(4, dtype=np.float32)
    res[:3, :3] = rot
    return res


def apply_transform3d(x, xform):
    if torch.is_tensor(x):
        assert torch.is_tensor(xform)
        if x.numel() == 0:
            return x
        assert x.ndim == 2 and x.shape[-1] == 3
        assert xform.ndim == 2
        x = F.pad(x, (0, 1), "constant", 1.0)
        x = x @ xform.transpose(0, 1)
        x = x[..., :3].contiguous()
        return x
    elif isinstance(x, np.ndarray):
        assert isinstance(xform, np.ndarray)
        if x.size == 0:
            return x
        assert x.ndim == 2 and x.shape[-1] == 3
        assert xform.ndim == 2 and xform.shape == (4, 4)
        x = np.concatenate([x, np.ones_like(x[:, :1])], axis=1)
        x = x @ xform.T
        x = x[:, :3]
        return x
    else:
        raise RuntimeError("[!] x and xform must be either torch.Tensor or np.ndarray!")


def apply_transforms3d(x, xforms):
    if torch.is_tensor(x):
        assert torch.is_tensor(xforms)
        if x.numel() == 0:
            return x
        assert x.ndim == 3 and x.shape[-1] == 3
        if xforms.ndim == 2:
            xforms = xforms.unsqueeze(0)
        assert xforms.ndim == 3
        x = F.pad(x, (0, 1), "constant", 1.0)
        x = x @ xforms.transpose(1, 2)
        x = x[..., :3].contiguous()
        return x
    elif isinstance(x, np.ndarray):
        assert isinstance(xforms, np.ndarray)
        if x.size == 0:
            return x
        assert x.ndim == 3 and x.shape[-1] == 3
        if xforms.ndim == 2:
            xforms = np.expand_dims(xforms, axis=0)
        assert xforms.ndim == 3
        x = np.concatenate([x, np.ones_like(x[:, :, :1])], axis=2)
        x = x @ np.transpose(xforms, (0, 2, 1))
        x = x[..., :3]
        return x
    else:
        raise RuntimeError("[!] x and xforms must be either torch.Tensor or np.ndarray!")


def intrinsic_to_gl_projection(K, width, height, near=0.1, far=100):
    """
    Convert a 3x3 camera intrinsic matrix K to a 4x4 OpenGL perspective projection matrix.

    Parameters:
    - K: 3x3 numpy array, the intrinsic matrix with fx, fy, cx, cy.
    - width: int, image width.
    - height: int, image height.
    - near: float, near clipping plane distance.
    - far: float, far clipping plane distance.

    Returns:
    - 4x4 numpy array, OpenGL perspective projection matrix.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Convert cx, cy to normalized device coordinates (NDC, range [-1,1])
    left = -cx * near / fx
    right = (width - cx) * near / fx
    bottom = -(height - cy) * near / fy
    top = cy * near / fy
    # Construct OpenGL perspective projection matrix
    P = np.zeros((4, 4))
    P[0, 0] = 2 * near / (right - left)
    P[1, 1] = 2 * near / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = -(far + near) / (far - near)
    P[3, 2] = -1
    P[2, 3] = -2 * far * near / (far - near)
    return P


def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, (scipy.sparse.coo_matrix, scipy.sparse.csc_matrix)):
        return sparse_np_to_torch(x)
    elif isinstance(x, (list, tuple)):
        return [to_torch(t) for t in x]
    elif isinstance(x, dict):
        return {k: to_torch(v) for k, v in x.items()}
    elif isinstance(x, (int, float, bool)):
        return torch.as_tensor(x)
    else:
        return x


def to_cuda(x, device="cuda"):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        return [to_cuda(t) for t in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        return x


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return [to_numpy(t) for t in x]
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    elif isinstance(x, (int, float, bool)):
        return np.asarray(x)
    else:
        return x


def to_list(x):
    if len(x) == 0:
        return list()
    if isinstance(x[0], (int, float, bool)):
        return [item for item in x]
    else:
        return [to_list(item) for item in x]


def seeding(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def to_o3d_mesh(V, F, VC=None):
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.copy(V)),
        o3d.utility.Vector3iVector(np.copy(F)),
    )
    if VC is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(np.copy(VC))
    return m


def to_o3d_pcd(V, VN=None, VC=None):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.copy(V)))
    if VN is not None:
        p.normals = o3d.utility.Vector3dVector(np.copy(VN))
    if VC is not None:
        p.colors = o3d.utility.Vector3dVector(np.copy(VC))
    return p


def load_o3d_pcd(filepath):
    pcd = o3d.t.io.read_point_cloud(filepath)
    out = dict()
    for k, v in pcd.point.items():
        if k == "positions":
            V = v.numpy().astype(np.float32)
            out["V"] = V
        elif k == "colors":
            VC = v.numpy()
            if VC.dtype != np.uint8:
                VC = np.clip(VC, 0, 1) * 255
            out["VC"] = VC.astype(np.uint8)
        elif k == "normals":
            VN = v.numpy().astype(np.float32)
            out["VN"] = VN
    return out


def save_o3d_pcd(filepath, V, VN=None, VC=None):
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(V, dtype=dtype)
    if VN is not None:
        pcd.point.normals = o3d.core.Tensor(VN, dtype=dtype)
    if VC is not None:
        if VC.dtype != np.uint8:
            VC = np.clip(VC, 0, 1) * 255
        VC = VC.astype(np.uint8)
        pcd.point.colors = o3d.core.Tensor(VC, dtype=o3d.core.uint8)
    return o3d.t.io.write_point_cloud(filepath, pcd)


def save_o3d_mesh(filepath, V, F, VC=None):
    dtype = o3d.core.float32
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(V, dtype=dtype)
    mesh.triangle.indices = o3d.core.Tensor(F, dtype=o3d.core.int32)
    if VC is not None:
        if VC.dtype != np.uint8:
            VC = np.clip(VC, 0, 1) * 255
        VC = VC.astype(np.uint8)
        mesh.vertex.colors = o3d.core.Tensor(VC, dtype=o3d.core.uint8)
    return o3d.t.io.write_triangle_mesh(filepath, mesh)


def sample_pcd(vertices, faces, n_points=10000):
    mesh = to_o3d_mesh(vertices, faces)
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return np.asarray(pcd.points).astype(np.float32)


def subsample_pcd(points, voxel_size=0.005, return_indices=False):
    pcd = to_o3d_pcd(points)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # pcd_down.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.0)
    points_down = np.asarray(pcd_down.points).astype(np.float32)
    if return_indices:
        knnsearch = KNNSearch(points)
        indices = knnsearch.query(points_down, k=1)
        points_down = points[indices, :]
        return points_down, indices
    else:
        return points_down


def load_trimesh(filepath, force=None):
    return trimesh.load(filepath, force=force, process=False, validate=False, maintain_order=True)


def to_trimesh(V, F, VC=None, VN=None):
    return trimesh.Trimesh(
        vertices=V,
        faces=F,
        vertex_colors=VC,
        vertex_normals=VN,
        process=False,
        validate=False,
    )


def load_mtl(filepath, verbose=False):
    folder = parent_folder(filepath)
    lines = read_lines(filepath)
    materials = list()
    for line in lines:
        if line.startswith("#"):
            continue
        items = line.split()
        if items[0] == "newmtl":
            materials.append({"material_name": items[1]})
        elif items[0] in ("Ka", "Kd", "Ks", "Ke"):
            materials[-1][items[0]] = np.asarray(list(map(float, items[1:4])), dtype=np.float32)
        elif items[0] in ("Ns", "Ni", "d"):
            materials[-1][items[0]] = float(items[1])
        elif items[0] == "Tr":
            materials[-1]["d"] = 1 - float(items[1])
        elif items[0] == "illum":
            materials[-1][items[0]] = int(items[1])
        elif items[0] in (
            "map_Ka",
            "map_Kd",
            "map_Ks",
            "map_Ke",
            "map_Ns",
            "map_Bump",
            "bump",
        ):
            image = imageio.imread(osp.join(folder, items[-1]))  # (H, W, C), uint8
            # image = image.astype(np.float32) / 255.0
            materials[-1][items[0]] = image
        else:
            if verbose:
                print(f"[*] load_mtl: skipping {line}")
    return materials


def load_obj(filepath, verbose=False):
    lines = read_lines(filepath)

    vertices = list()
    vertex_normals = list()
    uvs = list()
    faces = list()
    face_uv_indices = list()
    face_normal_indices = list()
    material_assigns = list()
    materials = list()

    def get_material_id(name):
        for idx, mat in enumerate(materials):
            if mat["material_name"] == name:
                return idx
        return None

    for line in lines:
        if line.startswith("mtllib"):
            mats = load_mtl(osp.join(parent_folder(filepath), line.split()[-1]), verbose=verbose)
            materials += mats

    material_id = None
    face_count = 0
    for line in lines:
        if line.startswith("#"):
            continue
        items = line.split()
        if items[0] == "v":
            vertices.append(list(map(float, items[1:4])))
        elif items[0] == "vn":
            vertex_normals.append(list(map(float, items[1:4])))
        elif items[0] == "vt":
            uvs.append(list(map(float, items[1:3])))
        elif items[0] == "f":
            indices = [list(), list(), list()]
            for item in items[1:4]:
                for idx, val in enumerate(item.split("/")):
                    if len(val) == 0:
                        continue
                    indices[idx].append(int(val) - 1)
            faces.append(indices[0])
            face_uv_indices.append(indices[1])
            face_normal_indices.append(indices[2])
            face_count += 1
        elif items[0] == "usemtl":
            if face_count > 0:
                material_assigns.extend([material_id] * face_count)
            material_id = get_material_id(items[1])
            face_count = 0
        else:
            if verbose:
                print(f"[*] load_obj: skipping {line}")
    if face_count > 0:
        material_assigns.extend([material_id] * face_count)

    vertices = np.asarray(vertices, dtype=np.float32)
    vertex_normals = np.asarray(vertex_normals, dtype=np.float32)
    if vertex_normals.size > 0:
        vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    uvs = np.asarray(uvs, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    face_uv_indices = np.asarray(face_uv_indices, dtype=np.int32)
    face_normal_indices = np.asarray(face_normal_indices, dtype=np.int32)
    material_assigns = np.asarray(material_assigns, dtype=np.int32)

    return (
        vertices,
        vertex_normals,
        uvs,
        faces,
        face_uv_indices,
        face_normal_indices,
        material_assigns,
        materials,
    )


def load_smplx_uv(template_path, texture_path, verbose=False):
    lines = read_lines(template_path)
    uvs = list()
    faces = list()
    face_uv_indices = list()
    face_normal_indices = list()
    for line in lines:
        if line.startswith("#"):
            continue
        items = line.split()
        if items[0] == "vt":
            uvs.append(list(map(float, items[1:3])))
        elif items[0] == "f":
            indices = [list(), list(), list()]
            for item in items[1:4]:
                for idx, val in enumerate(item.split("/")):
                    if len(val) == 0:
                        continue
                    indices[idx].append(int(val) - 1)
            faces.append(indices[0])
            face_uv_indices.append(indices[1])
            face_normal_indices.append(indices[2])
        else:
            if verbose:
                print(f"[*] load_obj: skipping {line}")
    uvs = np.asarray(uvs, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    face_uv_indices = np.asarray(face_uv_indices, dtype=np.int32)
    face_normal_indices = np.asarray(face_normal_indices, dtype=np.int32)

    texture = imageio.imread(texture_path)[..., :3]  # (H, W, C), uint8

    return {
        "uvs": uvs,
        "faces": faces,
        "face_uv_indices": face_uv_indices,
        "face_normal_indices": face_normal_indices,
        "map_Kd": texture,
        "map_Ks": np.asarray((0.5, 0.5, 0.5)),
    }


def save_smplx_mesh(
    filepath,
    template_path,
    texture_path,
    vertices,
    Ns=160,
    Ka=1.0,
    Ks=0.5,
    Ke=0.0,
    Ni=1.45,
    d=1.0,
    illum=2,
):
    root_dir = str(Path(filepath).parent)
    filename = Path(filepath).stem
    # obj file
    lines = list()
    lines.append(f"mtllib {filename}.mtl")
    lines.append("o SMPLX-mesh")
    for i in range(len(vertices)):
        lines.append(f"v {vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f}")
    with open(template_path, "r") as fh:
        temp_lines = [line.strip() for line in fh.readlines() if len(line.strip()) > 0]
    with open(filepath, "w") as fh:
        for line in lines + temp_lines:
            fh.write(line + "\n")
    # mtl file
    lines = "newmtl material_0\n"
    lines += f"Ns {Ns}\n"
    lines += f"Ka {Ka} {Ka} {Ka}\n"
    lines += f"Ks {Ks} {Ks} {Ks}\n"
    lines += f"Ke {Ke} {Ke} {Ke}\n"
    lines += f"Ni {Ni} {Ni} {Ni}\n"
    lines += f"d {d}\n"
    lines += f"illum {illum}\n"
    if texture_path is not None and texture_path != "":
        lines += f"map_Kd {Path(texture_path).name}\n"
    with open(osp.join(root_dir, f"{filename}.mtl"), "w") as fh:
        fh.write(lines)
    # texture
    if texture_path is not None and texture_path != "":
        shutil.copy(texture_path, osp.join(root_dir, Path(texture_path).name))
