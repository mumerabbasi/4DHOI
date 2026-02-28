import os
import os.path as osp
import sys
import numpy as np
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger
from PIL import Image
from pathlib import Path

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.common import (
    to_numpy,
    to_torch,
    get_device,
    get_rotation_matrix,
    apply_transform3d,
    apply_transforms3d,
    linear_weights,
)
from hms.config import PATH_PREFIX


class VideoCLIP(object):
    def __init__(self, model_name="PE-Core-G14-448"):
        # models available: ['PE-Core-G14-448', 'PE-Core-L14-336', 'PE-Core-B16-224']
        self.device = get_device()

        LIB_DIRS = [f"{PATH_PREFIX}/perception_models"]
        for LIB_DIR in LIB_DIRS:
            if LIB_DIR not in sys.path:
                sys.path.append(LIB_DIR)

        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        self.model = pe.CLIP.from_config(model_name, pretrained=True).to(self.device)  # Downloads from HF
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

    def preprocess_video(self, frames, transform=None, num_frames=None):
        """
        Uniformly samples a specified number of frames from a video and preprocesses them.

        Parameters:
        - video: str, path to the video file.
        - transform: torchvision.transforms, a transform function to preprocess frames.
        - num_frames: int, number of frames to sample. Defaults to None.

        Returns:
        - Video Tensor: a tensor of shape (num_frames, 3, H, W) where H and W are the height and width of the frames.
        """
        if isinstance(frames, str):
            # Causing SAM2 segmentation fault
            # https://github.com/facebookresearch/sam2/issues/298
            # import decord
            # vr = decord.VideoReader(frames)

            # torch.uint8 (num_frames, H, W, 3)
            frames = torchvision.io.read_video(frames, pts_unit="sec", output_format="THWC")[0]
        if torch.is_tensor(frames):
            frames = to_numpy(frames)
        assert isinstance(frames, np.ndarray), "Frames should be a numpy array."
        assert frames.ndim == 4
        if frames.shape[-1] != 3:
            assert frames.shape[1] == 3
            frames = np.transpose(frames, (0, 2, 3, 1))
        if frames.dtype != np.uint8:
            if np.amax(frames) < 1 + 1e-1:
                frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
            else:
                frames = np.clip(frames, 0, 255).astype(np.uint8)

        total_frames = frames.shape[0]
        if num_frames is None:
            frame_indices = list(range(total_frames))
        else:
            # Uniformly sample frame indices
            num_frames = min(num_frames, total_frames)
            frame_indices = [int(i * (total_frames / num_frames)) for i in range(num_frames)]
        frames = frames[frame_indices]

        # Preprocess frames
        preprocessed_frames = [transform(Image.fromarray(frame)) for frame in frames]
        return torch.stack(preprocessed_frames, dim=0)

    @torch.no_grad()
    def get_text_features(self, text, normalize=True):
        if isinstance(text, str):
            text = [text]
            is_batch = False
        else:
            is_batch = True
        assert isinstance(text, list)
        text = self.tokenizer(text).to(self.device)
        text_features = self.model.encode_text(text)  # [batch_size, 1280]
        if normalize:
            text_features = F.normalize(text_features, dim=-1)
        if not is_batch:
            text_features = torch.squeeze(text_features, dim=0)
        return text_features

    @torch.no_grad()
    def get_video_features(self, video, num_frames=None, normalize=True):
        if isinstance(video, np.ndarray) or torch.is_tensor(video):
            if video.ndim == 4:
                videos = [video]
                is_batch = False
            else:
                assert video.ndim == 5
                videos = [video[i] for i in range(video.shape[0])]
                is_batch = True
        elif isinstance(video, str):
            videos = [video]
            is_batch = False
        assert isinstance(videos, (list, tuple))

        video_features = list()
        for video in videos:
            video = self.preprocess_video(video, transform=self.preprocess, num_frames=num_frames)
            video = video.unsqueeze(0).to(self.device)
            vfeatures = self.model.encode_video(video)  # [1, 1280]
            if normalize:
                vfeatures = F.normalize(vfeatures, dim=-1)
            video_features.append(vfeatures)
        video_features = torch.cat(video_features, dim=0)
        if not is_batch:
            video_features = torch.squeeze(video_features, dim=0)
        return video_features

    @torch.no_grad()
    def get_similarities(self, x, y, normalized=True):
        if x.ndim == 1:
            x = x.unsqueeze(0)
            is_x_batch = False
        else:
            is_x_batch = True
        assert x.ndim == 2
        if y.ndim == 1:
            y = y.unsqueeze(0)
            is_y_batch = False
        else:
            is_y_batch = True
        assert y.ndim == 2
        if not normalized:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        sim = x @ y.T
        if not is_x_batch:
            sim = torch.squeeze(sim, dim=0)
        if not is_y_batch:
            sim = torch.squeeze(sim, dim=-1)
        return sim
