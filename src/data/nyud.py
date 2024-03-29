# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/astmt.
#
import logging
import os
import requests
import tarfile
import numpy as np
from PIL import Image
import torch


# to prevent pollution of debug log
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)


def download_file_from_google_drive(id, destination):
    """
    ---------------------------------------------------------------------------
    Copyright (c) Simon Vandenhende. All rights reserved.
    
    This source code is licensed under the license found in the
    LICENSE file in https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch.
    ---------------------------------------------------------------------------
    """
    URL = "https://drive.google.com/u/1/uc?export=download"
    CHUNK_SIZE = 32768

    session = requests.Session()

    params = {'id': id, 'confirm': 1}
    response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


class NYUD(torch.utils.data.Dataset):
    """
    NYUD dataset for multi-task learning.
    Includes edge detection, semantic segmentation, surface normals, and depth prediction
    """

    GOOGLE_DRIVE_ID = '14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw'

    semseg_num_classes = 40
    edge_pos_weight = 0.8
    edge_tolerance = 0.011

    image_dims = (3, 425, 560)

    def __init__(self,
                 data_dir,
                 split='train',
                 tasks=('semseg',),
                 transforms=None,
                 retname=False,
                 download=True):

        if download:
            self._download(data_dir)

        self.root = os.path.join(data_dir, 'NYUD_MT')

        self.transforms = transforms
        self.retname = retname

        centroids_path = os.path.join(os.path.dirname(__file__), 'db_info', 'nyu_centroids.npy')
        if os.path.exists(centroids_path):
            self.normals_centroids = torch.from_numpy(np.load(centroids_path).astype(np.float32))
        else:
            self.normals_centroids = None

        # Original Images
        self.im_ids = []
        self.images = []
        _image_dir = os.path.join(self.root, 'images')

        # Edge Detection
        self.do_edge = ('edge' in tasks)
        self.edges = []
        _edge_gt_dir = os.path.join(self.root, 'edge')

        # Semantic segmentation
        self.do_semseg = True
        # self.do_semseg = ('semseg' in tasks)
        self.semsegs = []
        _semseg_gt_dir = os.path.join(self.root, 'segmentation')

        # Surface Normals
        self.do_normals = ('normals' in tasks)
        self.normals = []
        _normal_gt_dir = os.path.join(self.root, 'normals')

        # Depth
        self.do_depth = ('depth' in tasks)
        self.depths = []
        _depth_gt_dir = os.path.join(self.root, 'depth')

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(self.root, 'gt_sets')

        with open(os.path.join(_splits_dir, split + '.txt'), 'r') as f:
            lines = f.read().splitlines()

        for line in lines:

            # Images
            _image = os.path.join(_image_dir, line + '.jpg')
            assert os.path.isfile(_image)
            self.images.append(_image)
            self.im_ids.append(line.rstrip('\n'))

            # Edges
            _edge = os.path.join(_edge_gt_dir, line + '.npy')
            assert os.path.isfile(_edge)
            self.edges.append(_edge)

            # Semantic Segmentation
            _semseg = os.path.join(_semseg_gt_dir, line + '.png')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

            _normal = os.path.join(_normal_gt_dir, line + '.npy')
            assert os.path.isfile(_normal)
            self.normals.append(_normal)

            _depth = os.path.join(_depth_gt_dir, line + '.npy')
            assert os.path.isfile(_depth)
            self.depths.append(_depth)

        if self.do_edge:
            assert len(self.images) == len(self.edges)
        if self.do_semseg:
            assert len(self.images) == len(self.semsegs)
        if self.do_normals:
            assert len(self.images) == len(self.normals)
        if self.do_depth:
            assert len(self.images) == len(self.depths)

    def __getitem__(self, index):
        sample = {}

        _im = self._load_img(index)
        sample['image'] = _im

        if self.do_edge:
            _edge = self._load_edge(index)
            sample['edge'] = _edge

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            sample['semseg'] = _semseg

        if self.do_normals:
            _normals = self._load_normals(index)
            sample['normals'] = _normals

        if self.do_depth:
            _depth = self._load_depth(index)
            sample['depth'] = _depth

        if self.retname:
            sample['meta'] = {'image': str(self.im_ids[index]),
                              'im_size': (sample['image'].shape[0], sample['image'].shape[1])}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _img = np.array(_img, dtype=np.float32, copy=False)
        return _img

    def _load_edge(self, index):
        _edge = np.load(self.edges[index])
        _edge = np.expand_dims(_edge.astype(np.float32), axis=2)
        return _edge

    def _load_semseg(self, index):
        # Note: We ignore the background class (40-way classification), as in related work:
        _semseg = Image.open(self.semsegs[index])
        _semseg = np.expand_dims(np.array(_semseg, dtype=np.float32, copy=False), axis=2) - 1
        _semseg[_semseg == -1] = 255
        return _semseg

    def _load_normals(self, index):
        _normals = np.load(self.normals[index])
        return _normals

    def _load_depth(self, index):
        _depth = np.load(self.depths[index])
        _depth = np.expand_dims(_depth.astype(np.float32), axis=2)
        return _depth
    
    def _download(self, data_dir):

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        _fpath = os.path.join(data_dir, 'NYUD_MT.tgz')

        if os.path.isfile(_fpath):
            return
        else:
            print('Downloading from google drive')
            os.makedirs(os.path.dirname(_fpath), exist_ok=True)
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(data_dir)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def __repr__(self):
        return self.__class__.__name__ + '()'
