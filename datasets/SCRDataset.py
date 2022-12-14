from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
import torch
import transforms3d as t3d


class SCRDataset(Dataset):
    def __init__(self, dataset_path, labels_file, intrinsics, n_kps=None, n_augment=10):

        super(SCRDataset, self).__init__()
        self.img_paths, self.poses, self.scenes, self.scenes_ids, self.sp_paths = read_labels_file(labels_file, dataset_path)
        self.dataset_size = self.poses.shape[0]
        self.num_scenes = np.max(self.scenes_ids) + 1
        self.n_kps = n_kps
        proj_mats = np.zeros((self.dataset_size, 3, 4))
        for i, p in enumerate(self.poses):
            pose_mat = t3d.affines.compose(self.poses[i, :3].reshape(3),
                                           t3d.quaternions.quat2mat(self.poses[i, 3:]),
                                                                    np.ones(3))
            inv_pose_mat = np.linalg.inv(pose_mat)
            proj_mats[i] = np.dot(intrinsics, inv_pose_mat[:3, :])
        self.proj_mats = proj_mats.astype(np.float32)
        self.n_augment=n_augment


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.n_augment > 0:
            j = np.random.randint(self.n_augment)
            features = torch.load(self.sp_paths[idx].replace(".pth", "_aug_{}.pth".format(j)))
        else:
            features = torch.load(self.sp_paths[idx])
        pose = self.poses[idx]
        scene = self.scenes_ids[idx]

        sample = {}
        sample["pose"] = pose
        sample["scene"] = scene

        if self.n_kps is not None:
            sample["keypoints"] = features["keypoints"][:self.n_kps, :]
            sample["descriptors"] = features["descriptors"][: ,:self.n_kps].transpose(0,1)
            sample["xyz"] = features["xyz"][:self.n_kps, :]
        else:
            sample["keypoints"] = features["keypoints"]
            sample["descriptors"] = features["descriptors"].transpose(0,1)
            sample["xyz"] = features["xyz"][:self.n_kps, :]

        sample["shape"] = np.array(features["shape"])
        sample["proj_mat"] = self.proj_mats[idx]

        return sample


def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    scenes = df['scene'].values
    scene_unique_names = np.unique(scenes)
    scene_name_to_id = dict(zip(scene_unique_names, list(range(len(scene_unique_names)))))
    scenes_ids = [scene_name_to_id[s] for s in scenes]
    sp_paths = [join(join(dataset_path, 'sp'), path) for path in df['sp_path'].values]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return imgs_paths, poses, scenes, scenes_ids, sp_paths