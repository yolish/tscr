import torch
import pandas as pd
from util import utils
import numpy as np
from os.path import join
import os
import argparse
import cv2
from models.SuperPoint import SuperPoint
import transforms3d as t3d
import matplotlib.pyplot as plt
from torchvision import transforms
import skimage.io


def draw_kp(img, x, y, color, size=1, thickness=2):
    x = np.round(x).astype(np.int)
    y = np.round(y).astype(np.int)
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)
    return img

def make_keypoints(x_coords, y_coords):
        keypoints = []

        for i in range(len(x_coords)):
            x = x_coords[i]
            y = y_coords[i]
            kp = cv2.KeyPoint()
            kp.pt = (x, y)
            keypoints.append(kp)
        return keypoints

def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    scenes = df['scene'].values
    scene_unique_names = np.unique(scenes)
    scene_name_to_id = dict(zip(scene_unique_names, list(range(len(scene_unique_names)))))
    scenes_ids = [scene_name_to_id[s] for s in scenes]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return imgs_paths, poses, scenes, scenes_ids

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_path", help="path where images are", default="/media/yoli/WDC-2.0-TB-Hard-/7Scenes/")
    arg_parser.add_argument("--sp_path", help="path to superpoint model", default="/media/yoli/WDC-2.0-TB-Hard-/"
                                                                                       "pretrained_models/"
                                                                                    "superpoint_v1.pth")
    arg_parser.add_argument("--labels_file", help="path to a file mapping images to their poses", default="datasets/7Scenes/abs_7scenes_pose.csv_heads_train.csv")
    arg_parser.add_argument("--device_id", help="torch device id", default="cuda:0")

    args = arg_parser.parse_args()


    out_path = join(args.data_path, "sp/")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    labels_file = args.labels_file
    device_id = "cuda:0"

    # Create and load the model
    x = torch.cuda.is_available()
    device = torch.device(device_id)
    sp = SuperPoint({"keypoint_threshold":0.0}).to(device).eval()
    sp.load_state_dict(torch.load(args.sp_path, map_location=args.device_id))
    sp.to(device)
    sp.eval()

    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                        transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)])


    sanity = False
    augmnt = False#True
    n_kps = 1024
    min_kps = 1024
    if augmnt:
        n_augmentations = 10
    else:
        n_augmentations = 1
    dataset_name = "7scenes"
    proj_th = 1
    # compute keypoints  and save them
    sp_paths = []
    img_paths, poses, scenes, scenes_ids =  read_labels_file(args.labels_file, args.data_path)
    n = len(img_paths)

    camera_intrinsic_dict = {
        "cambridge":np.array([[418.7109375, 0, 910], [0, 418.7109375, 540], [0, 0, 1]]),
         "7scenes""":np.array([[585.0, 0, 320.0], [0, 585.0, 240.0], [0, 0, 1]])}
    camera_intrinsic = camera_intrinsic_dict[dataset_name]
    all_xyz = []
    #if augmnt:
    #    camera_intrinsic[0,2] = 160
    #    camera_intrinsic[1,2] = 160

    with torch.no_grad():
        for i, img_path in enumerate(img_paths):
            print("Extracting SP points from image {}/{} at {}".format(i, n, img_path))
            for j in range(n_augmentations):
                img = cv2.imread(img_path, 0)
                img_shape = img.shape
                if augmnt:
                    img = skimage.io.imread(img_path)
                    img = train_transforms(img)
                    img = np.array(img)
                    _, input, _ = utils.superpoint_transform(img, None, False,
                                                             convert_from_scikit_image=True)
                else:
                    img = cv2.imread(img_path, 0)
                    _, input, _ = utils.superpoint_transform(img, None, False,
                                                                    convert_from_scikit_image=False)
                input = input.to(device)
                sp_features = sp({"image": input})

                # Sort by score before saving scores, descriptors and keypoints
                sorting_indices = torch.argsort(sp_features["scores"][0], descending=True).cpu()[:n_kps]
                sp_features["scores"] = sp_features["scores"][0].cpu()[sorting_indices]
                sp_features["descriptors"] = sp_features["descriptors"][0].cpu()[:, sorting_indices]
                sp_features["keypoints"] = sp_features["keypoints"][0].cpu()[sorting_indices, :]
                sp_features["shape"] = img_shape

                # compute scene coordinates
                u, v = sp_features["keypoints"][:, 0].numpy().astype(np.int64), sp_features["keypoints"][:, 1].numpy().astype(np.int64)
                n = v.shape[0]
                depth_path = img_path.replace("color", "depth")
                depth_img = cv2.imread(depth_path, 0)
                # go rom pixels (u, v) to world coordinates (Xw, Yx, Zw).
                inv_intrinsic = np.linalg.inv(camera_intrinsic)
                rmat = t3d.quaternions.quat2mat(poses[i][3:])
                pose = t3d.affines.compose(poses[i][:3], rmat, np.ones(3))

                uvz_mat = np.ones((3, n))
                #depth_img[depth_img == 0] = np.nan
                z = depth_img[(v, u)]
                uvz_mat[0, :] = np.array(u) * z
                uvz_mat[1, :] = np.array(v) * z
                uvz_mat[2, :] = z

                # Project pixels to camera's plane (u,v) ==> (Xc, Yc) at the focal plane
                xyz_cam = np.dot(inv_intrinsic, uvz_mat)

                # Project from camera to world (Xc, Yc, Zc) ==> (Xw, Yw, Zw)
                xyz_cam = np.concatenate((xyz_cam, np.ones((1, n))))
                xyz_world = np.dot(pose, xyz_cam)[:3]
                sp_features["xyz"] = torch.Tensor(xyz_world.transpose())

                # Filter keypoints with bad projections
                extrinsic = np.linalg.inv(pose)
                tvec = extrinsic[0:3, 3]
                rvec = extrinsic[0:3, 0:3]
                proj_pnts_2d, _ = cv2.projectPoints(xyz_world, rvec, tvec, camera_intrinsic, None)
                u_proj = proj_pnts_2d[:, 0, 0]
                v_proj = proj_pnts_2d[:, 0, 1]

                # compute pixel diff
                print("Before - Mean pixel diff x, y: {}, {}".format(np.mean(np.abs(u_proj - u)),
                                                                     np.mean(np.abs(v_proj - v))))
                valid = np.unique(np.concatenate([np.where(np.abs(v_proj - v) < proj_th)[0],
                                                  np.where(np.abs(u_proj-u) < proj_th)[0]]))


                sp_features["scores"] = sp_features["scores"][valid]
                sp_features["descriptors"] = sp_features["descriptors"][:, valid]
                sp_features["keypoints"] = sp_features["keypoints"][valid, :]
                sp_features["xyz"] = sp_features["xyz"][valid, :]

                print("After - Mean pixel diff x, y: {}, {}".format(np.mean(np.abs(u_proj[valid] - u[valid])),
                                                                   np.mean(np.abs(v_proj[valid] - v[valid]))))

                if len(valid) < min_kps:
                    min_kps = len(valid)

                outfile_name = img_path.replace(args.data_path, "").replace("/", "_").\
                    replace(".jpg", "").replace(".jpeg", "").replace(".png", "") + "_sp.pth"
                outfile_name = join(out_path, outfile_name)
                if augmnt:
                    torch.save(sp_features, outfile_name.replace(".pth", "_aug_{}.pth".format(j)))
                    print("Features saved to {}".format(outfile_name.replace(".pth", "_aug_{}.pth".format(j))))
                else:
                    torch.save(sp_features, outfile_name)
                    print("Features saved to {}".format(outfile_name))
                if j == 0:
                    sp_paths.append(outfile_name.replace(out_path, ""))


                if sanity:
                    u, v = sp_features["keypoints"][:, 0].numpy(), sp_features["keypoints"][:, 1].numpy()
                    xyz_world = sp_features["xyz"].numpy()

                    proj_pnts_2d, _ = cv2.projectPoints(xyz_world, rvec, tvec, camera_intrinsic, None)
                    u_proj = proj_pnts_2d[:, 0, 0]
                    v_proj = proj_pnts_2d[:, 0, 1]

                    kps = make_keypoints(u.astype(np.float32), v.astype(np.float32))
                    img_with_kps = cv2.drawKeypoints(img, kps, None, (0, 0, 255))
                    proj_kps = make_keypoints(u_proj, v_proj)
                    img_with_proj_kps = img_with_kps.copy()
                    for p in proj_kps:
                        img_with_proj_kps = draw_kp(img_with_proj_kps, p.pt[0], p.pt[1], (255, 0, 0))

                    fig = plt.figure(figsize=(1, 2))
                    fig.suptitle("Sanity")
                    ax = fig.add_subplot(121, title='Extracted')
                    plt.imshow(img_with_kps)
                    ax = fig.add_subplot(122, title='Extracted + Projected')
                    plt.imshow(img_with_proj_kps)
                    plt.show()

                    # compute pose error
                    orig_pose = torch.Tensor(poses[i]).unsqueeze(0)
                    _, rvec, est_t, _ = cv2.solvePnPRansac(xyz_world.astype(np.float32),
                                                           np.array(list(zip(u,v))).astype(np.float32), camera_intrinsic,
                                                                              None, False)

                    rot = cv2.Rodrigues(rvec)[0].reshape((3, 3))
                    est_extrinsic = t3d.affines.compose(est_t.reshape(3), rot, np.ones(3))
                    est_pose_mat = np.linalg.inv(est_extrinsic)
                    est_pose = np.zeros(7)
                    est_pose[:3] = est_pose_mat[0:3, 3]
                    est_pose[3:] = t3d.quaternions.mat2quat(est_pose_mat[:3, :3])
                    est_pose = torch.Tensor(est_pose).unsqueeze(0)
                    posit_err, orient_err = utils.pose_err(est_pose, orig_pose)
                    orient_err =  orient_err.item()
                    if np.isnan(orient_err):
                        orient_err = 0.0
                    print("Pose error: {:.3f}[m], {:.3f}[deg]".format(posit_err.item(), orient_err))

                #plot original and back-projected
#all_xyz = np.concatenate(all_xyz, axis=1)
#print("Mean x y z: {}".format(all_xyz.mean(axis=1)))
#print("SD x y z: {}".format(all_xyz.std(axis=1)))

print("Min kps: {}".format(min_kps))
df = pd.read_csv(args.labels_file)
df["sp_path"] = sp_paths
df.to_csv(labels_file+"_with_sp.csv", index=False)

# cambridge intrinsics



