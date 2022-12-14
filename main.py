"""
Entry point training and testing TSCR
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.SCRDataset import SCRDataset
from models.TSCR import TSCR
from os.path import join
import transforms3d as t3d
import cv2


def project_to_2d(xyz, proj_mat, w, h, scale_factor=1.0, clip=True):
    # code adapted from https://github.com/Tangshitao/Dense-Scene-Matching/blob/master/libs/model/geometry.py
    batch_size, n, _ = xyz.shape
    xyz = xyz.transpose(1, 2)
    ones = torch.ones(batch_size ,1, n, device=xyz.device)
    coords_homo = torch.cat([xyz, ones], dim=1)

    proj_mat = proj_mat.reshape(-1, 3, 4)
    proj_kpts = torch.bmm(proj_mat, coords_homo)[:, :, :]

    '''
    if clip:
        z = torch.clamp(proj_kpts[:, 1:2, :], min=0.1)
    else:
        z = proj_kpts[:, 1:2, :] + 1e-5
    '''

    z = proj_kpts[:, 2, :]

    valid_mask = z > 1e-5

    proj_kpts = (proj_kpts[:, :, :] / z.unsqueeze(1))[:, :2, :]

    proj_kpts = proj_kpts / scale_factor
    valid_mask = (
        valid_mask
        * (
            (proj_kpts[:, 0, :] >= 0)
            * (proj_kpts[:, 1, :] >= 0)
            * (proj_kpts[:, 0, :] <= w - 1)
            * (proj_kpts[:, 1, :] <= h - 1)
        ).float()
    )

    return proj_kpts, valid_mask.unsqueeze(1)

def estimate_pose(minibatch, est_xyz, camera_intrinsic):
    gt_pose = minibatch.get('pose').to(dtype=torch.float32)[0].unsqueeze(0)
    u, v = minibatch["keypoints"][0, :, 0].cpu().numpy(), \
           minibatch["keypoints"][0, :, 1].cpu().numpy()
    # Solve PnP ransac
    _, rvec, est_t, _ = cv2.solvePnPRansac(est_xyz, np.array(list(zip(u, v))).astype(np.float32),
                                           camera_intrinsic,
                                           None, False)
    rot = cv2.Rodrigues(rvec)[0].reshape((3, 3))
    est_extrinsic = t3d.affines.compose(est_t.reshape(3), rot, np.ones(3))
    est_pose_mat = np.linalg.inv(est_extrinsic)
    est_pose = np.zeros(7)
    est_pose[:3] = est_pose_mat[0:3, 3]
    est_pose[3:] = t3d.quaternions.mat2quat(est_pose_mat[:3, :3])
    est_pose = torch.Tensor(est_pose).unsqueeze(0).to(device)
    posit_err, orient_err = utils.pose_err(est_pose, gt_pose)
    return est_pose, posit_err, orient_err


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("mode", help="train or eval")
    arg_parser.add_argument("dataset_path", help="path to the physical location of the dataset")
    arg_parser.add_argument("labels_file", help="path to a file mapping images to their poses")
    arg_parser.add_argument("config_file", help="path to configuration file", default="7scenes-config.json")
    arg_parser.add_argument("--checkpoint_path",
                            help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")

    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {}ing TSCR".format(args.mode))

    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Create the model
    model = TSCR(config).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))
    camera_intrinsic = np.array(config.get("intrinsic"))

    #mean_xyz = torch.Tensor(config.get("mean_xyz")).to(device)
    #sd_xyz = torch.Tensor(config.get("sd_xyz")).to(device)

    if args.mode == 'train':
        # Set to train mode
        model.train()

        # Set the loss
        scr_loss = torch.nn.MSELoss()
        # Set the optimizer and scheduler
        params = model.parameters()
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Set the dataset and data loader
        equalize_scenes = config.get("equalize_scenes")
        dataset = SCRDataset(args.dataset_path, args.labels_file, camera_intrinsic, config.get("n_kps"))
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")


        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_xyz = minibatch["xyz"]
                batch_size = gt_xyz.shape[0]
                n_samples += batch_size

                # Zero the gradients
                optim.zero_grad()

                # Forward pass to estimate the pose
                xyz = model(minibatch)
                # Unnormalize
                #xyz = (xyz * sd_xyz.unsqueeze(0).repeat(batch_size, xyz.shape[1], 1)) + mean_xyz.unsqueeze(0).repeat(batch_size, xyz.shape[1], 1)
                if epoch < 100:
                    criterion = (xyz - gt_xyz).abs().sum() #scr_loss(xyz, gt_xyz)
                else:
                    proj_mats = minibatch["proj_mat"]
                    h, w = minibatch["shape"][0]
                    gt_kpts = minibatch["keypoints"]
                    proj_kpts, mask = project_to_2d(xyz, proj_mats, w, h)
                    l1_err = (proj_kpts * mask - gt_kpts.transpose(1,2) * mask).abs()
                    #proj_kpts_xx, mask_xx = project_to_2d(gt_xyz, proj_mats, w, h)
                    #x = (proj_kpts_xx * mask_xx - gt_kpts.transpose(1,2) * mask_xx).abs()
                    #x =  x.sum()
                    criterion = l1_err.sum() / torch.sum(mask)

                # Collect for recoding and plotting
                running_loss += criterion.item()

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    with torch.no_grad():
                        _, posit_err, orient_err = estimate_pose(minibatch,
                                                                 xyz[0,:, :].detach().cpu().numpy().astype(np.float32),
                                                                 camera_intrinsic)

                        logging.info("Batch-{}/Epoch-{}] running l1 loss: {:.3f}, Pose error: {:.3f}[m], {:.3f}[deg]".format(
                            batch_idx+1, epoch+1, running_loss/n_samples, posit_err.item(), orient_err.item()))
                    running_loss = 0.0
                    n_samples = 0
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_tscr_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_tscr_final.pth'.format(epoch))

    else: # Test
        # Set to eval mode
        model.eval()

        # Set the dataset and data loader
        dataset = SCRDataset(args.dataset_path, args.labels_file, camera_intrinsic, n_kps=607,n_augment=0)
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

        stats = np.zeros((len(dataloader.dataset), 3))

        with torch.no_grad():
            for i, minibatch in enumerate(dataloader, 0):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)

                # Forward pass to predict the scene coordinates
                tic = time.time()
                est_xyz = model(minibatch)
                #est_xyz = (est_xyz * sd_xyz.unsqueeze(0).repeat(1, est_xyz.shape[1], 1)) + \
                #          mean_xyz.unsqueeze(0).repeat(1, est_xyz.shape[1], 1)

                est_xyz = est_xyz.cpu().numpy()[0].astype(np.float32)
                est_pose, posit_err, orient_err = estimate_pose(minibatch, est_xyz, camera_intrinsic)
                toc = time.time()

                # Collect statistics
                stats[i, 0] = posit_err.item()
                stats[i, 1] = orient_err.item()
                stats[i, 2] = (toc - tic)*1000

                logging.info("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[i, 0],  stats[i, 1],  stats[i, 2]))

        # Record overall statistics
        logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
        logging.info("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
        logging.info("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))
