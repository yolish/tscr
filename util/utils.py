import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath
import time
from os import mkdir, getcwd
import torch
import numpy as np
import torch.nn.functional as F
import cv2
from skimage import img_as_ubyte

# Logging and output utils
##########################
def get_stamp_from_log():
    """
    Get the time stamp from the log file
    :return:
    """
    return split(logging.getLogger().handlers[0].baseFilename)[-1].replace(".log","")


def create_output_dir(name):
    """
    Create a new directory for outputs, if it does not already exist
    :param name: (str) the name of the directory
    :return: the path to the outpur directory
    """
    out_dir = join(getcwd(), name)
    if not exists(out_dir):
        mkdir(out_dir)
    return out_dir


def init_logger():
    """
    Initialize the logger and create a time stamp for the file
    """
    path = split(realpath(__file__))[0]

    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        filename = log_config_dict.get('handlers').get('file_handler').get('filename')
        filename = ''.join([filename, "_", time.strftime("%d_%m_%y_%H_%M", time.localtime()), ".log"])

        # Creating logs' folder is needed
        log_path = create_output_dir('out')

        log_config_dict.get('handlers').get('file_handler')['filename'] = join(log_path, filename)
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)



# Evaluation utils
##########################
def pose_err(est_pose, gt_pose):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1)
    est_pose_q = F.normalize(est_pose[:, 3:], p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose[:, 3:], p=2, dim=1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))
    orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / np.pi
    return posit_err, orient_err

# SuperPoint
##########################
def process_superpoint_resize(w, h, resize):

    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def superpoint_transform(image, resize=None, resize_float=1, rotation=0, convert_from_scikit_image=False): #modifed from superglue's utils
    """
    :param image: a grayscale image read with cv2.imread
    :param resize: Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize
    :param resize_float: Resize the image after casting uint8 to float
    :param rotation: integer, indicating whether and in how many degress to rotate image
    :param convert_from_scikit_image: boolean, indicating if the image was read with scikit and needs to be converted
    :return:
    """

    '''
    Notes from experiments in PoseLab:
    results are very sensitive to resie_float (true/false)
    results vary when using scikit image, even after conversion
    '''

    if image is None:
        return None, None, None
    if convert_from_scikit_image:
        image = cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_RGB2GRAY)

    if resize is not None:
        w, h = image.shape[1], image.shape[0]

        assert (len(resize) > 0 and len(resize) <= 2)
        if len(resize) == 1 and resize[0] > -1:
            scale = resize[0] / max(h, w)
            w_new, h_new = int(round(w * scale)), int(round(h * scale))
        elif len(resize) == 1 and resize[0] == -1:
            w_new, h_new = w, h
        else:  # len(resize) == 2:
            w_new, h_new = resize[0], resize[1]

        # Issue warning if resolution is too small or too large.
        if max(w_new, h_new) < 160:
            print('Warning: input resolution is very small, results may vary')
        elif max(w_new, h_new) > 2000:
            print('Warning: input resolution is very large, results may vary')

        w_new, h_new = process_superpoint_resize(w, h, resize)
        scales = (float(w) / float(w_new), float(h) / float(h_new))

        if resize_float:
            image = cv2.resize(image.astype('float32'), (w_new, h_new))
        else:
            image = cv2.resize(image, (w_new, h_new)).astype('float32')
    else:
        scales = None

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = torch.from_numpy(image / 255.).float()[None, None]
    return image, inp, scales