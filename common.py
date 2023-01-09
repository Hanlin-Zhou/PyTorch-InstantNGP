import cv2 as cv
import numpy as np
import torch
from torch import cuda
import imageio.v2 as imageio
import os
import sys
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
import logging
from datetime import datetime
import shutil

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
Image.MAX_IMAGE_PIXELS = None


def in_debug():
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def get_device():
    if cuda.is_available():
        print("Training on CUDA")
        return torch.device('cuda')
    else:
        print("Training on CPU")
        return torch.device('cpu')


def logging_setup(path):
    logging.basicConfig(level=logging.INFO, filename=path + '/trainer.log',
                        filemode='w', format='%(asctime)s|%(levelname)s| %(message)s')


def get_time_str():
    now = datetime.now()
    return now.strftime("%m%d%Y-%H%M%S")


def setup_result_output(args):
    new_result_dir_name = args.task_title + get_time_str()
    new_result_dir = os.path.join(args.result_directory, new_result_dir_name)
    os.mkdir(new_result_dir)
    args.result_directory = new_result_dir
    new_config_file = os.path.join(args.result_directory, "result_config.yaml")
    shutil.copyfile(args.config_path, new_config_file)


def copy_to_dir(file, dst_dir, new_name):
    ext = os.path.splitext(file)[1].lower()
    dst_file = os.path.join(dst_dir, new_name + ext)
    shutil.copyfile(file, dst_file)


def mse2psnr(x, r):
    return 20. * np.log(r)/np.log(10.) - 10.*np.log(x)/np.log(10.)


def interpolate_image_very_slow(img, x, y):
    batch_size = x.size
    bandwidth = 30000  # less than SHRT_MAX in cv2 32767
    num_split = np.ceil(batch_size/bandwidth)
    stacks = []
    for split in range(int(num_split)):
        print(f"Preparing Data {split} / {num_split}")
        s = int(split * bandwidth)
        e = int((split + 1) * bandwidth)
        interpolated = cv.remap(img.astype(np.float32), y[s:e], x[s:e],
                                interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        interpolated = interpolated.reshape(-1, img.shape[2])
        stacks.append(interpolated)
    check = np.concatenate(stacks, 0)
    return check


def interpolate_image_second_slow(img, coord):
    bordered_img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_REPLICATE)
    xc = np.linspace(0.5, bordered_img.shape[1] - 0.5, bordered_img.shape[1])
    yc = np.linspace(0.5, bordered_img.shape[0] - 0.5, bordered_img.shape[0])
    l = [yc, xc]
    interp = RegularGridInterpolator(l, bordered_img)
    # coord[:, [1, 0]] = coord[:, [0, 1]]
    coord += np.ones_like(coord)
    print("Interpolating targets...")
    result = interp(coord)
    print("Finished interpolating")
    return result


def read_image(path):
    if os.path.splitext(path)[1] == ".exr":
        img = cv.imread(path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH | cv.IMREAD_UNCHANGED)
    else:
        img = imageio.imread(path)
        img = np.asarray(img).astype(np.float32)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        if img.shape[2] == 4:
            img[..., 0:3] *= img[..., 3:4]
    return img / 255.


def write_image(img, path, quality=95):
    if img.shape[2] == 4:
        img = np.copy(img)
        img[..., 0:3] = np.divide(img[..., 0:3], img[..., 3:4], out=np.zeros_like(img[..., 0:3]),
                                  where=img[..., 3:4] != 0)
    img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    kwargs = {}
    if os.path.splitext(path)[1].lower() in [".jpg", ".jpeg"]:
        kwargs["quality"] = quality
        kwargs["subsampling"] = 0
    imageio.imwrite(path, img, **kwargs)


def img_pyramid(path, device):
    img = read_image(path)
    pyramid = []
    pyramid.append(img)
    total_miplvl = np.floor(np.log2(min(img.shape[0], img.shape[1])))
    print(f"Image mip lvl : {total_miplvl}")
    for i in range(total_miplvl):
        img = cv.pyrDown(img)
        pyramid.append(img)
    tensors = []
    for i in pyramid:
        tensors.append(torch.from_numpy(i).float().to(device))
    return tensors, total_miplvl


# This is a much faster way of interpolating
# use with torch jit
class Image(torch.nn.Module):
    def __init__(self, path, device):
        super(Image, self).__init__()
        self.data = read_image(path)
        self.shape = self.data.shape
        self.data = torch.from_numpy(self.data).float().to(device)

    def forward(self, xs):
        with torch.no_grad():
            shape = self.shape
            xs = xs * torch.tensor([shape[1], shape[0]], device=xs.device).float()
            indices = xs.long()
            lerp_weights = xs - indices.float()

            x0 = indices[:, 0].clamp(min=0, max=shape[1] - 1)
            y0 = indices[:, 1].clamp(min=0, max=shape[0] - 1)
            x1 = (x0 + 1).clamp(max=shape[1] - 1)
            y1 = (y0 + 1).clamp(max=shape[0] - 1)

            return (
                    self.data[y0, x0] * (1.0 - lerp_weights[:, 0:1]) * (1.0 - lerp_weights[:, 1:2]) +
                    self.data[y0, x1] * lerp_weights[:, 0:1] * (1.0 - lerp_weights[:, 1:2]) +
                    self.data[y1, x0] * (1.0 - lerp_weights[:, 0:1]) * lerp_weights[:, 1:2] +
                    self.data[y1, x1] * lerp_weights[:, 0:1] * lerp_weights[:, 1:2]
            )

# future experiment with multiscale supervision
class MultiscaleImage(torch.nn.Module):
    def __init__(self, path, device):
        super(MultiscaleImage, self).__init__()
        self.data, self.total_miplvl = img_pyramid(path, device)

    def forward(self, xs, scale):
        with torch.no_grad():
            mip_index = torch.log2(scale)
            mip_index = torch.clamp(mip_index, 0.0, self.total_miplvl)
            m0 = torch.floor(mip_index).long()
            m1 = torch.ceil(mip_index).long()
            m_lerp_weight = mip_index - m0.float()
            
            m0shape = self.data[m0].shape
            m0xs = xs * torch.tensor([m0shape[1], m0shape[0]], device=xs.device).float()
            m0indices = m0xs.long()
            m0lerp_weights = m0xs - m0indices.float()

            m0x0 = m0indices[:, 0].clamp(min=0, max=m0shape[1] - 1)
            m0y0 = m0indices[:, 1].clamp(min=0, max=m0shape[0] - 1)
            m0x1 = (m0x0 + 1).clamp(max=m0shape[1] - 1)
            m0y1 = (m0y0 + 1).clamp(max=m0shape[0] - 1)
            m0data = self.data[m0]
            m0val = m0data[m0y0, m0x0] * (1.0 - m0lerp_weights[:, 0:1]) * (1.0 - m0lerp_weights[:, 1:2]) + \
                    m0data[m0y0, m0x1] * m0lerp_weights[:, 0:1] * (1.0 - m0lerp_weights[:, 1:2]) + \
                    m0data[m0y1, m0x0] * (1.0 - m0lerp_weights[:, 0:1]) * m0lerp_weights[:, 1:2] + \
                    m0data[m0y1, m0x1] * m0lerp_weights[:, 0:1] * m0lerp_weights[:, 1:2]

            m1shape = self.data[m1].shape
            m1xs = xs * torch.tensor([m1shape[1], m1shape[0]], device=xs.device).float()
            m1indices = m1xs.long()
            m1lerp_weights = m1xs - m1indices.float()

            m1x0 = m1indices[:, 0].clamp(min=0, max=m1shape[1] - 1)
            m1y0 = m1indices[:, 1].clamp(min=0, max=m1shape[0] - 1)
            m1x1 = (m1x0 + 1).clamp(max=m1shape[1] - 1)
            m1y1 = (m1y0 + 1).clamp(max=m1shape[0] - 1)
            m1data = self.data[m1]
            m1val = m1data[m1y0, m1x0] * (1.0 - m1lerp_weights[:, 0:1]) * (1.0 - m1lerp_weights[:, 1:2]) + \
                    m1data[m1y0, m1x1] * m1lerp_weights[:, 0:1] * (1.0 - m1lerp_weights[:, 1:2]) + \
                    m1data[m1y1, m1x0] * (1.0 - m1lerp_weights[:, 0:1]) * m1lerp_weights[:, 1:2] + \
                    m1data[m1y1, m1x1] * m1lerp_weights[:, 0:1] * m1lerp_weights[:, 1:2]

            return m0val * (1.0 - m_lerp_weight) + m1val * m_lerp_weight
            

# From https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/hash_encoding.py
# Spherical harmonics basis
class ViewEncoding(torch.nn.Module):
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = (1.0925484305920792, -1.0925484305920792,
          0.31539156525252005, -1.0925484305920792,
          0.5462742152960396,)
    C3 = (-0.5900435899266435, 2.890611442640554,
          -0.4570457994644658, 0.3731763325901154,
          -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435,)
    C4 = (2.5033429417967046, -1.7701307697799304,
          0.9461746957575601, -0.6690465435572892,
          0.10578554691520431, -0.6690465435572892,
          0.47308734787878004, -1.7701307697799304,
          0.6258357354491761,)

    def __init__(self, degree=4):
        assert degree >= 1 and degree <= 5
        super().__init__()
        self.degree = degree
        self.encoding_size = degree ** 2

    def forward(self, input):
        result = torch.empty(
            (*input.shape[:-1], self.encoding_size), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)
        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
        return result
