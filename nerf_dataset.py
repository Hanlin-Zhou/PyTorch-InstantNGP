import torch
from torch.utils.data import Dataset
import os
import glob
import json
from tqdm import tqdm
import cv2 as cv
import numpy as np

class NerfDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path):
        super().__init__()
        self.root_path = root_path
        self.aabb_scale = 3.0
        self.offset = torch.FloatTensor([0.5, 0.5, 0.5])

        # in [0, 1] or the grid hashing breaks
        self.aabb = torch.tensor([0., 0., 0., 1., 1., 1])
        # self.render_step_size =

        self.load_data()

    def load_data(self):
        transforms = sorted(glob.glob(os.path.join(self.root_path, "*.json")))
        train_transform = transforms[0]
        if len(transforms) != 1:
            print("Currently only support one split")
            print("Continuing with only train data...")
            for i in range(len(transforms)):
                if "train" in transforms[i]:
                    train_transform = transforms[i]
                    break
        with open(train_transform, 'r') as f:
            json_file = json.load(f)

        imgs = []
        poses = []
        for frame in tqdm(range(len(json_file['frames']))):
            curr_frame = json_file['frames'][frame]

            if curr_frame['file_path']:
                img_path = os.path.join(self.root_path, curr_frame['file_path'].replace("\\", "/"))
                ext = os.path.splitext(os.listdir(os.path.split(img_path)[0])[0])[-1]
                img_path = img_path + ext
                if os.path.exists(img_path):
                    img = cv.imread(img_path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH | cv.IMREAD_UNCHANGED)
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
                    imgs.append(torch.from_numpy(img.astype(np.float32) / 255.))
                else:
                    print(f"Can't find {img_path}.")
            else:
                print("No file_path during Nerf Dataset construction")

            if curr_frame["transform_matrix"]:
                poses.append(torch.tensor(curr_frame["transform_matrix"]))
            else:
                print("No transform_matrix during Nerf Dataset construction")


        self.imgs = torch.stack(imgs)
        self.poses = torch.stack(poses)

        img_h, img_w = self.imgs.shape[1:3]

        self.height = img_h
        self.width = img_w

        if "camera_angle_x" in json_file:
            camera_angle_x = json_file['camera_angle_x']
            fx = (0.5 * img_w) / np.tan(0.5 * float(camera_angle_x))
            if 'camera_angle_y' in json_file:
                camera_angle_y = json_file['camera_angle_y']
                fy = (0.5 * img_h) / np.tan(0.5 * float(camera_angle_y))
            else:
                fy = fx

        self.K = torch.tensor(
            [[fx, 0, img_h / 2.0],
                [0, fy, img_w / 2.0],
                [0, 0, 1]], dtype=torch.float32, )

        self.poses[..., :3, 3] /= self.aabb_scale
        self.poses[..., :3, 3] += self.offset

    def __len__(self):
        return self.imgs.shape[0]

    def pixel_per_img(self):
        return self.width * self.height

    @torch.no_grad()
    def __getitem__(self, index):
        """ Rays and color for image of index"""
        x, y = torch.meshgrid(
            torch.arange(self.width, device=self.imgs.device),
            torch.arange(self.height, device=self.imgs.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        pixel_val = self.imgs[index, y, x]
        cam2world = self.poses[index]

        dirs = torch.stack([(x - self.K[0][2]) / self.K[0][0], -(y - self.K[1][2]) / self.K[1][1],
                            -torch.ones_like(x)], -1)
        rays_direction = torch.sum(dirs[..., np.newaxis, :] * cam2world[:3, :3], -1)
        rays_direction = torch.nn.functional.normalize(rays_direction)
        rays_origin = cam2world[:3, -1].expand(rays_direction.shape)

        color, alpha = torch.split(pixel_val, [3, 1], dim=-1)
        bg_color = torch.ones(3, device=self.imgs.device)  # white background!
        color = color * alpha + bg_color * (1.0 - alpha)
        return rays_origin, rays_direction, color





