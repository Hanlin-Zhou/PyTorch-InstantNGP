import torch
import torch.nn as nn
from torch import optim, cuda
import cv2 as cv
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from torchinfo import summary
import logging
from utils import *

class Trainer:
    def __init__(self,
                 num_epoch: int,
                 batch_size: int,
                 range_clamping: bool,
                 save_every: int,
                 log_every: int,
                 learning_rate,
                 betas,
                 eps,
                 weight_decay,
                 grid,
                 decoder,
                 result_dir,
                 source_path,
                 force_cpu
                 ):
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.range_clamping = range_clamping
        self.save_every = save_every
        self.log_every = log_every
        self.lr = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grid = grid
        self.decoder = decoder
        self.result_dir = result_dir
        self.source_path = source_path
        if force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = get_device()
        self.logger = logging.getLogger()
        self.setup()

    def setup(self):
        self.grid.to(device=self.device)
        self.decoder.to(device=self.device)

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam([
                {'params': self.grid.parameters()},
                {'params': self.decoder.parameters(), 'weight_decay ': self.weight_decay}
            ], lr=self.lr, betas=self.betas, eps=self.eps)

        self.ground_truth = self.read_image()
        train_input = torch.rand([self.batch_size * self.num_epoch, 2], dtype=torch.float32)
        index = train_input * torch.tensor([self.ground_truth.shape[0], self.ground_truth.shape[1]])
        train_target = self.ground_truth[index.to(torch.int32)[:, 0], index.to(torch.int32)[:, 1]]
        train_target = torch.tensor(train_target[..., 0:3]).to(torch.float32)

        train_input.to(device=self.device)
        train_target.to(device=self.device)
        dataset = TensorDataset(train_input, train_target)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size)


    def train(self):
        for ii, (data, target) in enumerate(self.dataloader):
            epoch = ii + 1

            data = data.to(device=self.device)
            target = target.to(device=self.device)

            feature = self.grid(data)
            out = self.decoder(feature)

            self.optimizer.zero_grad()
            loss = self.loss_func(out, target)
            loss.backward()
            self.optimizer.step()

            # Logging
            loss_val = loss.item()
            loss_msg = f"Epoch#{epoch}: loss={loss_val} "
            self.logger.info(loss_msg)
            if epoch % self.log_every == 0:
                loss_val = loss.item()
                print(loss_msg)

            # Saving result
            with torch.no_grad():
                if epoch == 1:
                    self.save_img(self.ground_truth, "GroundTruth")
                elif epoch % self.save_every == 0 or epoch == self.num_epoch:
                    print(f"----- Saving on Epoch {epoch} -----")
                    render = self.grid(self.get_render_grid_float())
                    render = self.decoder(render)
                    self.save_img(render.reshape(self.ground_truth.shape).clamp(0.0, 1.0).detach().cpu().numpy(),
                                  f"{epoch}")
        print("Training Finished:)")

    def read_image(self):
        img = cv.imread(self.source_path).astype(np.float32)
        if self.range_clamping:
            img /= 255.0
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img

    def save_img(self, img, filename):
        name = self.result_dir + filename + ".jpg"
        if self.range_clamping:
            img = (img * 255.).astype(np.int)
        else:
            img = img.astype(np.int)
        cv.imwrite(name, img)

    def get_render_grid_float(self):
        resolution = self.ground_truth.shape
        half_dx = 0.5 / resolution[0]
        half_dy = 0.5 / resolution[1]
        xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=self.device)
        ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=self.device)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((xv.flatten(), yv.flatten())).t()
        return xy
