import torch
import torch.nn as nn
import numpy as np
from common import *


class Grid(nn.Module):
    """
        input x needs to be within [0, 1]
    """
    def __init__(self,
                 feature_dim: int,
                 grid_dim: int,
                 num_lvl: int,
                 max_res: int,
                 min_res: int,
                 hashtable_power: int,
                 force_cpu: bool
                 ):
        super().__init__()
        if force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = get_device()
        self.feature_dim = feature_dim
        self.grid_dim = grid_dim
        self.num_lvl = num_lvl
        self.max_res = max_res
        self.min_res = min_res
        self.hashtable_power = hashtable_power
        self.prime = [3367900313, 2654435761, 805459861]
        self.max_entry = 2 ** self.hashtable_power
        self.factor_b = np.exp((np.log(self.max_res) - np.log(self.min_res)) / (self.num_lvl - 1))

        self.resolutions = []
        for i in range(self.num_lvl):
            self.resolutions.append(np.floor(self.min_res * (self.factor_b**i)))

        self.hashtable = nn.ParameterList([])
        for res in self.resolutions:
            total_res = res**self.grid_dim
            table_size = min(total_res, self.max_entry)
            # +/- 10**-4 in InstantNGP paper
            table = torch.randn(int(table_size), self.feature_dim, device=self.device) * 0.0001
            self.hashtable.append(table)



    def forward(self, x):
        out_feature = []
        for lvl in range(self.num_lvl):
            coord = self.to_hash_space(x, self.resolutions[lvl])
            floor_corner = torch.floor(coord)
            corners = self.get_corner(floor_corner).to(torch.long)
            feature_index = self.hash(corners, self.hashtable[lvl].shape[0], self.resolutions[lvl])
            flat_feature_index = feature_index.to(torch.long).flatten()
            corner_feature = torch.reshape(self.hashtable[lvl][flat_feature_index],
                                           (corners.shape[0], corners.shape[1], self.feature_dim))
            weights = self.interpolation_weights(coord - floor_corner)
            # weights = self.alt_weights(corners, coord)
            weights = torch.stack([weights, weights], -1)
            weighted_feature = corner_feature * weights
            summed_feature = weighted_feature.sum(-2)
            out_feature.append(summed_feature)
        return torch.cat(out_feature, -1)

    def to_hash_space(self, x, resolution):
        # don't want the (res-1, res-1) corner. Easier for later get_corner()
        return torch.clip(x * (resolution - 1), 0, resolution - 1.0001)

    def interpolation_weights(self, diff):
        ones = torch.ones_like(diff, device=self.device)
        minus_x = (ones - diff)[..., 0]
        x = diff[..., 0]
        minus_y = (ones - diff)[..., 1]
        y = diff[..., 1]
        if self.grid_dim == 2:
            stacks = torch.stack([minus_x * minus_y,
                                  x * minus_y,
                                  minus_x * y,
                                  x * y], -1)
            return stacks
        else:
            minus_z = (ones - diff)[..., 2]
            z = diff[..., 2]
            stacks = torch.stack([minus_x * minus_y * minus_z,
                                  x * minus_y * minus_z,
                                  minus_x * y * minus_z,
                                  x * y * minus_z,
                                  minus_x * minus_y * z,
                                  x * minus_y * z,
                                  minus_x * y * z,
                                  x * y * z], -1)
            return stacks

    def alt_weights(self, corner, coord):
        # change diag accoring to dim
        diag_length = torch.full_like(coord[:, 0], 2.**(1/2), device=self.device)
        w = torch.empty(corner.shape[0], corner.shape[1], device=self.device)
        for c in range(corner.shape[1]):
            dist = torch.norm(corner[:, c, :] - coord, dim=1)
            w[:, c] = diag_length - dist
        normed_w = torch.nn.functional.normalize(w, p=1)
        return normed_w

    def hash(self, x, num_entry, res):
        if num_entry != self.max_entry:
            index = 0
            for i in range(self.grid_dim):
                index += x[..., i] * res**i
            return index
        else:
            _sum = 0
            for i in range(self.grid_dim):
                _sum = _sum ^ (x[..., i] * self.prime[i])
            index = _sum % num_entry
            return index

    def get_corner(self, floor_corner):
        num_entry = floor_corner.shape[0]
        if self.grid_dim == 2:
            c000 = floor_corner
            c001 = floor_corner + torch.tensor([0, 1], device=self.device).repeat(num_entry, 1)
            c010 = floor_corner + torch.tensor([1, 0], device=self.device).repeat(num_entry, 1)
            c011 = floor_corner + torch.ones_like(floor_corner, device=self.device)
            stacks = torch.stack([c000, c010, c001, c011], -2)
            return stacks
        else:
            c000 = floor_corner
            c001 = floor_corner + torch.tensor([0, 0, 1], device=self.device).repeat(num_entry, 1)
            c010 = floor_corner + torch.tensor([0, 1, 0], device=self.device).repeat(num_entry, 1)
            c011 = floor_corner + torch.tensor([0, 1, 1], device=self.device).repeat(num_entry, 1)
            c100 = floor_corner + torch.tensor([1, 0, 0], device=self.device).repeat(num_entry, 1)
            c101 = floor_corner + torch.tensor([1, 0, 1], device=self.device).repeat(num_entry, 1)
            c110 = floor_corner + torch.tensor([1, 1, 0], device=self.device).repeat(num_entry, 1)
            c111 = floor_corner + torch.ones_like(floor_corner, device=self.device)
            stacks = torch.stack([c000, c010, c001, c011, c100, c101, c110, c111], -2)
            return stacks





