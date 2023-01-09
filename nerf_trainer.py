import torch
import torch.nn as nn
from torch import optim, cuda
from decoder import Decoder
from grid import Grid
import nerfacc
from nerf_dataset import *
from common import *

class NerfTrainer:
    def __init__(self, args):
        if args.force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = get_device()
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.range_clamping = args.range_clamping
        self.range = 1.
        self.save_every = args.save_every
        self.save_gt = args.save_gt
        self.log_every = args.log_every
        self.lr = args.learning_rate
        self.betas = (args.beta1, args.beta2)
        self.eps = args.eps
        self.weight_decay = args.weight_decay
        self.view_encoding = ViewEncoding(args.view_encoding_degree)
        self.grid = Grid(args.feature_dim, args.grid_dim, args.num_lvl, args.max_res, args.min_res,
                         args.hashtable_power, args.force_cpu)
        #  1 hidden layer for density decoder according to paper
        self.density_decoder = Decoder(args.feature_dim * args.num_lvl, args.output_dim, args.activation,
                                       args.last_activation, args.bias, 1, args.hidden_dim)
        #  2 hidden layer for color decoder according to paper
        self.color_decoder = Decoder(args.output_dim + self.view_encoding.encoding_size, 3, args.activation,
                                     args.last_activation, args.bias, 2, args.hidden_dim)

        self.grid.to(device=self.device)
        self.density_decoder.to(device=self.device)
        self.color_decoder.to(device=self.device)
        self.result_dir = args.result_directory
        self.source_path = args.source_directory
        self.logger = logging.getLogger()
        self.grid.to(device=self.device)
        self.density_decoder.to(device=self.device)
        self.color_decoder.to(device=self.device)
        self.view_encoding.to(device=self.device)
        self.dataset = NerfDataset(self.source_path)
        self.loss_func = nn.MSELoss()
        self.render_pose = args.render_pose_id

        self.optimizer = optim.Adam([
            {'params': self.grid.parameters()},
            {'params': self.density_decoder.parameters(), 'weight_decay ': self.weight_decay},
            {'params': self.color_decoder.parameters(), 'weight_decay ': self.weight_decay}],
            lr=self.lr, betas=self.betas, eps=self.eps)

    def train(self):
        print("starting...")
        for epoch in range(1, self.num_epoch + 1):
            total_loss = 0
            for img in range(len(self.dataset)):
                index = torch.randint(self.dataset.pixel_per_img(), (self.batch_size,))
                ray_origin, ray_direction, target = self.dataset[img]
                ray_origin = ray_origin[index].to(device=self.device)
                ray_direction = ray_direction[index].to(device=self.device)
                target = target[index].to(device=self.device)
                color, alpha, depth = self.render(ray_origin, ray_direction)
                bg_color = torch.ones(3, device=color.device)  # white background!
                color = color * alpha + bg_color * (1.0 - alpha)

                self.optimizer.zero_grad()
                loss = self.loss_func(color, target)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f"Image{img} / {len(self.dataset)}")
            # Logging
            psnr = mse2psnr(total_loss, self.range)
            loss_msg = f"Epoch#{epoch}: loss={total_loss:.8f}  PSNR:{psnr:.4f}"
            self.logger.info(loss_msg)
            if epoch % self.log_every == 0:
                print(loss_msg)

            # Saving result
            with torch.no_grad():
                if epoch == 1 and self.save_gt:
                    print("Saving ground truth...")
                    gt_file = os.path.join(self.result_dir, "reference.jpg")
                    gt = ((self.dataset[self.render_pose])[2].cpu().numpy()).reshape((self.dataset.height,
                                                                                      self.dataset.width,
                                                                                      3))
                    write_image(gt, gt_file)
                if epoch % self.save_every == 0 or epoch == self.num_epoch + 1:
                    print(f"----- Saving on Epoch {epoch} -----")
                    dst_file = os.path.join(self.result_dir, f"{epoch}" + ".jpg")
                    ray_origin, ray_direction, target = self.dataset[self.render_pose]
                    num_rays = ray_origin.shape[0]
                    chunk = 330000
                    colors = []
                    for i in range(0, num_rays, chunk):
                        ray_chunk_o = ray_origin[i: i + chunk].to(device=self.device)
                        ray_chunk_d = ray_direction[i: i + chunk].to(device=self.device)
                        color, alpha, depth = self.render(ray_chunk_o, ray_chunk_d, 20000)
                        bg_color = torch.ones(3, device=color.device)  # white background!
                        color = color * alpha + bg_color * (1.0 - alpha)
                        colors.append(color.cpu().numpy())
                        print(f"{i}/{num_rays}")
                    write_image(np.concatenate(colors, 0).reshape((self.dataset.height,
                                                                   self.dataset.width, 3)).clip(0.0, 1.0), dst_file)
        print("Training Finished:)")


    def render(self, rays_o, rays_d, chunk=4000):
        num_rays = rays_o.shape[0]
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = ray_chunk_o[ray_indices]
            t_dirs = ray_chunk_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            features = self.grid(positions)
            sigmas = torch.exp(self.density_decoder(features)[..., 0:1])
            return sigmas

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = ray_chunk_o[ray_indices]
            t_dirs = ray_chunk_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            features = self.grid(positions)
            density_decoder_out = self.density_decoder(features)
            color_decoder_input = torch.concat([self.view_encoding(t_dirs), density_decoder_out], -1)
            rgbs = self.color_decoder(color_decoder_input)
            sigmas = torch.exp(density_decoder_out[..., 0:1])
            return rgbs, sigmas

        result = []
        for i in range(0, num_rays, chunk):
            with torch.no_grad():
                ray_chunk_o = rays_o[i: i + chunk]
                ray_chunk_d = rays_d[i: i + chunk]
                ray_indices, t_starts, t_ends = nerfacc.ray_marching(ray_chunk_o,
                                                                     ray_chunk_d,
                                                                     sigma_fn=sigma_fn,
                                                                     near_plane=0.2,
                                                                     far_plane=5.0,
                                                                     scene_aabb=self.dataset.aabb.cuda(),
                                                                     early_stop_eps=1e-4,
                                                                     alpha_thre=0.0,
                                                                     render_step_size=1e-2)

            color, opacity, depth = nerfacc.rendering(t_starts, t_ends, ray_indices, n_rays=ray_chunk_o.shape[0],
                                                      rgb_sigma_fn=rgb_sigma_fn)
            result.append([color, opacity, depth])

        colors, opacities, depths = [torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*result)]
        return colors, opacities, depths
