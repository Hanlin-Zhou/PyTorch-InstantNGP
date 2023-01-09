import torch
import torch.nn as nn
from torch import optim, cuda
from decoder import Decoder
from grid import Grid
from common import *

class ImageTrainer:
    def __init__(self, args):
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
        self.grid = Grid(args.feature_dim, args.grid_dim, args.num_lvl, args.max_res, args.min_res,
                         args.hashtable_power, args.force_cpu)
        self.decoder = Decoder(args.feature_dim * args.num_lvl, args.output_dim, args.activation, args.last_activation,
                               args.bias, args.num_layer, args.hidden_dim)
        self.result_dir = args.result_directory
        self.source_path = args.source_directory
        if args.force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = get_device()
        self.logger = logging.getLogger()
        self.grid.to(device=self.device)
        self.decoder.to(device=self.device)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam([
            {'params': self.grid.parameters()},
            {'params': self.decoder.parameters(), 'weight_decay ': self.weight_decay}
        ], lr=self.lr, betas=self.betas, eps=self.eps)
        loaded_image = Image(self.source_path, self.device)
        self.image_shape = loaded_image.shape
        self.lerp_img = torch.jit.trace(loaded_image, torch.rand([self.batch_size, 2],
                                                                 device=self.device, dtype=torch.float32))

    def train(self):
        print("starting...")
        for i in range(self.num_epoch):
            epoch = i + 1

            data = torch.rand([self.batch_size, 2], device=self.device, dtype=torch.float32)
            target = self.lerp_img(data)

            feature = self.grid(data)
            out = self.decoder(feature)

            self.optimizer.zero_grad()
            loss = self.loss_func(out, target)
            loss.backward()
            self.optimizer.step()

            # Logging
            loss_val = loss.item()
            psnr = mse2psnr(loss_val, self.range)
            loss_msg = f"Epoch#{epoch}: loss={loss_val:.8f}  PSNR:{psnr:.4f}"
            self.logger.info(loss_msg)
            if epoch % self.log_every == 0:
                print(loss_msg)

            # Saving result
            with torch.no_grad():
                if epoch == 1 and self.save_gt:
                    print("Saving Ground Truth")
                    copy_to_dir(self.source_path, self.result_dir, "GroundTruth")
                elif epoch % self.save_every == 0 or epoch == self.num_epoch:
                    print(f"----- Saving on Epoch {epoch} -----")
                    # render = self.grid(self.get_render_grid_float())
                    # render = self.decoder(render)
                    dst_file = os.path.join(self.result_dir, f"{epoch}" + ".jpg")
                    write_image(self.render(), dst_file)
        print("Training Finished:)")

    def get_render_grid_float(self):
        resolution = self.image_shape
        half_dx = 0.5 / resolution[0]
        half_dy = 0.5 / resolution[1]
        xs = torch.linspace(half_dx, 1 - half_dx, resolution[0], device=self.device)
        ys = torch.linspace(half_dy, 1 - half_dy, resolution[1], device=self.device)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        return xy

    def render(self):

        render_grid = self.get_render_grid_float()
        total_size = render_grid.shape[0]
        bandwidth = 5000000
        num_split = np.ceil(total_size / bandwidth)
        stacks = []
        # since my memory not large enough
        for split in range(int(num_split)):
            if split % 100 == 0:
                print(f"Rendering Data {split} / {num_split}")
            s = int(split * bandwidth)
            e = int((split + 1) * bandwidth)
            render = self.grid(render_grid[s:e, :])
            render = self.decoder(render)
            stacks.append(render.cpu())
        all_render = torch.cat(stacks, 0)
        result = all_render.reshape(self.image_shape).clamp(0.0, 1.0).detach().cpu().numpy()
        return result

