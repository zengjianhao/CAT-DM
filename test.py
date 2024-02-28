import os
import torch
import argparse
import torchvision
import pytorch_lightning as pl
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from tqdm import tqdm
from PIL import Image
from torch import autocast
from einops import rearrange
from functools import partial
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like, make_ddim_sampling_parameters
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def un_norm(x):
    return (x+1.0)/2.0

def un_norm_clip(x):
    x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
    x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
    x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
    return x

def p_sample_ddim(model,
                  x,
                  inpaint,
                  mask,
                  hint,
                  reference,
                  t,
                  index, 
                  ddim_sigmas, 
                  ddim_alphas, 
                  ddim_alphas_prev, 
                  ddim_sqrt_one_minus_alphas):
    b, *_, device = *x.shape, x.device

    x = torch.cat([x, inpaint, mask],dim=1)

    reference_clip = model.cond_stage_model.encode(reference)
    reference_clip= model.proj_out(reference_clip)
    dino = model.dinov2_vits14(reference,is_training=True)
    dino1 = dino["x_norm_clstoken"].unsqueeze(1)
    dino2 = dino["x_norm_patchtokens"]
    reference_dino = torch.cat((dino1, dino2), dim=1)
    reference_dino = model.linear(reference_dino)
    control = model.control_model(x, hint, t, reference_dino)
    e_t = model.model(x, t, reference_clip, control)


    alphas = ddim_alphas
    alphas_prev = ddim_alphas_prev
    sqrt_one_minus_alphas = ddim_sqrt_one_minus_alphas
    sigmas = ddim_sigmas
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)


    pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    noise = sigma_t * noise_like(dir_xt.shape, device, False)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise


    return x_prev, pred_x0


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, 
                 batch_size,                        # N
                 train=None,                        # {...}
                 validation=None,                   # {...}
                 test=None,                         # {...}
                 wrap=False,                        # False
                 shuffle_val_dataloader=False,      # False
                 shuffle_test_loader=False,         # False
                 use_worker_init_fn=False):         # False
        super().__init__()
        print("???????????")
        self.batch_size = batch_size
        self.num_workers = batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        self.wrap = wrap
        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs["train"] = train                                               # {...}
            self.train_dataloader = self._train_dataloader                                      # shuffle = True
        if validation is not None:
            self.dataset_configs["validation"] = validation                                     # {...}
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader) # shuffle = False
        if test is not None:
            self.dataset_configs["test"] = test                                                 # {...}
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)  # shuffle = False


    def setup(self):
        # 分别加载对应的类，并组成一个字典
        # {"train":         OpenImageDataset(state="train",         arbitrary_mask_percent=0.5, image_size=256, dataset_dir="/home/sd/Harddisk/zjh_diffusion/Dataset")}
        # {"validation":    OpenImageDataset(state="validation",    arbitrary_mask_percent=0.5, image_size=256, dataset_dir="/home/sd/Harddisk/zjh_diffusion/Dataset")}
        # {"test":          OpenImageDataset(state="test",          arbitrary_mask_percent=0.5, image_size=256, dataset_dir="/home/sd/Harddisk/zjh_diffusion/Dataset")}
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                          shuffle=True,
                          worker_init_fn=None)

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle,
                          worker_init_fn=None)

    def _test_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["test"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle,
                          worker_init_fn=None)


if __name__ == "__main__":
    # =============================================================
    # 处理 opt
    # =============================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base", type=str, default="configs/test.yaml")
    parser.add_argument("-c", "--ckpt", type=str, default="./model.ckpt")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-d", "--ddim", type=int, default=64)
    parser.add_argument("-t", "--type", type=str, default="unpaired")
    opt = parser.parse_args()



    # =============================================================
    # 设置 seed
    # =============================================================
    seed_everything(opt.seed)

    # =============================================================
    # 初始化 config
    # =============================================================
    config = OmegaConf.load(f"{opt.base}")

    # =============================================================
    # 加载 dataloader
    # =============================================================
    data = instantiate_from_config(config.data)
    data.setup()
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")


    # =============================================================
    # 加载 model
    # =============================================================
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.ckpt, map_location="cpu")["state_dict"], strict=False)
    model.cuda()
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # =============================================================
    # 设置精度
    # =============================================================
    precision_scope = autocast
    os.makedirs("results50/P-GP2D")
    os.makedirs("results50/P-GP2C")


    # =============================================================
    # 开始测试
    # =============================================================
    with torch.no_grad():
        with precision_scope("cuda"):
            for i,batch in enumerate(data.test_dataloader()):
                # 加载数据
                inpaint = batch["inpaint_image"].to(torch.float16).to(device)
                reference = batch["ref_imgs"].to(torch.float16).to(device)
                mask = batch["inpaint_mask"].to(torch.float16).to(device)
                hint = batch["hint"].to(torch.float16).to(device)
                truth = batch["GT"].to(torch.float16).to(device)
                gan = batch["gan"].to(torch.float16).to(device)

                encoder_posterior = model.first_stage_model.encode(gan)                           
                z_gan = model.scale_factor * (encoder_posterior.sample()).detach()

                encoder_posterior_inpaint = model.first_stage_model.encode(inpaint)
                z_inpaint = model.scale_factor * (encoder_posterior_inpaint.sample()).detach()

                mask_resize = torchvision.transforms.Resize([z_inpaint.shape[-2],z_inpaint.shape[-1]])(mask)


                # 数据处理
                c = 1000 // opt.ddim # 每次跳多少步
                ddim_timesteps = np.asarray(list(range(0, 1000, c)))
                ddim_timesteps = ddim_timesteps + 1
                time_range = np.flip(ddim_timesteps) # 1000 950 900 ..... 1
                total_steps = ddim_timesteps.shape[0]
                ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=model.alphas_cumprod.cpu(),
                                                                                           ddim_timesteps=ddim_timesteps,
                                                                                           eta=0,
                                                                                           verbose=False)
                ddim_sqrt_one_minus_alphas = np.sqrt(1. - ddim_alphas)

                iterator = tqdm(time_range, desc='CAT-DM', total=total_steps)
                img = model.q_sample(z_gan, torch.full((z_gan.shape[0],), time_range[-1], device=model.device, dtype=torch.long))
                for j, step in enumerate(iterator):
                    index = total_steps - j - 1
                    ts = torch.full((z_gan.shape[0],), step, device=device, dtype=torch.long)
                    outs = p_sample_ddim(model, img, z_inpaint, mask_resize, hint, reference, ts, index, ddim_sigmas, ddim_alphas, ddim_alphas_prev, ddim_sqrt_one_minus_alphas)

                    img, _ = outs

                samples = 1. / model.scale_factor * img
                x_samples = model.first_stage_model.decode(samples[:,:4,:,:])

                x_samples_ddim = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                x_checked_image=x_samples_ddim
                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)       
                # 保存图像
                all_img=[]
                all_img_C = []

                mask = mask.cpu().permute(0, 2, 3, 1).numpy()
                mask = torch.from_numpy(mask).permute(0, 3, 1, 2)
                truth = torch.clamp((truth + 1.0) / 2.0, min=0.0, max=1.0)
                truth = truth.cpu().permute(0, 2, 3, 1).numpy()
                truth = torch.from_numpy(truth).permute(0, 3, 1, 2)
                x_checked_image_torch_C = x_checked_image_torch*(1-mask) + truth.cpu()*mask
                x_checked_image_torch = torch.nn.functional.interpolate(x_checked_image_torch.float(), size=[512,384])
                x_checked_image_torch_C = torch.nn.functional.interpolate(x_checked_image_torch_C.float(), size=[512,384])
                
                all_img.append(x_checked_image_torch[0])
                all_img_C.append(x_checked_image_torch_C[0])


                grid = torch.stack(all_img, 0)
                grid = torchvision.utils.make_grid(grid)
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                img = Image.fromarray(grid.astype(np.uint8))
                img.save('results50/P-GP2D/'+str(i)+".png")

                grid_C = torch.stack(all_img_C, 0)
                grid_C = torchvision.utils.make_grid(grid_C)
                grid_C = 255. * rearrange(grid_C, 'c h w -> h w c').cpu().numpy()
                img_C = Image.fromarray(grid_C.astype(np.uint8))
                img_C.save('results50/P-GP2C/'+str(i)+".png")


                