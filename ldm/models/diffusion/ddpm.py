import torch
import einops
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from functools import partial
from torchvision.utils import make_grid
from ldm.util import default, count_params, instantiate_from_config
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.transforms import Resize
import random



def disabled_train(self, mode=True):
    return self


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, unet_config):
        super().__init__()
        self.diffusion_model = instantiate_from_config(unet_config)

    def forward(self, x, timesteps=None, context=None, control=None):
        out = self.diffusion_model(x, timesteps, context, control)
        return out


class DDPM(pl.LightningModule):
    def __init__(self,
                 unet_config,
                 linear_start=1e-4,                 # 0.00085
                 linear_end=2e-2,                   # 0.0120
                 log_every_t=100,                   # 200
                 timesteps=1000,                    # 1000
                 image_size=256,                    # 32
                 channels=3,                        # 4
                 u_cond_percent=0,                  # 0.2
                 use_ema=True,                      # False
                 beta_schedule="linear",
                 loss_type="l2",
                 clip_denoised=True,
                 cosine_s=8e-3,
                 original_elbo_weight=0.,
                 v_posterior=0.,
                 l_simple_weight=1.,              
                 parameterization="eps",
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.):
        super().__init__()
        self.parameterization = parameterization
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.image_size = image_size 
        self.channels = channels
        self.u_cond_percent=u_cond_percent
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config) # 调用 UNet 模型

        self.use_ema = use_ema
        self.use_scheduler = True
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        self.register_schedule(beta_schedule=beta_schedule,     # "linear"
                               timesteps=timesteps,             # 1000
                               linear_start=linear_start,       # 0.00085
                               linear_end=linear_end,           # 0.0120
                               cosine_s=cosine_s)               # 8e-3
        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))

    def register_schedule(self, 
                          beta_schedule="linear", 
                          timesteps=1000,
                          linear_start=0.00085, 
                          linear_end=0.0120, 
                          cosine_s=8e-3):
        betas = make_beta_schedule(beta_schedule, 
                                   timesteps, 
                                   linear_start=linear_start, 
                                   linear_end=linear_end,
                                   cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

    def get_input(self, batch):
        x = batch['GT']
        mask = batch['inpaint_mask']
        inpaint = batch['inpaint_image']
        reference = batch['ref_imgs']
        hint = batch['hint']

        x = x.to(memory_format=torch.contiguous_format).float()
        mask = mask.to(memory_format=torch.contiguous_format).float()
        inpaint = inpaint.to(memory_format=torch.contiguous_format).float()
        reference = reference.to(memory_format=torch.contiguous_format).float()
        hint = hint.to(memory_format=torch.contiguous_format).float()

        return x, inpaint, mask, reference, hint

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if mean:
            loss = torch.nn.functional.mse_loss(target, pred)
        else:
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        return loss

