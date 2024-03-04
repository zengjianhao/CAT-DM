import random
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from einops import rearrange
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)
from ldm.models.diffusion.ddpm import DDPM
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.util import instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like, make_ddim_sampling_parameters
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def disabled_train(self, mode=True):
    return self


# =============================================================
# 可训练部分 ControlNet
# =============================================================
class ControlNet(nn.Module):
    def __init__(
            self,
            in_channels,                                # 9
            model_channels,                             # 320
            hint_channels,                              # 20
            attention_resolutions,                      # [4,2,1]
            num_res_blocks,                             # 2
            channel_mult=(1, 2, 4, 8),                  # [1,2,4,4]
            num_head_channels=-1,                       # 64
            transformer_depth=1,                        # 1
            context_dim=None,                           # 768
            use_checkpoint=False,                       # True
            dropout=0,
            conv_resample=True,
            dims=2,
            num_heads=-1,
            use_scale_shift_norm=False):
        super(ControlNet, self).__init__()
        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        # time 编码器
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # input 编码器
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        # hint 编码器
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        # UNet
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    num_heads = ch // num_head_channels
                    dim_head = num_head_channels
                    disabled_sa = False

                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
    
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))
    
    def forward(self, x, hint, timesteps, reference_dino):
        # 处理输入
        context = reference_dino
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        guided_hint = self.input_hint_block(hint, emb)

        # 预测 control
        outs = []
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs

# =============================================================
# 固定参数部分 ControlledUnetModel
# =============================================================
class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None):
        hs = []
        
        # UNet 的上半部分
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        # 注入 control
        if control is not None:
            h += control.pop()
        
        # UNet 的下半部分
        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        # 输出
        h = h.type(x.dtype)
        h = self.out(h)

        return h

# =============================================================
# 主干网络 ControlLDM
# =============================================================
class ControlLDM(DDPM):
    def __init__(self,
                 control_stage_config,          # ControlNet
                 first_stage_config,            # AutoencoderKL
                 cond_stage_config,             # FrozenCLIPImageEmbedder
                 scale_factor=1.0,              # 0.18215
                 *args, **kwargs):
        self.num_timesteps_cond = 1
        super().__init__(*args, **kwargs)                                       # self.model 和 self.register_buffer
        self.control_model = instantiate_from_config(control_stage_config)      # self.control_model
        self.instantiate_first_stage(first_stage_config)                        # self.first_stage_model 调用 AutoencoderKL
        self.instantiate_cond_stage(cond_stage_config)                          # self.cond_stage_model 调用 FrozenCLIPImageEmbedder
        self.proj_out=nn.Linear(1024, 768)                                      # 全连接层
        self.scale_factor = scale_factor                                        # 0.18215
        self.learnable_vector = nn.Parameter(torch.randn((1,1,768)), requires_grad=False)
        self.trainable_vector = nn.Parameter(torch.randn((1,1,768)), requires_grad=True)
        self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
        self.dinov2_vits14.eval()
        self.dinov2_vits14.train = disabled_train
        for param in self.dinov2_vits14.parameters():
            param.requires_grad = False 
        self.linear = nn.Linear(1024, 768)

    # AutoencoderKL 不训练
    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    # FrozenCLIPImageEmbedder 不训练
    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

    # 训练
    def training_step(self, batch, batch_idx):
        z_new, reference, hint= self.get_input(batch)               # 加载数据
        loss= self(z_new, reference, hint)                          # 计算损失
        self.log("loss",                                            # 记录损失
                 loss,                                  
                 prog_bar=True,
                 logger=True, 
                 on_step=True, 
                 on_epoch=True)
        self.log('lr_abs',                                          # 记录学习率
                 self.optimizers().param_groups[0]['lr'], 
                 prog_bar=True, 
                 logger=True, 
                 on_step=True, 
                 on_epoch=False)
        return loss

    # 加载数据
    @torch.no_grad()
    def get_input(self, batch):
        
        # 加载原始数据
        x, inpaint, mask, reference, hint = super().get_input(batch)
        
        # AutoencoderKL 处理真值
        encoder_posterior = self.first_stage_model.encode(x)                           
        z = self.scale_factor * (encoder_posterior.sample()).detach()
        
        # AutoencoderKL 处理 inpaint
        encoder_posterior_inpaint = self.first_stage_model.encode(inpaint)     
        z_inpaint = self.scale_factor * (encoder_posterior_inpaint.sample()).detach()
        
        # Resize mask
        mask_resize = torchvision.transforms.Resize([z.shape[-2],z.shape[-1]])(mask)
        
        # 整理 z_new
        z_new = torch.cat((z,z_inpaint,mask_resize),dim=1)
        out  = [z_new, reference, hint]
        
        return out
    
    # 计算损失
    def forward(self, z_new, reference, hint):
        
        # 随机时间 t
        t = torch.randint(0, 50, (z_new.shape[0],), device=self.device).long()
        
        # CLIP 处理 reference
        reference_clip = self.cond_stage_model.encode(reference)
        reference_clip = self.proj_out(reference_clip)

        # DINO 处理 reference
        dino = self.dinov2_vits14(reference,is_training=True)
        dino1 = dino["x_norm_clstoken"].unsqueeze(1)
        dino2 = dino["x_norm_patchtokens"]
        reference_dino = torch.cat((dino1, dino2), dim=1)
        reference_dino = self.linear(reference_dino)

        # 随机加噪
        noise = torch.randn_like(z_new[:,:4,:,:])
        x_noisy = self.q_sample(x_start=z_new[:,:4,:,:], t=t, noise=noise)           
        x_noisy = torch.cat((x_noisy, z_new[:,4:,:,:]),dim=1)
        
        # 预测噪声
        model_output = self.apply_model(x_noisy, hint, t, reference_clip, reference_dino)

        # 计算损失
        loss = self.get_loss(model_output, noise, mean=False).mean([1, 2, 3])
        loss = loss.mean()
        
        return loss
    
    # 预测噪声
    def apply_model(self, x_noisy, hint, t, reference_clip, reference_dino):

        # 预测 control
        control = self.control_model(x_noisy, hint, t, reference_dino)

        # 调用 PBE
        model_output = self.model(x_noisy, t, reference_clip, control)

        return model_output

    # 优化器
    def configure_optimizers(self):
        # 学习率设置
        lr = self.learning_rate
        params = list(self.control_model.parameters())+list(self.linear.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        return opt
    
    # 采样
    @torch.no_grad()
    def sample_log(self, batch, ddim_steps=4):
        z_new, reference, hint = self.get_input(batch)
        x, _, mask, _, _ = super().get_input(batch)
        log = dict()

        log["mask"] = mask

        # 处理数据

        inpaint = z_new[:,4:8,:,:]
        mask = z_new[:,8:,:,:]

        # 处理设置
        device = self.betas.device
        b = z_new.shape[0]

        # 处理参数 t
        c = 10000 // ddim_steps
        ddim_timesteps = np.asarray(list(range(0, 1000, c)))
        ddim_timesteps = ddim_timesteps + 1
        time_range = np.flip(ddim_timesteps)
        total_steps = ddim_timesteps.shape[0]

        # 处理预参数
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=ddim_timesteps,
                                                                                   eta=0,
                                                                                   verbose=False)
        # 修改后的代码
        self.ddim_sigmas = ddim_sigmas
        self.ddim_alphas = ddim_alphas
        # self.register_buffer('ddim_sigmas', ddim_sigmas)
        # self.register_buffer('ddim_alphas', ddim_alphas)
        self.ddim_alphas_prev =  ddim_alphas_prev
        # self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.ddim_sqrt_one_minus_alphas = np.sqrt(1. - ddim_alphas)
        # self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))


        iterator = tqdm(time_range, desc='CAT-DM', total=total_steps)
        img = super().q_sample(z_new[:,:4,:,:], torch.full((z_new.shape[0],), time_range[-1], device=self.device, dtype=torch.long))
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            outs = self.p_sample_ddim(img, inpaint, mask, hint, reference, ts, index)

            img, _ = outs



        samples = 1. / self.scale_factor * img
        x_samples = self.first_stage_model.decode(samples[:,:4,:,:])
        # log["samples"] = x_samples

        x = torchvision.transforms.Resize([512, 512])(x)
        reference = torchvision.transforms.Resize([512, 512])(reference)
        x_samples = torchvision.transforms.Resize([512, 512])(x_samples)
        log["grid"] = torch.cat((x, reference, x_samples), dim=2)
        
        return log
    



    def p_sample(self, x, t, reference, hint, inpaint, mask):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x, t, reference, hint, inpaint, mask)
        noise = noise_like(x.shape, device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    def p_mean_variance(self, x, t, reference, hint, inpaint, mask):
        reference_clip = self.cond_stage_model.encode(reference)
        reference_clip = self.proj_out(reference_clip)
        dino = self.dinov2_vits14(reference,is_training=True)
        dino1 = dino["x_norm_clstoken"].unsqueeze(1)
        dino2 = dino["x_norm_patchtokens"]
        reference_dino = torch.cat((dino1, dino2), dim=1)
        reference_dino = self.linear(reference_dino)
        model_out  = self.apply_model(torch.cat([x, inpaint, mask], dim=1), hint, t, reference_clip, reference_dino)
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        x_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_sample_ddim(self,
                      x,
                      inpaint,
                      mask,
                      hint,
                      reference,
                      t,
                      index):
        b, *_, device = *x.shape, x.device

        x = torch.cat([x, inpaint, mask],dim=1)

        reference_clip = self.cond_stage_model.encode(reference)
        reference_clip= self.proj_out(reference_clip)
        dino = self.dinov2_vits14(reference,is_training=True)
        dino1 = dino["x_norm_clstoken"].unsqueeze(1)
        dino2 = dino["x_norm_patchtokens"]
        reference_dino = torch.cat((dino1, dino2), dim=1)
        reference_dino = self.linear(reference_dino)
        control = self.control_model(x, hint, t, reference_dino)
        e_t = self.model(x, t, reference_clip, control)


        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
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
