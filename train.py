import argparse, os, sys, datetime
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.util import instantiate_from_config

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    # default
    parser.add_argument("-n", "--name",             type=str,       nargs="?",  default="")
    parser.add_argument("-r", "--resume",           type=str,       nargs="?",  default="")
    parser.add_argument("-t", "--train",            type=str2bool,  nargs="?",  default=True)
    parser.add_argument("-s", "--seed",             type=int,       nargs="?",  default=3407)
    parser.add_argument("-f", "--postfix",          type=str,       nargs="?",  default="")
    parser.add_argument("--train_from_scratch",     type=str2bool,  nargs="?",  default=False)
    parser.add_argument("-d", "--debug",            type=str2bool,  nargs="?",  default=False)
    # train.sh
    parser.add_argument("-b", "--base",             type=str,       nargs="?",  default="configs/train_vitonhd.yaml")
    parser.add_argument("-l", "--logdir",           type=str,       nargs="?",  default="logs")
    parser.add_argument("-p", "--pretrained_model", type=str,       nargs="?",  default="checkpoints/pbe_dim6.ckpt")
    return parser


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


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.log_steps = [1]
        
    # 每个训练 batch 结束后调用
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="train")

    # 加载图片
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = pl_module.global_step
        if (self.check_frequency(check_idx) and hasattr(pl_module, "sample_log") and callable(pl_module.sample_log)):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                images = pl_module.sample_log(batch)
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    images[k] = torch.clamp(images[k], -1., 1.)
            self.log_local(pl_module.logger.save_dir, 
                           split, 
                           images,
                           pl_module.global_step, 
                           pl_module.current_epoch, 
                           batch_idx)

            self._testtube(pl_module, 
                           images, 
                           pl_module.global_step, 
                           split)

            if is_train:
                pl_module.train()

    # check_index 是 500 的倍数，或者 check_idx 在 [1, 2, 4, 8, 16, 32, 64, 128, 256] 中
    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and check_idx > 0:
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    # 存储在本地
    def log_local(self, 
                  save_dir, 
                  split, 
                  images,
                  global_step, 
                  current_epoch, 
                  batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)


class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)

        epoch = trainer.current_epoch
        if epoch % 5 == 0 and epoch !=0 :
            trainer.save_checkpoint(f'epoch={epoch}.ckpt')
        
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time
        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":

    sys.path.append(os.getcwd())

    # =============================================================
    # Get parser and generate opt
    # =============================================================
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()


    # =============================================================
    # Generate logdir path
    # =============================================================
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")     # 2023-05-18T04-27-24
    cfg_fname = os.path.split(opt.base)[-1]                         # train_vitonhd.yaml
    cfg_name = os.path.splitext(cfg_fname)[0]                       # train_vitonhd
    nowname = now + "_" + cfg_name                                  # 2023-05-18T04-27-24_train_vitonhd
    logdir = os.path.join(opt.logdir, nowname)                      # logs/2023-05-18T04-27-24_train_vitonhd
    ckptdir = os.path.join(logdir, "checkpoints")                   # logs/2023-05-18T04-27-24_train_vitonhd/checkpoints
    cfgdir = os.path.join(logdir, "configs")                        # logs/2023-05-18T04-27-24_train_vitonhd/configs


    # =============================================================
    # Set seed
    # =============================================================
    seed_everything(opt.seed)


    # =============================================================
    # Initialize config
    # =============================================================
    config = OmegaConf.load(opt.base)                                       # Load the yaml file to DictConfig
    lightning_config = config.pop("lightning", OmegaConf.create())          # Remove lightning from config and return lightning_config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())    # Extract trainer from lightning_config
    trainer_opt = argparse.Namespace(**trainer_config)                      # argparse.Namespace(accelerator='ddp', gpus='0,1', max_epochs=200, num_nodes=1)


    # =============================================================
    # Load model and initialize it
    # =============================================================
    # Use config.model["params"] to initialize config.model["target"]
    model = instantiate_from_config(config.model)
    # Load pre-trained model weights
    model.load_state_dict(torch.load(opt.pretrained_model, map_location='cpu'), strict=False)


    # =============================================================
    # Set trainer_kwargs
    # =============================================================
    trainer_kwargs = dict()

    # Gradient accumulation
    trainer_kwargs["accumulate_grad_batches"] = 8

    # Log the training process in logdir/testtube
    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.TestTubeLogger",
        "params": {"name": "testtube", "save_dir": logdir}
    }
    logger_cfg = OmegaConf.create(default_logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # Callbacks setting
    default_callbacks_cfg = {
        # 训练过程中将图片存储在本地文件夹和 tensorboard 中
        "image_logger": {
            "target": "train.ImageLogger",
            "params": {
                "batch_frequency": 2000,         # 保存图片的频率 
                "max_images": 4                 # grid 中最大图片数量
            }
        },
        # 训练过程中自动记录学习率
        "learning_rate_logger": {
            "target": "train.LearningRateMonitor",
            "params": {
                "logging_interval": "step",     # 根据 step 记录学习率
            }
        },
        # CUDA 和 GPU 相关
        "cuda_callback": {
            "target": "train.CUDACallback"
        },
    }
    callbacks_cfg = OmegaConf.create(default_callbacks_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]


    # =============================================================
    # 使用 trainer_kwargs 初始化 trainer
    # =============================================================
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir # models/Paint-by-Example/nowname


    # =============================================================
    # 加载数据集
    # =============================================================
    # 使用 config.data["params"] 初始化 main 中的 DataModuleFromConfig
    data = instantiate_from_config(config.data)
    # 分别初始化 ldm.data.open-images 的 OpenImageDataset 得到 self.datasets
    data.setup()
    # self.datasets.keys 有 train validation test
    # 给出各个数据集的大小
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # =============================================================
    # 设置 batchsize 和 learning_rate
    # =============================================================
    bs, base_lr= config.data.params.batch_size, config.model.base_learning_rate
    num_nodes = 1
    model.learning_rate = base_lr


    # =============================================================
    # 训练及测试
    # =============================================================
    trainer.fit(model, data)
    # trainer.test(model, data)
