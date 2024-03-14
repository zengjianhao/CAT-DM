<div align="center">

<h1>CAT-DM: Controllable Accelerated Virtual Try-on with Diffusion Model</h1>

<div>
     <a href="https://zengjianhao.github.io/" target="_blank">Jianhao Zeng</a><sup>1</sup>,
     <a href="http://seea.tju.edu.cn/info/1014/1460.htm" target="_blank">Dan Song</a><sup>1,*</sup>,
     <a href="https://seea.tju.edu.cn/info/1014/1451.htm" target="_blank">Weizhi Nie</a><sup>1</sup>,
     <a href="https://seea.tju.edu.cn/info/1014/3931.htm" target="_blank">Hongshuo Tian</a><sup>1</sup>,
</div>
<div>
     <a href="https://tongttwang.github.io/" target="_blank">Tongtong Wang</a><sup>2</sup>,
     <a href="https://liuanantju.github.io/" target="_blank">Anan Liu</a><sup>1,*</sup>
</div>

<div>
    <sup>1</sup>Tianjin University &emsp; <sup>2</sup>Tencent LightSpeed Studio
</div>

Paper: [https://arxiv.org/abs/2311.18405](https://arxiv.org/abs/2311.18405)
Progect: [https://zengjianhao.github.io/CAT-DM](https://zengjianhao.github.io/CAT-DM)

<img src="./assets/CAT-DM.png" style="width:30%;">

</div>



The code has not been fully uploaded......

## Abstract

> Image-based virtual try-on enables users to virtually try on different garments by altering original clothes in their photographs. Generative Adversarial Networks (GANs) dominate the research field in image-based virtual try-on, but have not resolved problems such as unnatural deformation of garments and the blurry generation quality. Recently, diffusion models have emerged with surprising performance across various image generation tasks. While the generative quality of diffusion models is impressive, achieving controllability poses a significant challenge when applying it to virtual try-on tasks and multiple denoising iterations limit its potential for real-time applications. In this paper, we propose Controllable Accelerated virtual Try-on with Diffusion Model called CAT-DM. To enhance the controllability, a basic diffusion-based virtual try-on network is designed, which utilizes ControlNet to introduce additional control conditions and improves the feature extraction of garment images. In terms of acceleration, CAT-DM initiates a reverse denoising process with an implicit distribution generated by a pre-trained GAN-based model. Compared with previous try-on methods based on diffusion models, CAT-DM not only retains the pattern and texture details of the in-shop garment but also reduces the sampling steps without compromising generation quality. Extensive experiments demonstrate the superiority of CAT-DM against both GAN-based and diffusion-based methods in producing more realistic images and accurately reproducing garment patterns.

## Hardware Requirement

Our experiments were conducted on two NVIDIA GeForce RTX 4090 graphics cards, with a single RTX 4090 having 24GB of video memory. Please note that our model cannot be trained on graphics cards with less video memory than the RTX 4090.

## Environment Requirement

1.   Clone the repository

```bash
git clone https://github.com/zengjianhao/CAT-DM
```

2.   A suitable `conda` environment named `CAT-DM` can be created and activated with:

```bash
cd CAT-DM
conda env create -f environment.yaml
conda activate CAT-DM
```

-   If you want to change the name of the environment you created, you need to modify the `name` in both `environment.yaml` and `setup.py`.
-   You need to make sure that `conda` is installed on your computer.
-   If there is a network error, try updating the environment using `conda env update -f environment.yaml`.

3.   Installing xFormers：

```bash
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git submodule update --init --recursive
pip install -r requirements.txt
pip install -U xformers
cd ..
rm -rf xformers
```

4.   open `src/taming-transformers/taming/data/utils.py`, delete `from torch._six import string_classes`, and change `elif isinstance(elem, string_classes):` to `elif isinstance(elem, str):`

## Dataset Preparing

### VITON-HD

1.  Download the [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset
2.  Create a folder `datasets`
3.  Put the VITON-HD dataset into this folder and rename it to `vitonhd`
4.  Generate the mask images

```bash
# Generate the train dataset mask images
python tools/mask_vitonhd.py datasets/vitonhd/train datasets/vitonhd/train/mask
# Generate the test dataset mask images
python tools/mask_vitonhd.py datasets/vitonhd/test datasets/vitonhd/test/mask
```

### DressCode

1. Download the [DressCode](https://github.com/aimagelab/dress-code) dataset
2. Create a folder `datasets`
3. Put the DressCode dataset into this folder and rename it to `dresscode`
4. Generate the mask images and the agnostic images

```bash
# Generate the dresses dataset mask images and the agnostic images
python tools/mask_dresscode.py datasets/dresscode/dresses datasets/dresscode/dresses/mask
# Generate the lower_body dataset mask images and the agnostic images
python tools/mask_dresscode.py datasets/dresscode/lower_body datasets/dresscode/lower_body/mask
# Generate the upper_body dataset mask images and the agnostic images
python tools/mask_dresscode.py datasets/dresscode/upper_body datasets/dresscode/upper_body/mask
```

### Details
`datasets` folder should be as follows:

```
datasets
├── vitonhd
│   ├── test
│   │   ├── agnostic-mask
│   │   ├── mask
│   │   ├── cloth
│   │   ├── image
│   │   ├── image-densepose
│   │   ├── ...
│   ├── test_pairs.txt
│   ├── train
│   │   ├── agnostic-mask
│   │   ├── mask
│   │   ├── cloth
│   │   ├── image
│   │   ├── image-densepose
│   │   ├── ...
│   └── train_pairs.txt
├── dresscode
│   ├── dresses
│   │   ├── dense
│   │   ├── images
│   │   ├── mask
│   │   ├── ...
│   ├── lower_body
│   │   ├── dense
│   │   ├── images
│   │   ├── mask
│   │   ├── ...
│   ├── upper_body
│   │   ├── dense
│   │   ├── images
│   │   ├── mask
│   │   ├── ...
│   ├── test_pairs_paired.txt
│   ├── test_pairs_unpaired.txt
│   ├── train_pairs.txt
│   └── ...
```
PS: When we conducted the experiment, VITON-HD did not release the `agnostic-mask`. We used our own implemented `mask`, so if you are using VITON-HD's `agnostic-mask`, the generated results may vary.


## Required Model

1. Download the [Paint-by-Example](https://drive.google.com/file/d/15QzaTWsvZonJcXsNv-ilMRCYaQLhzR_i/view) model
2. Create a folder `checkpoints`
3. Put the Paint-by-Example model into this folder and rename it to `pbe.ckpt`
4. Make the ControlNet model:

- VITON-HD:
```bash
python tools/add_control.py checkpoints/pbe.ckpt checkpoints/pbe_dim6.ckpt configs/train_vitonhd.yaml
```

- DressCode:
```bash
python tools/add_control.py checkpoints/pbe.ckpt checkpoints/pbe_dim5.ckpt configs/train_dresscode.yaml
```

5.   `checkpoints` folder should be as follows:

```
checkpoints
├── pbe.ckpt
├── pbe_dim5.ckpt
└── pbe_dim6.ckpt
```


## Training

### VITON-HD

```bash
bash scripts/train_vitonhd.sh
```

### DressCode

```bash
bash scripts/train_dresscode.sh
```


## Testing

### VITON-HD

1. Download the [checkpoint](https://huggingface.co/JianhaoZeng/CAT-DM/tree/main) for VITON-HD dataset and put it into `checkpoints` folder.

2. Directly generate the try-on results:

```bash
bash scripts/test_vitonhd.sh
```

3. Poisson Blending

```python
python tools/poisson_vitonhd.py
```

### DressCode

1. Download the [checkpoint](https://huggingface.co/JianhaoZeng/CAT-DM/tree/main) for DressCode dataset and put it into `checkpoints` folder.

2. Directly generate the try-on results:

```bash
bash scripts/test_dresscode.sh
```

3. Poisson Blending

```python
python tools/poisson_dresscode.py
```

## Evaluation

### FID

### KID

### SSIM

### LPIPS





## Citing

```
@article{zeng2023cat,
  title={CAT-DM: Controllable Accelerated Virtual Try-on with Diffusion Model},
  author={Zeng, Jianhao and Song, Dan and Nie, Weizhi and Tian, Hongshuo and Wang, Tongtong and Liu, Anan},
  journal={arXiv preprint arXiv:2311.18405},
  year={2023}
}
```
