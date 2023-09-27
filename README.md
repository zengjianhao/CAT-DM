# Faster Controllable Virtual Try-on with Diffusion Models

Paper | Huggingface Demo

Jianhao Zeng

## Abstract



## Environment Requirements

1.   Clone the repository

```bash
git clone https://github.com/zengjianhao/FC-VTON.git
```

2.   A suitable `conda` environment named `FC-VTON` can be created and activated with:

```bash
conda env create -f environment.yaml
conda activate FC-VTON
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

## Dataset Preparing



## Protrain Model



## Testing



## Training






