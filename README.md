## DistgSSR: Disentangling Mechanism for Light Field Statial Super-Resolution
<br>
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgSSR/master/Figs/DistgSSR.png" width="90%"> </p>

This is the PyTorch implementation of the spatial SR method in our paper "Disentangling Light Fields for Super-Resolution and Disparity Estimation". Please refer to our [paper](https://yingqianwang.github.io/) and [project page](https://yingqianwang.github.io/DistgLF) for details.<br>

## News and Updates:
* 2022-02-22: Optimize `LFdivide` and `LFintegrate`, and modify our codes to enable inference with a batch of patches.
* 2022-02-22: Checkpoints `DistgSSR_4xSR_6x6.pth.tar` and `DistgSSR_4xSR_7x7.pth.tar` are available.
* 2022-02-22: Our DistgSSR has been added into the repository [*BasicLFSR*](https://github.com/ZhengyuLiang24/BasicLFSR).
* 2022-02-16: Our paper is accepted by IEEE TPAMI.

## Preparation:
#### 1. Requirement:
* PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.6, cuda=9.0.
* Matlab for training/test data generation and performance evaluation.
#### 2. Datasets:
* We used the EPFL, HCInew, HCIold, INRIA and STFgantry datasets for training and test. Please first download our dataset via [Baidu Drive](https://pan.baidu.com/s/1mYQR6OBXoEKrOk0TjV85Yw) (key:7nzy) or [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/zyliang_stu_xidian_edu_cn/EpkUehGwOlFIuSSdadq9S4MBEeFkNGPD_DlzkBBmZaV_mA?e=FiUeiv), and place the 5 datasets to the folder `./Datasets/`.
#### 3. Generating training/test data:
* Run `Generate_Data_for_Train.m` to generate training data. The generated data will be saved in `./Data/train_kxSR_AxA/`.
* Run `Generate_Data_for_Test.m` to generate test data. The generated data will be saved in `./Data/test_kxSR_AxA/`.
#### 4. Download our pretrained models:
We provide the models of each angular resolution (2×2 to 9×9) for 2×/4× SR. Download our models through the following links:
| **Upscaling Factor** |  **Angular Resolution** | **Channel Depth** | **Download Link** |
| :---------: |  :---------: | :----------: | :---------------: |
|    2×SR  |   5×5  |  32  | [DistgSSR_2xSR_5x5_C32.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_2xSR_5x5_C32.pth.tar) |
|    2×SR  |   2×2  |  64  | [DistgSSR_2xSR_2x2.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_2xSR_2x2.pth.tar) |
|    2×SR  |   3×3  |  64  | [DistgSSR_2xSR_3x3.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_2xSR_3x3.pth.tar) |
|    2×SR  |   4×4  |  64  | [DistgSSR_2xSR_4x4.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_2xSR_4x4.pth.tar) |
|    2×SR  |   5×5  |  64  | [**DistgSSR_2xSR_5x5.pth.tar**](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_2xSR_5x5.pth.tar) |
|    2×SR  |   6×6  |  64  | [DistgSSR_2xSR_6x6.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_2xSR_6x6.pth.tar) |
|    2×SR  |   7×7  |  64  | [DistgSSR_2xSR_7x7.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_2xSR_7x7.pth.tar) |
|    2×SR  |   8×8  |  64  | [DistgSSR_2xSR_8x8.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_2xSR_8x8.pth.tar) |
|    2×SR  |   9×9  |  64  | [DistgSSR_2xSR_9x9.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_2xSR_9x9.pth.tar) |
|    4×SR  |   5×5  |  32  | [DistgSSR_4xSR_5x5_C32.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_4xSR_5x5_C32.pth.tar) |
|    4×SR  |   2×2  |  64  | [DistgSSR_4xSR_2x2.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_4xSR_2x2.pth.tar) |
|    4×SR  |   3×3  |  64  | [DistgSSR_4xSR_3x3.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_4xSR_3x3.pth.tar) |
|    4×SR  |   4×4  |  64  | [DistgSSR_4xSR_4x4.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_4xSR_4x4.pth.tar) |
|    4×SR  |   5×5  |  64  | [**DistgSSR_4xSR_5x5.pth.tar**](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_4xSR_5x5.pth.tar) |
|    4×SR  |   6×6  |  64  | [DistgSSR_4xSR_6x6.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_4xSR_6x6.pth.tar) |
|    4×SR  |   7×7  |  64  | [DistgSSR_4xSR_7x7.pth.tar](https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgSSR_4xSR_7x7.pth.tar) |
|    4×SR  |   8×8  |  64  | Coming soon |
|    4×SR  |   9×9  |  64  | Coming soon |


## Train:
* Set the hyper-parameters in `parse_args()` if needed. We have provided our default settings in the realeased codes.
* Run `train.py` to perform network training.
* Checkpoint will be saved to `./log/`.

## Test on the datasets:
* Run `test_on_dataset.py` to perform test on each dataset.
* The original result files and the metric scores will be saved to `./Results/`.

## Test on your own LFs:
* Place the input LFs into `./input` (see the attached examples).
* Run `demo_test.py` to perform spatial super-resolution. Note that, the selected pretrained model should match the input in terms of the angular resolution. 
* The super-resolved LF images will be automatically saved to `./output`.

## Results:

### Quantitative Results:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgSSR/master/Figs/QuantitativeSSR.png" width="100%"> </p>

### Visual Comparisons:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgSSR/master/Figs/Visual-SSR.png" width="100%"> </p>

### Efficiency:
<p align="center"> <img src="https://raw.github.com/YingqianWang/DistgSSR/master/Figs/Efficiency-SSR.png" width="50%"> </p>

### Angular Consistency:
<p align="center"> <a href="https://wyqdatabase.s3.us-west-1.amazonaws.com/DistgLF-SpatialSR.mp4"><img src="https://raw.github.com/YingqianWang/DistgSSR/master/Figs/AngCons-SSR.png" width="80%"></a> </p>


## Citiation
**If you find this work helpful, please consider citing:**
```
@Article{DistgLF,
    author    = {Wang, Yingqian and Wang, Longguang and Wu, Gaochang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
    title     = {Disentangling Light Fields for Super-Resolution and Disparity Estimation},
    journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    year      = {2022},   
}
```
<br>

## Contact
**Welcome to raise issues or email to [yingqian.wang@outlook.com](yingqian.wang@outlook.com) for any question regarding this work.**
