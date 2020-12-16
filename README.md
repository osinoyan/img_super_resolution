# Image Super Resolution

Code for 3x image super-resolution adapting ESRGNN.
 
 
## Environment
- Ubuntu 16.04 LTS

## Outline
1. [Installation](#Installation)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Training](#Training)


## Installation

### clone necessary repo

First, clone our [img_super_resolution repo](https://github.com/osinoyan/img_super_resolution)
Then, clone the [idealo/image-super-resolution repo](https://github.com/idealo/image-super-resolution) inside *img_super_resolution/*


```
$ git clone https://github.com/osinoyan/img_super_resolution
$ cd img_super_resolution
$ https://github.com/idealo/image-super-resolution
```

### environment installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```bash
$ conda create -n cv_img_sr python=3.6
$ conda activate cv_img_sr
$ pip install -r requirements.txt
```
:::danger
Make sure you have installed the correct version of packages:
`tensorflow=2.3.1`
:::

## Dataset Preparation

### HW4 Dataset
You can download the HW4 dataset from [this google drive link](https://drive.google.com/drive/folders/1H-sIY7zj42Fex1ZjxxSC3PV1pK4Mij6x).
Unzip *training_hr_images.zip* and locate the training images like below:
```
    training/
    +- 2092.png
    +- 8049.png
    +- ...
```
There should be 291 images in directory *training/*.

### Downsample 3x
For training for 3x super-resolution network, you have to downsample the original images. The downsampled images are used as relative low-resolution training data.
There are many useful public tools for this task, you can try this  [img-downsampler repo](https://github.com/wqi/img-downsampler). It is super simple, just run the command below:
```bash
$ mkdir training_lr
$ python img-downsampler/downsample.py training/ training_lr/
```

:::danger
Sadly, for training images with size(height/width) not multply of 3 can cause problem of inconsistency inputs to the network, it is necessory to modify the *image-super-resolution/ISR/utils/datahandler.py*. The modified file was under the root directory of our repo *img_super_resolution/datahandler.py*, you have to copy it and replace *image-super-resolution/ISR/utils/datahandler.py* manually.
```
$ cd img_super_resolution/
$ cp -f datahandler.py image-super-resolution/ISR/utils/
```
Thanks to this issue comment [#54](https://github.com/idealo/image-super-resolution/issues/54#issuecomment-519464785), the modification is mostly based on code from this comment.
:::
## Training
We briefly provide the instructions to train and test the model on HW4 dataset.
For more information, see [idealo/image-super-resolution repo](https://github.com/idealo/image-super-resolution).

### train models
To train ESRGNN(using RDNN) model, run the following commands.
```
python train_custom.py --epoch 1000
```
### test and make submission files
After training, try using the trained model to test the prepared testing data.
```
python test_custom.py --weights path/to/your/weights.hdf5 --output path/to/output/dir
```
