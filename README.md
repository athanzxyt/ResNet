# ResNet in PyTorch

![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff&style=flat)
![PyTorch Badge](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=fff&style=flat)

This is a re-implementation of the ResNet model into PyTorch. Please note that ResNet architecture for the ImageNet dataset found in the paper was implemented. However, was tested on the CIFAR-10 dataset. 

Training on NVIDIA RTX A6000 yielded the following results:

|               | ResNet18 | ResNet50 |
| -------       | ---------| -------- |
| Top1 Accuracy | 87.5%    | 81.3%    | 
| Top5 Accuracy | 98.4%    | 99.2%    |

As mentioned before, ResNet18 and ResNet50 were originally made for the ImageNet dataset. The authors of the papers did create other ResNet infrastructures for CIFAR, namely Resnet32 and others.

Here is a summary of the files in this respository:

| File Name      | Description |
| ----------- | ----------- |
| resnet.py      | ResNet class       |
| main.py   | Script to run tests on the CIFAR dataset        |
| accXX.png | Accuracy graph of Classification over Iteration. |
| image.png| Sample of CIFAR dataset |


### Citations

[1] He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/cvpr.2016.90 
