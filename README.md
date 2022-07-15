# ResNet in PyTorch

This is a re-implementation of the ResNet model into PyTorch. Please note that ResNet architecture for the ImageNet dataset found in the paper was implemented. However, was tested on the CIFAR-10 dataset. 

Training on NVIDIA RTX A6000 yielded the following results:

| | ResNet18 | ResNet50 |
| ------- | -------- |
| Top1 Accuracy | 87.5% | 81.3% | 
| Top5 Accuracy | 98.4% | 99.2% |

| File Name      | Description |
| ----------- | ----------- |
| resnet.py      | ResNet class       |
| main.py   | Script to run tests on the CIFAR dataset        |
| accXX.png | Accuracy graph of Classification over Iteration. |
| image.png| Sample of CIFAR dataset |
