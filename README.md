## simple pytorch segmentation tutorial with apex mixed precision and distributed training
https://github.com/dtrimina/pytorch_segmentation_tutorial
#### version  
- python 3.7.5
- torch 1.3.0
- torchvision 0.4.1
- apex 0.1

#### database  

- [x] [CamVid](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/CamVid) : It is a automotive dataset which contains **367** training, **101** validation, and **233** testing images. There are **eleven** different classes such as building, tree, sky, car, road, etc. while the twelfth class contains unlabeled data, which we ignore during training. The original frame resolution for this dataset is 960 × 720. We use **480 x 360** images in [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid).    
CamVid：这是一个汽车数据集，包含367张训练图像、101张验证图像和233张测试图像。有11个不同的类别，如建筑、树木、天空、汽车、道路等，而第12个类别包含未标注的数据，我们在训练时忽略了这些数据。这个数据集的原始帧分辨率为960×720。我们在SegNet-Tutorial中使用了480×360的图像.    
- [x] [SUNRGBD](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/SUNRGBD) : The SUN RGB-D dataset has **10,335** densely annotated RGB-D images taken from 20 different scenes. It also include all images data from NYU Depth v2 dataset, and selected images data from Berkeley B3DO and SUN3D dataset. Each pixel in the RGB-D images is assigned a semantic label in one of the **37 classes** or the ‘unknown’ class. It has **5285 training/validation** instances and **5050 testing** instances. If you want to train your segmentation model with **depth** information.    
SUNRGBD：SUN RGB-D数据集有10,335张来自20个不同场景的RGB-D密集注释图像。它还包括纽约大学Depth v2数据集中的所有图像数据，以及伯克利B3DO和SUN3D数据集中的部分图像数据。RGB-D图像中的每个像素都被分配了37个类或 "未知 "类中的一个语义标签。它有5285个训练/验证实例和505050个测试实例。如果你想用深度信息训练你的分割模型。   
- [x] [Cityscapes](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/Cityscapes) : This dataset consists of 5000 fineannotated images, out of which **2975** are available for training, **500** for validation, and the remaining **1525** have been selected as test set. It contains **19** classes and an unlabel class. The raw resolution is 2048*1024.    
 城市景观:该数据集由5000张精细注释的图像组成，其中2975张可用于训练，500张用于验证，其余1525张被选为测试集。它包含了19个类和一个非标签类。原始分辨率为2048*1024。    
- [x] [ADE20k](https://github.com/dtrimina/pytorch_segmentation_tutorial/tree/master/database/ADEChallengeData2016): ADE20K dataset is a challenge scence parsing dataset providing **150** class dense labels, which consists of **20K/2K/3K** images for training/validation/test.    
ADE20k：ADE20K数据集是一个挑战scence解析数据集，提供150个类密集标签，由20K/2K/2K/3K图像组成，用于训练/验证/测试。

#### models

- training on 4 taitanxp GPUs  
- miou on validation set, use model trained after the last epoch to evaluate  
- msc miou means using multi-scale images (0.5 0.75 1.0 1.25 1.5 1.75) and their flip version for evaluation  

model | miou | msc miou | params size(fp32) |  
:-: | :-: | :-: | :-:   
[unet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | - | - | 51.14MB |  
[segnet](https://blog.dtrimina.cn/Segmentation/segmentation-2/) | - | - | 117MB |   
[LinkNet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | - | - | 44.07MB |   
[FC-DenseNet103](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | - | - | 35.58MB |  
[ENet](https://blog.dtrimina.cn/Segmentation/segmentation-3/) | - | - | 1.34MB |  
[DRN-C-26](https://blog.dtrimina.cn/Segmentation/segmentation-4/) | 0.671 | 0.702 | 78.67MB |   
Deeplabv3+_resnet50 | 0.746 | 0.765 | 153.94MB |  
Deeplabv3+_resnet101 | - | - | 226.39MB |  

#### default training config  

- data augmentation: colorjit + randomhflip + randomscale + randomcrop  
- input image normalize: ToTensor + Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- loss function: CrossEntropyLoss + class_weight  
- 150 epoch, Adam optimizer, initial_lr=5e-4  

#### train and evaluate

```
# train on 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/cityscape_drn_c_26.json

# evaluate
python evaluate.py --logdir [run logdir] [-s] 

# Moreover, you can add [your configs].json in run_tasks.sh
sh run_tasks.sh
```

#### reference
- https://github.com/meetshah1995/pytorch-semseg  
- https://nvidia.github.io/apex/#  
- https://github.com/nvidia/apex  
