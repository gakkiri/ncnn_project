# ncnn project
This repo is dedicated to the my ncnn's toy projects.

## content
1. [Gated-SCNN](#Gated-SCNN)
2. [CenterNet](#CenterNet)
3. [UNet-VB](#UNet-VB)

<div id="Gated-SCNN"></div>

## [Gated-SCNN](https://github.com/nv-tlabs/gscnn)
The heaviest model `best_cityscapes_checkpoint.pth` are used.  
The input size is fixed at `h, w = 512, 1024`.  
The `.param` and `.bin` can be taken from [google drive](https://drive.google.com/drive/folders/1SUPz7yl5l2mYTYgZR9sYLkZci5MB1PHv?usp=sharing)

#### Take a random sample from a Beijing street scene
![e1](https://github.com/gakkiri/ncnn_project/blob/main/gscnn/asserts/test.jpg?raw=true?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![e1](https://github.com/gakkiri/ncnn_project/blob/main/gscnn/asserts/vis.png?raw=true?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)

<div id="CenterNet"></div>

## [CenterNet](https://github.com/xingyizhou/CenterNet)
The model `resnet50_coco_wo_deconv.pth` are used.  
The input size is fixed at `h, w = 448, 672`.  
The ncnn weight [google drive](https://drive.google.com/drive/folders/1CKPbjzmL2GWwlEgicdVrCgSAz0SBhnR2?usp=sharing)  
![e1](https://github.com/gakkiri/ncnn_project/blob/main/centernet/asserts/test.jpg?raw=true?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![e1](https://github.com/gakkiri/ncnn_project/blob/main/centernet/asserts/det.png?raw=true?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)

<div id="UNet-VB"></div>

## [UNet-VB](https://github.com/xingyizhou/CenterNet)

The model is just a very common [UNet](https://arxiv.org/abs/1505.04597) with ResNet  
The input size is fixed at `h, w = 256, 256`. In order to achieve a better effect on high resolution pictures, the grid cutting trick was used.  
The ncnn weight [google drive](https://drive.google.com/drive/folders/10NicU1cK7Yuj2LFiEQblxeDiO5Tb7zsR?usp=sharing)  

![e1](https://github.com/gakkiri/ncnn_project/blob/main/unet_vb/asserts/test.jpg?raw=true?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
![e1](https://github.com/gakkiri/ncnn_project/blob/main/unet_vb/asserts/vis.jpg?raw=true?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzNDk3ODQ1,size_16,color_FFFFFF,t_70)
