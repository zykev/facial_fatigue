# Machine vision to human facial fatigue with nonlocal 3D attention network

![](https://github.com/zeyuchen-kevin/paper-images/raw/main/cam.jpg)

<center class="half">    
    <img src="https://github.com/zeyuchen-kevin/paper-images/raw/main/1.gif" width="200"/><img src="https://github.com/zeyuchen-kevin/paper-images/raw/main/2.gif" width="200"/><img src="https://github.com/zeyuchen-kevin/paper-images/raw/main/3.gif" width="200"/><img src="https://github.com/zeyuchen-kevin/paper-images/raw/main/4.gif" width="200"/>
</center>

**Facial fatigue detection in the wild using 3D-ResNet with non-local attention block.**

## Overview

In order to have a better understanding of people’s mental or energy status, we want to detect cognitive fatigue through some subtle features of people’s faces, so that people can work and live better.

The detection of  video-based fatigue especially mild fatigue in the wild is a challenge in computer vision. We build an audiovisual dataset named LFD dataset (life fatigue dataset) for the convenience of research on facial fatigue detection in real life scenario. Then we combine 3D-ResNet and non-local attention block together to extract spatio-temporal features in both local and long-range dependencies. Fatigue degree can be predicted both in continuous value ranging from 0 to 1 and in categorical label as fatigue and non-fatigue. 

## LFD Dataset

LFD dataset is an audiovisual dataset containing self-captured videos by volunteers when they are in different life scenarios such as awaking, working, off work, breaking, etc. and at different time from morning to midnight.  To some extent, this dataset record people's natural rather than pretended fatigue state as much as possible in the real life situation.

Volunteers record a video with a duration ranging from 5 to 60 seconds to take pictures of their own facial conditions. When shooting, the volunteers face the camera so that the camera captures the full picture of the face. At the same time, they briefly describe their mood and mental state to the camera. . In this way, different volunteers shoot different videos in different life scenarios (such as waking up, going to work, lunch break, off work, preparing to fall asleep). After the video is taken, three people will score cognitive fatigue based on the video and audio information. The score is on a 5-point scale. The higher the score, the more fatigue. Generally, 1, 2, 3, 4, and 5 points are marked. If it is difficult to determine accurately, 0.5 points are allowed, such as 2.5, 3.5, and so on. After that, the average value is taken according to the scores given by the three annotators, and then standardized to the [0,1] interval, which is used as the fatigue value data label.

## Model Architecture

The overall framework can be divided into several parts which consists of face detection and alignment, feature extractor and prediction. In face detection and alignment, MTCNN is used for the extraction of facial region and the five facial landmark is used for alignment. The feature extractor is designed upon the backbone of ResNet-18. The non-local attention block is inserted after the third residual block and the linear layers are modified compared to the standard ResNet-18.  In prediction, two kinds of loss function are designed for producing both continuous and categorical prediction.

![](https://github.com/zeyuchen-kevin/paper-images/raw/main/model_structure.png)

## Results

![](https://github.com/zeyuchen-kevin/paper-images/raw/main/bi_analysis.jpg)

**Curves for 3D-ResNet18 model with non-local attention block using MSE & CE loss.** That shows loss values and accuracies during iterations for training set via orange solid lines and for validation set via blue dashed lines. Average accuracy on validation set surpasses 0.9, which is highlighted by a green horizontal line. 

| Treatment           | Accuracy | Precision | Recall | F1 score |
| ------------------- | -------- | --------- | ------ | -------- |
| Fatigue as Positive | 0.6930   | 0.8287    | 0.8089 | 0.8187   |
| Alert as Positive   | 0.7573   | 0.8540    | 0.8699 | 0.8619   |
| Average             | 0.7252   | 0.8414    | 0.8394 | 0.8404   |

**Performance evaluation for framework using MSE + CE loss.** Fatigue and Alert label are regarded as positive group in true for the calculation of each metric. 

## Training

Training Environments:

*  Ubuntu 18.04 with NVIDIA GeForce RTX 2080Ti (at least 11GB of GPU memory)
* 64-bit Python 3.7 or 3.8 with Pytorch 1.6.0 or 1.7.0
* Necessary packages: opencv, PIL, skimage

---

Run *Train_FrameAttention_tomse2.py* to train the framework.

The setting of training hyper-parameters:

```
'--epochs', default=500, type=int, help='number of total epochs to run'
'--lr', '--learning-rate', default=1e-5, type=float, help='initial learning rate'
'--momentum', default=0.9, type=float, help='momentum'
'--weight-decay', default=1e-4, type=float, help='weight decay'
'--pf', default=200, type=int, help='print frequency'
'-e', default=False, help='evaluate model on validation set'
'--is_test', default=True, help='testing when is traing'
'--is_pretreat', default=True, help='pretreating when is traing'
'--accumulation_step', default=1, type=int, help='gradient accumulation'
'--loss_alpha', default=0.1, type=float, help='adjust loss for crossentrophy'
'--num_classes', default=10, type=int, help='number of categorical classes'
'--first_channel', default=64, type=int, help='number of channel in first convolution layer in resnet'
'--non_local_pos', default=3, type=int, help='the position to add non_local block'
'--batch_size', default=16, type=int, help='batch size'
'--arg_rootTrain', default=None, type=str, help='the path of train sample '
'--arg_rootEval', default=None, type=str, help='the path of eval sample '
```
## Technique Report
The working paper for this project can be obtained at https://arxiv.org/abs/2104.10420.

## Necessary Attachment

We transferred our network for a pre-trained 2D ResNet-18 model using MS-Celeb-1M dataset, which can be found [here.](https://github.com/kaiwang960112/Challenge-condition-FER-dataset) LFD dataset can be obtained after contacting the authors.



