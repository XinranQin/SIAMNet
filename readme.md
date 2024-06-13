# Siamese Cooperative Learning for Unsupervised Image Reconstruction from Incomplete Measurements
This repository contains the Pytorch codes for paper "# Siamese Cooperative Learning for Unsupervised Image Reconstruction from Incomplete Measurements" (tpami 2024) 
by YuhuiQuan, XinranQin*, TongyaoPang, HuiJi

## Content
* [Overview](#Overview)
* [Dataset](#Dataset)
* [Requirements](#Requirements)
* [Reference](#Reference)

## Overview
Image reconstruction from incomplete measurements is one basic task in imaging. While supervised deep learning has emerged as a powerful tool for image reconstruction in recent years, its applicability is limited by its prerequisite on a large number of latent images for model training. To extend the application of deep learning to the imaging tasks where acquisition of latent images is challenging, this article proposes an unsupervised deep learning method that trains a deep model for image reconstruction with the access limited to measurement data. We develop a Siamese network whose twin sub-networks perform reconstruction cooperatively on a pair of complementary spaces: the null space of the measurement matrix and the range space of its pseudo inverse. The Siamese network is trained by a self-supervised loss with three terms: a data consistency loss over available measurements in the range space, a data consistency loss between intermediate results in the null space, and a mutual consistency loss on the predictions of the twin sub-networks in the full space. The proposed method is applied to four imaging tasks from different applications, and extensive experiments have shown its advantages over existing unsupervised solutions.
![image](https://github.com/XinranQin/SIAMNet/blob/main/images/Model.eps)

## Dataset
Dataset: [Training dataset](https://drive.google.com/drive/folders/1gZbM0DTXHLpf-CsMHg5HrTWWNIcXvX1z?usp=sharing "悬停显示")  
 
## Requirements
RTX3090 Python==3.90 Pytorch>1.8.0+cu101


## References

```
@article{quan2024siamese,
  title={Siamese Cooperative Learning for Unsupervised Image Reconstruction from Incomplete Measurements},
  author={Quan, Yuhui and Qin, Xinran and Pang, Tongyao and Ji, Hui},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```