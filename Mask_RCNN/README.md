Mask-RCNN
====
这是一个使用Mask-RCNN实现的demo，当mode="inference"时，会调用预训练好的mask_rcnn_coco.h5模型进行预测，但是要想对算法的实现过程有更详细的认识，还是应该将mode="training",重新在MS COCO数据集上训练，进行debug调试，或者参考下面这篇文章，学习从标注图像开始完全训练自己的数据集。
https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46
pic_demo.py文件是对图片进行目标检测和分割。
我还写了一个video_demo.py的文件，这个文件可以实现对摄像头的调用，当然你也可以稍作修改，使用它来处理你想要检测的视频。
