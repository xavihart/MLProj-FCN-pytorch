# MLProj
 This is a semi-personal implemetation of the FCN(fully convolutional network) trained in Pascal-voc2012 ,using the conv structure(vgg16)

 *paper:
 >[Fully Convolutional Networks for Semantic Segmentation CVPR2015](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
 
 >It took about 4.8 hours to train FCN8s which are composed of a pretrained VGG16 and the upsample net of FCN.
 >It should be noted that I use the CrossEntropy2d as the loss function as [fcn_pytorch](https://github.com/xavihart/fcn_pytorch), it may not be the best choice though.
 
 
 ![FCN8s training curves](https://raw.githubusercontent.com/xavihart/MLProj-FCN-pytorch/master/curve.jpg)
 
 
 
 >Some examples are as follows
 ![More examples](https://github.com/xavihart/MLProj-FCN-pytorch/blob/master/MLProj2/exp.PNG)
 
 
 >Not quite good. The model seems still has problems.
 >the tempt acc can be referred to in the open.PDF
