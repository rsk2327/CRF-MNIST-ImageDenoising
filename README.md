# CRF-Digit-Image-Denoising
Denoising of noisy MNIST dataset images using Conditional Random Fields

Noisy images of a particular digit are denoised using a trained CRF model. The CRF model is implemented using the [PyStruct](http://pystruct.github.io/) library

The problem statement requires the denoising of images of a particular digit using CRF's.

## Models

* **CRFBasic.py** Basic model with unary potentials defined by node pixel value.
* **CRFBasic_Iterative.py** Basic model with iterative saving of models at different stages of model training.
* **CRFEdgeFeatures.py** Model with unary potentials defined by node pixel value and pair-wise potential defined pixel values of connecting nodes.
* **CRFNeighborhood.py** Model with unary potentials defined by pixel values of node and neighboring nodes.
* **CRFNeighborEdgeFeatures** Combination of CRFEdgeFeatures and CRFNeighborhood.


## Results
![](https://github.com/rsk2327/CRF-MNIST-ImageDenoising/blob/master/Results/2.png)
![](https://github.com/rsk2327/CRF-MNIST-ImageDenoising/blob/master/Results/3.png)
![](https://github.com/rsk2327/CRF-MNIST-ImageDenoising/blob/master/Results/4.png)
![](https://github.com/rsk2327/CRF-MNIST-ImageDenoising/blob/master/Results/5.png)
