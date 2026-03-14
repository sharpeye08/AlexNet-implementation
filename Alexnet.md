Date: 2026-03-14
Tags: [[deeplearning]] , [[cnn]] , [[Alexnet]] , [[research paper]] 

## Gist from the video
- [[Alexnet]]  is the architecture which led to people thinking about architectures like [[VGG16]] and others.
- [[ReLU]]  activation function was used rather than sigmoid or tanh function because it learns faster than the other 2.
- [[ImageNet]]  dataset was used.
- uses overlapped pooling rather than the conventional pooling methods
- used data augmentation so that the model does not overfit


## Architecture

![[Pasted image 20260314005110.png]]
- it contains 8 learned layers - 5 convolutional and 3 fully connected layers
#### ReLU Normality
- Deep convolutional neural networks with [[ReLU]] train several times faster than their tanh equivalents. Faster learning has a great influence on the performance of large models trained on large datasets.
- ReLUs have the desirable property that they do not require the input normalization to prevent them from saturating. If at least some training examples produce a positive input to a ReLU, learning will happen in that neuron.
#### Overlapped Pooling
- pooling in CNNs summarize the outputs of neighboring groups of neurons in the same kernel map. Traditionally the neighborhoods summarized by pooling units do not overlap.
-  a pooling layer can be thought of as consisting of a grid of pooling units spaced `s`  pixels apart, each summarizing a neighborhood of size `z × z `centered at the location of the pooling unit. 
	- if we set `s = z`, we obtain traditional local pooling commonly used in CNNs.
	- if we set `s < z` , we obtain **overlapping pooling**. 
- **overlapping pooling** is used throughout our network, with `s = 2` and `z = 3`. 
- we generally observe during training that models with overlapping pooling find it slightly difficult to overfit.
#### Overall Architecture
- the network contains 8 layers with weights; the first 5 are convolutional and the rest 3 are fully connected.
- the output of the last fully connected layer is fed to a 1000 way softmax which produces a distribution over the 1000 class labels.
- the kernels of the second, fourth and fifth convolutional layers are only connected to those kernel maps in the previous layer that resides in the same GPU.
- the neurons in the fully connected layers are connected to all neurons in the previous layer. 
- response normalization layers follow the first and second convolutional layers.
- max pooling layers follow both response normalization layers as well as the fifth convolutional layer.
- the ReLU non linearity is applied to the output of every convolutional and fully connected layer.

1. The first convolutional layer filters the `224 * 224 * 3` input image with 96 kernels of size `11 * 11 * 3` with a stride of 4 pixels. 
2. the second convolutional layer takes as input the response normalized and pooled output of the first convolutional layer and filters it with `256 kernels` of size `5 * 5 * 48`.
3. the third, fourth and fifth convolutional layers are connected to one another without any intervening pooling or normalization layers.
4. the third convolutional layer has 384 kernels of size `3 * 3 * 256` connected to the normalized, pooled outputs of the second convolutional layer.
5. the fourth convolutional layer has 384 kernels of size `3 * 3 * 192`, and the fifth convolutional layer has 256 kernels of size `3 * 3 * 192`. 
6. the fully connected layers have 4096 neurons each.

## Reducing Overfitting

#### 1. Data Augmentation
- the easiest and the most common method to reduce overfitting on image data is to artificially enlarge the dataset using label preserving transformations (eg : [25,4,5]).
- in the first form we extract random `224 * 224` patches (and their horizontal reflections) from the `256 * 256` images and training our network on these extracted patches.
- this increases the size of our training set by factor of 2048, though the resulting training examples are highly inter dependent. without this scheme our network suffers from substantial overfitting which would have forced us to use much smaller networks.
- at test time, the network makes a prediction by extracting five `224 * 224` patches (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all), and averaging the predictions made by the network's softmax layer on the ten patches.
- second form of data augmentation consists of altering the intensities of the RDB channels in training images.
-  PCA is preformed on the set of RGB pixel values throughout the [[ImageNet]] training set.
#### 2. Dropout
- a very efficient version of model combination that only costs about a factor of 2 during training  called **dropout** .
- **dropout** consists of setting to zero the output of each hidden neuron with probability 0.5. The neurons which are dropped out in this way donot contribute to the forward pass and do not participate in backpropagation.
-  At test time, we use all the neurons but multiply their outputs by 0.5, which is a reasonable approximation to taking the geometric mean of the predictive distributions produced by the exponentially-many dropout networks.

## Details of Learning
- we trained our models using stochastic gradient descent with a batch size of 128, momentum of 0.9 and weight decay of 0.0005.
- weight decay is not merely a regularizer but it reduces the model's training error
- We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01. We initialized the neuron biases in the second, fourth, and fifth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1. Neuron biases in the remaining layers were initialized with 0/
- the learning rate was initialized at 0.01 and reduced three times prior to termination.
- network was trained for roughly 90 cycles through the training set of 1.2 million images.
