## Project: Follow Me (RoboND-DeepLearning-Project)

---

**Steps to complete the project:**

1. Clone the project repo here
2. Fill out the TODO's in the project code as mentioned here
3. Optimize your network and hyper-parameters.
4. Train your network and achieve an accuracy of 40% (0.40) using the Intersection over Union IoU metric which is final_grade_score at the bottom of your notebook.
5. Make a brief writeup report summarizing why you made the choices you did in building the network.

[//]: # (Image References)

[imaged1]: ./image/fcn_diagram.png "FCN Diagram"
[imagef1]: ./image/following1.png "Image following the target 1"
[imagef2]: ./image/following2.png "Image following the target 2"
[imagef3]: ./image/following3.png "Image following the target 3"
[imagewo1]: ./image/without1.png "Image without the target 1"
[imagewo2]: ./image/without2.png "Image without the target 2"
[imagewo3]: ./image/without3.png "Image without the target 3"
[imagew1]: ./image/with1.png "Image with the target 1"
[imagew2]: ./image/with2.png "Image with the target 2"
[imagew3]: ./image/with3.png "Image with the target 3"
[imageg1]: ./image/loss_graph.png "Loss Graph 210 Epochs"
[imageg2]: ./image/IoU_graph.png "IoU Graph 210 Epochs"
[imageg3]: ./image/loss_graph2.png "Loss Graph 570 Epochs"
[imageg4]: ./image/IoU_graph2.png "IoU Graph 570 Epochs"
[imagee1]: ./image/encoding.png "Encoding Operation"
[imagee2]: ./image/decoding.png "Decoding Operation"

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### FCN Model (model_training.ipynb)
#### Visualization


![alt text][imaged1]


#### Explanation
The conventional CNNs are suitable for classification of a given image. However, when the task is to detect where is the target object in a given image, CNNs do not have this information since they lose the spatial information during classification.

The FCNs main difference with CNNs is to keep the spatial information, and to classify each pixel of the image. In order to achieve this, there are three main changes applied to the CNN architecture.

1. The fully connected layers are replaced with 1x1 Convolutional Layers
2. Up sampling through the use of transposed convolutional layers.
3. Skip connections.

##### 1. 1x1 Convolutions
After decoding the image, the fully connected layer comes in CNNs. The fully connected layer has a 4D input and 2D output. Replacing it with a 1 by 1 convolutional layer changes the output to remain 4D. This enables to keep the spatial information. For 1 by 1 convolutional layers, the shape of the output is the same as the shape of the input except the depth, the number of filters. The batch size, the height and the width of the input is preserved.

1 by 1 convolution is actually is a convolution with
* 1x1 filter size, kernel_size=1
* Strides=1
* Padding=SAME
* The number of filters is the variable to set.

When classifying an image fully connected layers are more suitable, because they can easily connected to the following layers and eventually a 1D output. However, when classifying each pixel, 1x1 convolutional layers are an appropriate choice. Since they can keep the spatial information.


##### 2. Up Sampling (Transposed Convolution)
The up sampling operation is basically a reverse convolution. Therefore, the differentiability property is preserved. The up-sampling layers are used to decode the encoded image layer by layer.

##### 3. Skip Connections
Use information from multiple resolution scales. This result in the network to be able to make more precise segmentation decisions. In addition, the lost information during encoding can be retained back.
In the decoder layers, the output of the previous layer and the corresponding encoding layer or the input layer are concatenated. During this operation, the connections are skipped and the two nonconsecutive layers are connected.


##### 4. Encoder
Series of convolutional layers. The goal of the encoder is to extract features from the image. These features provide information for classification. They contain the pixel wise information and the relations between neighbor pixels.

The purpose of the encoder layers is to represent the data with less information as in the compression. However, the encoders in neural networks cannot compress well enough to keep all the details of the image.

Encoding is looking closely at the picture. This result in narrowing down the scope and the loss of the bigger picture. Once the encoder lost the details of the picture, decoding the encoded information cannot retrieve the original image.

The below image from lesson shows an example of the encoding operation.


![alt text][imagee1]


##### 5. Decoder
The decoder up scales the output of the encoder. The resulting image has the same size with the input image. Since the encoded image provides information for classification, the decoder layers keep this information and the resulting image has classified pixels.

During the up-scaling process, decoding the output is not sufficient, because the encoder had lost the details of the original image. Decoder layers can recover this details by using the input to the corresponding encoder layer. "Skip Connections" section explains this operation. The effect of "Skip Connections" is given in the following image.  It helps to recover the image in high resolution.


![alt text][imagee2]


#### Encoder (encoder_block)
There are 2 encoder layers. First one has 32 filters and the second has 64 filters. They consist of separable convolution layers.
##### Separable Convolution (separable_conv2d_batchnorm)
Contains two operations SeparableConv2DKeras, and BatchNormalization.
SeparableConv2DKeras function uses the following inputs:
* provided filter size (32, 64)
* kernel_size=3
* strides = 2
* The padding is same
* The activation is relu

Then BatchNormalization was applied to the output of SeparableConv2DKeras function.


#### 1x1 Convolution
##### Conv2d_batchnorm (conv2d_batchnorm)
Applied regular 2D convolution with
* Filter size:128
* kernel_size:1
* strides:1
* The padding is same
* The activation is relu

#### Decoder (decoder_block)
Has 2 decoder layers.
* Decoder1, concatenates output of 1x1 Convolution layer and the output of the first encoder layer. Filter size is 64.
* Decoder2, concatenates output of first decoder layer and the input. Filter size is 32.
##### Up-sample (bilinear_upsample)
Used bilinear_upsample function to the small_ip layer.

##### Concatenate (layers.concatenate)
Then the output of the up-sample operation concatenated with the large_ip layer.

##### Separable Convolution (separable_conv2d_batchnorm)
Applied to the concatenated layer.

### Hyperparameters
* learning_rate = 0.01
* batch_size = 16
* num_epochs = 10
* num_epoch_loop = 20
* steps_per_epoch = 200
* validation_steps = 50
* workers = 8

Learning rate was selected as 0.01. Smaller rates such as 0.001 performed worse because the model learned slowly.

Batch size set to 16. Since the memory of the used GPU was 2 GB, batch size can not be too high. 4, 8 caused the model to learn slowly.

num_epochs is the number of epochs to test the model. num_epoch_loop is the number to define the number of the loops. Each loop has num_epochs epochs. The initial loop was not included. Therefore, there were 200 + 10 = 210 epochs.

As the second step, the number of epochs was increased to 570.

steps_per_epoch and validation steps were leaved as they were.

### Model 2 (model_training_model2.ipynb)
#### Layers
Has 4 encoder layers with filter sizes 8, 16, 32, and 64.
1x1 Convolution layer is the same with the first model.
Has 4 decoder layers, with filter sizes 64, 32, 16, and 8. The concatenated layers are as follows:
* decoder1 conv1_1, encoder3
* decoder2 decoder1, encoder2
* decoder3 decoder2, encoder1
* decoder4 decoder3, inputs

#### Hyperparameters
learning_rate = 0.01
batch_size = 64
num_epochs = 10
num_epoch_loop = 100
steps_per_epoch = 200
validation_steps = 50
workers = 8

#### Results of Model 2
It took much more time to train the second model. 65 epochs in 16 hours on a E3 8cores 3.4Ghz processor. The best final score value obtained was 0.388289145337 at epoch 20.

### Results
The first model trained for 210 epochs. For every 10 epochs, the results were tested. The best final score value obtained was 0.41578874130264754 at epoch 170. The model was trained on 650M 2GB GPU for 10 hours.

After training to epoch 570, the best final score value obtained was 0.4355026050577782 at epoch 260.

#### Loss Graph
The loss graph on train data and validation data for the 210 epochs are given in the following graph:


![alt text][imageg1]


The loss graph on train data and validation data for the 570 epochs are given in the following graph:


![alt text][imageg3]


#### IoU Graph
The IoU metrics were calculated for every ten epochs. The graph of the metrics for 210 epochs are given in the following graph.


![alt text][imageg2]


The IoU metrics were calculated for every ten epochs. The graph of the metrics for 570 epochs are given in the following graph.


![alt text][imageg4]


#### Following the Target
The original image, ground truth image and the processed image for three samples are as follows:


![alt text][imagef1]


![alt text][imagef2]


![alt text][imagef3]


#### At Patrol without the Target
The original image, ground truth image and the processed image for three samples are as follows:


![alt text][imagewo1]


![alt text][imagewo2]


![alt text][imagewo3]


#### At Patrol with the Target
The original image, ground truth image and the processed image for three samples are as follows:


![alt text][imagew1]


![alt text][imagew2]


![alt text][imagew3]




### Future Enhancements
The graphs from the previous section shows that there is a room for improvement if the model was trained for longer epochs.
Similarly, the batch size can be increased. After the second step, training for 570 epochs, increasing the number of epochs does not seem to improve the results.


A better learning rate may exists. The time limit and computation power limit prevented training with different learning rates such as 0.0075 and 0.0125.


Filter size can be increased. More layers can be added. Again the limitations prevent from experimenting these options. In addition, the 4 layer model did not performed well. Possible reasons for this are small filter sizes, and small number of epochs.


In order to make this model work for following another object, it should be trained with different data. In which, the target object was specified in the mask files.