## Project: Follow Me (RoboND-DeepLearning-Project)

---

**Steps to complete the project:**

1. Clone the project repo here
2. Fill out the TODO's in the project code as mentioned here
3. Optimize your network and hyper-parameters.
4. Train your network and achieve an accuracy of 40% (0.40) using the Intersection over Union IoU metric which is final_grade_score at the bottom of your notebook.
5. Make a brief writeup report summarizing why you made the choices you did in building the network.

[//]: # (Image References)

[imagef1]: ./image/following1.png "Image following the target 1"
[imagef2]: ./image/following2.png "Image following the target 2"
[imagef3]: ./image/following3.png "Image following the target 3"
[imagewo1]: ./image/without1.png "Image without the target 1"
[imagewo2]: ./image/without2.png "Image without the target 2"
[imagewo3]: ./image/without3.png "Image without the target 3"
[imagew1]: ./image/with1.png "Image with the target 1"
[imagew2]: ./image/with2.png "Image with the target 2"
[imagew3]: ./image/with3.png "Image with the target 3"
[imageg1]: ./image/loss_graph.png "Loss Graph"
[imageg2]: ./image/IoU_graph.png "IoU Graph"

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### FCN Model (model_training.ipynb)
#### Encoder (encoder_block)
There are 2 encoder layers. First one has 32 filters and the second has 64 filters. They consist of separable convolution layers.
##### Separable Convoulution (separable_conv2d_batchnorm)
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
##### Upsample (bilinear_upsample)
Used bilinear_upsample function to tha small_ip layer.

##### Concatenate (layers.concatenate)
Then the output of the upsample operation concatenated with the large_ip layer.

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

steps_per_epoch and validation steps were leaved as they were.

### Model 2 (model_training_model2.ipynb)
#### Layers
Has 4 encoder layers with filter sizes 8, 16, 32, and 64.
1x1 Convolution layer is the same with the first model.
Has 4 decoler layers, with filter sizes 64, 32, 16, and 8. The concatenated layers are as follows:
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

#### Loss Graph
The loss graph on train data and validation data for the 210 epochs are given in the following graph:


![alt text][imageg1]



#### IoU Graph
The IoU metrics were calculated for every ten epochs. And the graph of the metrics are given in the following graph.


![alt text][imageg2]


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
Similarly, the batch size can be increased.


A better learning rate may exists. The time limit and computation power limit prevented training with different learning rates such as 0.0075 and 0.0125.


Filter size can be increased. More layers can be added. Again the limitations prevent from experimenting these options. In addition, the 4 layer model did not performed well. Possible reasons for this are small filter sizes, and small number of epochs.


In order to make this model work for following another object, it should be trained with different data. In which, the target object was specified in the mask files.