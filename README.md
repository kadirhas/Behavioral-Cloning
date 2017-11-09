# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/NvidiaNetwork.png "Model Visualization"
[image2]: ./examples/training_performance.png "Training performance"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3 . Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. It also augments the data by flipping image to simulate a mirror track. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of Nvidia's end to end driving model network. It is implemented with keras. The data is normalized and cropped to just use the region of interest in the image. 

#### 2. Attempts to reduce overfitting in the model

With the correct shuffling and augmentation, a dropout layer was not necessary for this problem to overcome overfitting. The model was trained and validated on different data sets to ensure that the model was not overfitting. The model can also drive around a track clockwise or counterclockwise without any problems.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

At first a keyboard is used to get training data. However, since the steering input is similar to bang bang control with keyboard, the trained model was not successful. Because of that data created by udacity is used to train the model. The data is augmented by flipping the image and reversing the rotating angle to simulate a mirrored track. Additionally, beside the center image, left and right camera images are used to train the model to correct its position if it is not on the center of the road. This is achieved by adding a correction value to the steering angle if the image is taken by left or right camera.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to LeNet, which I've modified for the traffic sign project. Since that network was pretty successful at recognizing images without overfitting, I thought it can easily detect roadlines and create a suitable steering input. Even thought the model was successful at training, it could not stay on the road. Since the model had already many dropout layers, I thought it is too large for this kind of problem. Since Nvidia had successful results with its end to end model with real vehicle, I've implemented the same model. Even from the beginning, the validation and training accuracy was similar, which means the model wasn't overfitting, at least with the current data. To fully test if there is an overfitting issue or not, I've tested the model on simulation. The vehicle is able to drive autonomously around the track without leaving the road.

I think the model didn't need that much correction because of two things. First, the data provided by Udacity was really good, which also included correction movements. Second, with the advantage of a large ram on my computer, I didn't use generator and shuffled the data after the augmentation. Thanks to that, in every batch the augmented data was not derived from rest of the batch. 


####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Image   							| 
| Cropping layer    	| 50 pixels from top, 20 pixels from bottom 	|
| Lambda layer			| normalize the image around zero				|
| Convolution 5x5      	| 2x2 stride, depth 24							|
| RELU					|         										|
| Convolution 5x5	    | 2x2 stride, depth 36  						|
| RELU					|         										|
| Convolution 5x5	    | 2x2 stride, depth 48  						|
| RELU					|         										|
| Convolution 3x3	    | 1x1 stride, depth 64  						|
| RELU					|         										|
| Convolution 3x3	    | 1x1 stride, depth 64  						|
| RELU					|         										|
| Fully connected		| 100 Nodes 									|
| Fully connected		| 50 Nodes 										|
| Fully connected		| 10 Nodes 										|
| Output				| 1 Node	 									|


Here is a visualization of the architecture, taken from Nvidia's paper.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I've tried to capture a training data with a keyboard. This approach has failed because I could not provide a smooth input for the vehicle, which caused the model to steer less than usual.

Because of this I've used the data provided by Udacity, which has a smooth steering input.

To augment the data, I've used two methods. First is using right and left camera images to train the model. In this approach, I've added small correction value to the corresponding steering. This way if the vehicle is not in the center of the road, the model can drive the vehicle back in the center of the road. The second method is flipping the images and multiplying the steering angles with -1. This way I obtained a mirrored track. 

After the collection process, I've shuffled the data, and seperated them as validation and test data. I had 38572 samples for training. The model had two preprocessing layers. One of them crops the image to get only region of interest (in this case the road). The other one normalizes the RGB values in a way that their mean is zero.

In my tests, I've seen that increasing epochs causes to improvement in training but also causes overfitting. Because of that I've used only 2 Epochs to get the final model.

You can see the training and validation accuracy below for each epoch:
![alt text][image2]

#### 4. Conclusion & Improvement suggestions

As you can see on run1.mp4 and run2.mp4 the model can run pretty good around the track. On run1, the vehicle speed is adjusted as 8 in drive.py. On run2, the vehicle speed is adjusted as 20. The model can't drive faster than this, because it starts to oscillate on the road. This problem can be solved by decreasing the correction values on the augmented data generation. 

To overcome of overfitting, additional dropout layers can be implemented, and more data can be obtained from different tracks to generalize the model. Generating fake shadows, brightness and gama modifications, applying noise to the collected data would provide more generalized data.