# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture using tensorflow. This project can be done in fewer lines of code using Keras but the focus is to get familiar with the low level tensorflow API.
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test_images2/1.jpg
[image2]: ./test_images2/2.jpg
[image3]: ./test_images2/3.jpg
[image4]: ./test_images2/4.jpg
[image5]: ./test_images2/5.jpg
[image6]: ./test_images2/6.jpg
[image7]: ./test_images2/7.jpg
[image8]: ./test_images2/8.jpg


Here is a link to my [project code](https://github.com/shlok-agarwal/Traffic-Sign-Classifier/blob/main/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

Here is some information about the dataset:   
  
Number of training examples = 34799    
Number of testing examples = 12630      
Image data shape = (32, 32, 3)     
Number of classes = 43       
     
After data augmentations,         
    
New Training Set Examples:   69598 samples   
    
### Design and Test a Model Architecture


I normalized the image data because it improves the speed and quality of the optimzed solution.

![image](https://user-images.githubusercontent.com/22652444/118424371-1d20c000-b695-11eb-9348-b8bd9048f8e1.png)


I decided to generate additional data because I found that the training and validation loss were lower after adding more images

To add more data to the the data set, I used rotated and scaled the images in a random manner in an attempt to add more uncertain labelled data.

After some experimentation, I found this network architecture to give me the best results:

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5, 6 layers     	| 1x1 stride, same padding, outputs 32x32x6 	|
| RELU					|			Activation									|
| Flatten output (f1)					|			Flatten the convolution output									|
| Convolution 5x5, 16 layers     	| 1x1 stride, same padding, outputs 32x32x16 	|
| Max pooling	      	| 2x2 stride,  outputs 16x16x6 				|
| RELU					|			Activation									|
| Flatten output (f2)					|			Flatten the convolution output									|
| Add (f1) and (f2)					|			Add the flattened output from both convolution layers									|
| Fully connected		| Output of space 400        									|
| RELU					|			Activation									|
| Dropout					|												|
| Fully connected		| Output of space 240        									|
| RELU					|			Activation									|
| Dropout					|												|
| Fully connected		| Output of space 84        									|
| RELU					|			Activation									|
| Fully connected		| Output of space 43        									|
| Softmax				| Activation        									|

 


#### Training parameters

```
params['dropout'] = 0.5
params['epochs'] = 100
params['batch_size'] = 128
params['learn_rate'] = 0.001
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Validation Accuracy = 0.931
* Test Accuracy = 0.917

2 approaches were explored
* Standard LeNet: The first approach was trying a standard LeNet network with the given training samples. Through this process after tweaking the hyperparameters, the validation accuracy was 0.85. After data augmentation, the validation accuracy increased to 0.88.
* Modified LeNet: Using the idea of skip layers seen in ResNet, the output of the first and second convolution layers were concatenated and passed to the first fully connected layer. This improved the validation accuracy significantly to the 0.93.
   
To compensate for larger number of epochs, two dropout layers were added. 

### Test a Model on New Images

To test of new data, here are 8 German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10]

When predicting images and gathering the label from the dataset( right), we can see the model does a decent job in prediction. It got 4/8 correct and the others were close predictions.

![image](https://user-images.githubusercontent.com/22652444/118428331-39285f80-b69d-11eb-8a96-6746ee6131d9.png)


| Image			        |     Prediction	        					| Softmax Prob	        					| 
|:---------------------:|:---------------------------------------------:|:---------------------------------------------:| 
| General caution      		| Road work   									| 1 |
| No passing     			| No passing 										| 1 |
| Speed limit 20 km/h					| Speed limit 30 km/h											| 0.92 |
| Traffic light ahead (missing from training dataset) 	      		| Road Work					 				| 1 |
| Stop		| Stop      							| 1 |
| Priority Road		| Priority Road      							| 1 |
| Yield		| Yield      							| 1 |
| Turn right ahead		| Keep left      							| 0.99 | 


The model was able to correctly guess 4 of the 8 traffic signs, which gives an accuracy of 50%. Two other images were predicted almost correctly. This indicates that the network could possibly benefit from more data and trying deeper neural networks.

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Looking at the feature map, the first convolution layer learns the shape of the traffic sign. You can see the border of the sign is recoginized in each of the six layers. Although the output of the second convolution layer is not clear, intuitively the second layer learns the finer details of the signs. I think about it in this way because when architecting the network, I found that adding a second layer improves the validation accuracy.


