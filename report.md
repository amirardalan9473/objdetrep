#####################################################################################################
   
 # Object Detection For CFIA
 
 There are numerous ways computer vision can help out the CFIA to carry its task. To give some details, we must first state and describe what computer vision tasks we are referring to.
 Previously we spoke about computer vision as a field, here we will talk about some tasks that can assist us in doing our job at the CFIA.
 
 For example, let's consider the task of counting the cows in an image or video.
 There are three computer vision tasks of interest:

- Object Recognition
- Object Detection
- Object Segmentation

Object recognition is the technique of identifying the object present in images and videos. An example would be the algorithm or system receiving an image containing a cat and/or a dog. After processing the image, the system would return the label or labels corresponding to the objects it detects in a photo.

A visual example is illustrated below showing object recognition in two systems. A classical machine learning system and a deep learning system. It is worth noting the deep learning systems almost always beat the classical ones when it comes to computer vision tasks. There was more talk about this in previous posts.


![N|Object Recognition](https://github.com/amirardalan9473/objdetrep/blob/main/mlvsdp.jpg?raw=true)


Now that we briefly talked about Object Recognition, let's move onto the next task. Object Detection. Object Detection algorithms try to go one step beyond and not only recognize the objects in the photo, but they also localize and identify their position in the image by drawing a bounding box around it.

An example illustrating the differences in the output of such a system is shown below

[![N|Object Recognition](https://miro.medium.com/max/1600/0*ly6JczJzzpIxQl1S) ]()

On the left side, the system returns a single label, whereas on the right it returns a picture identifying different objects present in the photo.

Now the last task, Object Segmentation; is defined as:
>**The process of partitioning a digital image into multiple segments (sets of pixels, also known as image objects)**

An example illustrating the differences is shown below:

[![N|Object Recognition](https://lh3.googleusercontent.com/proxy/i7xOOcJI4vIH0R2UIwhroXvTIyfJmuOQltkHDtgU_bWsKjTEjg9nyTCpUGt6V2nIvS-bIQLlZ-r-79VLnVt80YK0obwO76S_g3G05gqaEEgveK40R5kuR6VEgsOq1bcjZ3UgsqAErZb2PGzzvtWi4UQcyA37WSZ3IluU7x8Y6fgVVlWy439ExFLaWA)]()

Each one of these tasks has its challenges. Each requires its training and testing dataset even if it's trained on the same pictures. For example in the Object Detection task, the training dataset requires the pictures to have bounding boxes and labels, as opposed to classification where only the label is required to accompany the pictures.

For now, we consider the task of counting cows in an image or video. For this task, an object detection system would be best suited, where we detect all the cows in a picture and at the end take the count. 

Here we will go over some systems with their corresponding pros and cons.

The first system we introduce is **YOLOV4**

#  **YOLOv4**: You only look once

YOLO is a family of one-stage object detectors that are fast and accurate.

The general architecture of object detection systems consists of 3 components:
* Backbone
* Neck
* Head

An image illustrating the overall architecture is shown below.

[![N|Object Recognition](https://miro.medium.com/max/660/1*jLUJU34dSbrRWdspJZbLXA.png
)]()

The backbone of the model can be a typical Neural Net such as DenseNet, ResNet, VGG, or any other flavor of convolutional neural networks for the task of image classification. These networks are typically trained on ImageNet and then fine-tuned on the custom datasets for object detection. Convolutional Neural Networks extract and produce different features at different levels of the network. At the shallow layer, they typically make geometric features such as straight or tilted lines, and at the deeper layers, they make more semantically meaningful features. 

This is shown below
[![N|Object Recognition](https://image.slidesharecdn.com/nvidiaces2015presentationdeck-150105190022-conversion-gate02/95/visual-computing-the-road-ahead-nvidia-ceo-jenhsun-huang-at-ces-2015-30-638.jpg?cb=1424436369)]()

However, as you can imagine, for object detection, we might have to predict multiple objects in the same photo, as opposed to predicting a single label. Therefore we must make use of the entire receptive field and features at every level.

This is where the next stage comes in, the **Neck**.

Taken directly from the YOLOv4 paper:
>**a neck is composed of several bottom-up paths and several topdown paths. Networks equipped with this mechanism include Feature Pyramid Network (FPN), Path Aggregation Network (PAN), BiFPN, and NAS-FPN**

The FPN example is illustrated below.

[![N|Object Recognition](https://miro.medium.com/max/442/1*BU-7aau6rI_rjVX6NQYtYg.png)]()


The Head part, outputs the coordinates of the bounding boxes for the objects in the image, along with a confidence score for the class.

Here we have a sample photo from the internet, to demonstrate how these algorithms would fare in a task of interest to the CFIA.

[![N|Object Recognition](https://news.psu.edu/sites/default/files/styles/threshold-992/public/PSU-Dairy-Cows.png?itok=UeR4cXQZ)]()

The output produced by the YOLOv4 algorithm is:

[![N|Object Recognition](https://github.com/amirardalan9473/objdetrep/blob/main/detection1.png?raw=true)]()

# EfficientDet: Scalable and Efficient Object Detection

The next model is the *EfficientDet* 
It is a model produced by Google Brain. Here for the Backbone, they use the EfficientNet, and for the neck, they use **BiFPN**; a weighted bi-directional feature pyramid network. 

An overview of the model is shown below.

[![N|Object Recognition](https://github.com/amirardalan9473/objdetrep/blob/main/Screen%20Shot%202021-01-19%20at%2010.56.32%20PM.png?raw=true)]()

Without getting into details, EfficientNets are a class of neural nets where neural architecture search is utilized to design a new baseline network and scale it up to obtain a family of models.

The key contribution of this model is the accuracy and the efficiency of object detection. The result of our cow photos with EfficientDet is shown below.

[![N|Object Recognition](https://github.com/amirardalan9473/objdetrep/blob/main/effdet.jpeg?raw=true)]()


Now the third and what I find personally the most interesting method of object detection is **DETR: End-to-End Object Detection with Transformers**

# DETR: End-to-End Object Detection with Transformers

This is a new method produced by Facebook AI. To give some background, Transformers are a new class of Neural NEtwork architectures originally designed to replace recurrent neural networks. The original application was intended for the Natural Language Processing ***NLP*** tasks. The transformer architecture was developed to capture attention. What do we mean by it?
let's consider an example. Take the following sentence.
> John enjoyed reading his book. Over the weekend with nothing planned and lockdowns in place, **he** decided to finish the **task** at hand.

You see, if you use a normal recurrent neural networks ***RNN***, you would consider one toke/word at a time. And a machine would probably not understand what the word *he* refers to by the time it got to it since John was mentioned at the very beginning of the sentence. 

However, there is an attention mechanism developed by Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio that can address this issue. The original attention mechanism was developed for the task of translating between languages. As you can imagine, Languages are not perfectly aligned, hence you would need some form of attention to align and translate.

The attention mechanism was integrated into the RNN, however, Transformers were developed to completely remove the need for recurrence. 

An image of the original transformer architecture is attached below

[![N|Object Recognition](https://miro.medium.com/max/2880/1*BHzGVskWGS_3jEcYYi6miQ.png)]()

Ever since there has been a boom in transformers research, and they were mainly used in the NLP tasks. However, they have been recently used in other domains of artificial intelligence such as computer vision.

I'm going to give a brief description of how they work. So at the input layer, you have your input, in the case of the NLP, it would be a sequence of words. You also have a positional encoding which indicates the position of each word. 

The same word can have two different meanings/implications based on its position. The model can not differentiate between different tokens because, in a transformer, all the words flow throw the model simultaneously as opposed to the RNNs where tokens are processed one after another. This is where the positional encoding comes in and alleviate the situation.

After some layers of multiheaded attention and Add & Norm layers, the **encoder** part of the transformer outputs a vector which is fed into the decoder. 

The decoder then decides what to output, based on the encoded attention data, and the output target.
So for example, if the input is *I am Nice* and the french target is *Je suis gentil*, at the first step, the decoder receives the attention data and a blank toke, and will output **Je**.
At the second step, the decoder has the attention data, and **Je** as the output which now is shifted and used as an input, and will output **suis**, and so on and on.

Attention driven methods are currently the state of the art in pretty much every domain of AI and benchmark.


Now that we have an idea of what the transformers are, let us dive into the method.


The architecture is laid out below.

[![N|Object Recognition](https://github.com/amirardalan9473/objdetrep/blob/main/Screen%20Shot%202021-01-19%20at%2011.14.21%20PM.png?raw=true
)]()

The transformer here serves as the neck of the model.

You have a convolutional neural network that extracts the features from the images. Then the features are fed into the transformer encoder to get the attention vector, afterward, the decoder makes its predictions which are fed into feed-forward neural networks to get the class and box coordinates.

A few noteworthy things here, in the decoder, the object queries are learned, and the paper goes into details about how they can be interpreted. You can think of each as querying a specific region of the photo.

The loss function here is a bipartite matching loss function. And the way it works is that you specify a parameter **N**, which can be thought of as the maximum number of objects present in a photo. Afterward, the loss function will match *N* predictions to N objects in the photo. If there are fewer objects, Null is used instead.

One of the cons of this model is the N hyperparameter, which could be large in our cow counting case should the expected number of cows in the picture be too large.

The output of this model processing our photo is 


[![N|Object Recognition](https://github.com/amirardalan9473/objdetrep/blob/main/demodetr.png?raw=true)]()

#Detectron2

Detectron2 is a software system for implementing implements state-of-the-art object detection algorithms. They have a model zoo and features such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend, DeepLab, etc.

A pre-trained Cascade R-CNN model from the library yields the following result on our cow photo:


[![N|Object Recognition](https://github.com/amirardalan9473/objdetrep/blob/main/cowresim.png?raw=true)]()


In summary, this was a brief overview of some methods that could be used to assist the CFIA in performing its tasks and delivering results. The use of AI and deep learning can boost efficiency, increase ease and cut costs for the CFIA.
