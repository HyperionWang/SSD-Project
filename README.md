# Semantic Segmentation

[//]: # (Image References)
[image1]: ./docs/VGG_FCN.png
[image2]: ./runs/1527143741.7427516/um_000000.png
[image3]: ./runs/1527143741.7427516/um_000018.png
[image4]: ./runs/1527143741.7427516/um_000030.png
[image5]: ./runs/1527143741.7427516/uu_000003.png
[image6]: ./runs/1527143741.7427516/uu_000096.png

### Introduction
In this project, The fully convolution network (FCN) is trained and used to label the pixels of a road in images using a Fully Convolutional Network (FCN).

The VGG16 architechture is used and FCN is connected following the VGG to have the pixel by pixel labeling.

### Implementations

FCN-8s with VGG16 is used. The network connection is as following:

![alt text][image1]

The data set used to train and test is Kitti Road dataset.

The test on the VGG Model loading, layer test, optimizer test, and netwrok training are all passed.

The following paramters are used to train the FCN:

    L2_REG = 1e-4
    KEEP_PROB = 0.5
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    BATCH_SIZE = 16
    IMAGE_SIZE = (160, 576)
    NUM_CLASSES = 2
    Optimizer = Adam


The PC platform is using NVIDIA GTX1080 Ti.
The loss at the starting point is 8.5, and after the 70th epoch, the loss is around 0.16. And the final loss after 100 epoches is 0.083.

### Results

The final results could be found in the runs/1527143741.7427516

Some images of labeling using the trained model:

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]


### Summary:

The implemented FCN with VGG16 model is trained and valided using the road image data set. And according to the final testing results, the model is able to label most pixels of roads close to the best solution. And the error of the result is reasonably low.


### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
