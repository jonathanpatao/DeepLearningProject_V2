# DeepLearningProject_V2
MSC in TAU, Course: Deep Learning, Final Project
<br> starting at 3.7.22
# start here: <br>
read the "Grand Challenge - repor" file.

# 1'st place code: <br>
https://github.com/jmaces/aapm-ct-challenge

# 3'th place code: <br>
https://github.com/MaxenceLarose/GLO-4030-7030-DL-Project

## usfule notes from 3'th place: <br>
* To improve our network (called breast_CNN), you could try to train using Cosine Annealing With warm restart for the learning rate (e.g., with parameters T=10, T_mult=2) <br>
* Train for a large number of epochs (more than 200) <br>
* Try using a larger batchsize (we used 4 because we were limited in memory) <br>
* Try using group normalization layers instead of batch normalization <br>
* Try using 5x5 convolutions instead of 3x3 <br>
* bicubic upsampling gave the best results in our case <br>
* Data augmentation might help <br>
* Our architecture is similar to the U-net, but it is not published. It was inspired by this work : Shading artifact correction in breast CT using an interleaved deep learning segmentation and maximum-likelihood polynomial fitting approach - PubMed (nih.gov) <br>
https://pubmed.ncbi.nlm.nih.gov/31102462/


# 5'th place code: <br>
* at the branch above 
* The attached .xz file is our code for DL sparse view CT Challenge.
I think the following issues are important to improve the model performance:
1) fine tuning the hyperparameters,
2) change the model architecture,
3) accurate scanning geometry,
4) augmented training data.
It is welcom to have further discussion on the deep learning models for sparse view CT imaging.


