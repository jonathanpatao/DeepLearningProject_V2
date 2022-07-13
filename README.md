# DeepLearningProject_V2
MSC in TAU, Course: Deep Learning, Final Project
<br> starting at 3.7.22
# start here: <br>
* great info on 'sparse view CT' :<br>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3701091/ <br>

* Filtered BackProjection (FBP) explenation: <br>
(a) https://www.youtube.com/watch?v=pZ7JlXagT0w <br>
(b) https://www.youtube.com/watch?v=GGR6NTAvPao <br>

* FBP and Sinogram explenation: <br>
https://humanhealth.iaea.org/HHW/MedicalPhysics/NuclearMedicine/ImageAnalysis/3Dimagereconstruction/index.html

* read the "Grand Challenge - report" file.

# 1'st place: <br>
## Group code paper: <br>
  https://arxiv.org/pdf/2206.07050.pdf <br>
  OPTIONAL :
  https://www.researchgate.net/publication/352053806_AAPM_DL-Sparse-View_CT_Challenge_Submission_Report_Designing_an_Iterative_Network_for_Fanbeam-CT_with_Unknown_Geometry <br>
## Code (Pytorch):
* https://github.com/jmaces/aapm-ct-challenge

# 3'th place: <br>
## Group code paper: <br>
NO PAPER AVAILABLE - on 13.7 a mail was sent to the group for more info. <br>

## Code (Pytorch): <br>
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


# 5'th place: <br>
## based on paper : <br>
* "JSR-Net: A Deep Network for Joint Spatial-radon Domain CT Reconstruction from Incomplete Data" <br>
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8682178

## Code (Tensorflow): <br>
* at the branch above ("code" branch) 
* The attached .xz file is our code for DL sparse view CT Challenge.
I think the following issues are important to improve the model performance:
1) fine tuning the hyperparameters,
2) change the model architecture,
3) accurate scanning geometry,
4) augmented training data.
It is welcom to have further discussion on the deep learning models for sparse view CT imaging.


