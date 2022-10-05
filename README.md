# Natural_Image
The classification of images that are being provided by different sources is a key visual competence that computer vision
systems focus on in recent years. Due to sudden advent of Machine learning algorithms and their good compatibility made
classification of image datasets into their corresponding classes possible enough without being explicitly programmed. This
document will capture and convey the results obtained by specific training classification models on the CIFAR-10 dataset. The
paper involves the comparison and brief analysis of the defined models with and without applying the concept of dimensionality
reduction.The results of the comparison emphasize the fact that the accuracy and performance of the system improves, especially
in the datasets which consist of a large number of images.
# **I. INTRODUCTION**
OUR course project for Pattern Recognition and Machine Learning course is ”Natural image classification”. The dataset
”CIFARr-10” contains 60,000 32x32 color images in 10 classes, with 6000 images per class. The dataset preserves 50,000
images for training and 10,000 images for testing the classification model. Our agenda is to come up with the classification
model which will give overall the best performance with respect to different image datasets. Systems extract vast amounts
of information from imagery. Information that can be vital in areas such as robotics, hospitals, self-driving cars, surveillance
or building 3D representations of objects. While each of the above-mentioned applications differs by numerous factors, they
share the common process of correctly annotating an image with one or a probability of labels that correlates to a series of
classes or categories.
May 16, 2021
# A. About ML pipeline used
This subsection describes the process(pipeline) we followed to train the models.
## 1. Data importing: 
     - We have imported the dataset from in-built keras module available in python. The original dataset contains RGB images.There are 60,000 images of size 32X32.
## 2. Data preprocessing: 
     - We converted the RGB images to grey scale. And then normalised the pixel values. To train CNN, we binarized the label column . To train rest of the models we transformed all the columns back to usual labelled form. Moreover, some other preprocessing techniques are applied based on the model which is discussed below in the "Model training subsection".
## 3. Dimensionality reduction: 
    - Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information. PCA involvs standardization, covariance matrix computation, computing eigen values and eigen vectors regarding the same. We used original data set to train CNN. We have executed two different colab notebooks of which one evaluates the mentioned machine learning models without applying dimensionality reduction techniques. While in the other notebook we have applied the concept of dimensionality reduction wherein we decided to continue with PCA. To train the rest of the models we have reduce the dimensionality using PCA. We have chosen the first 100 principal components which preserves 92 percent of variance.
## 4. Model training and Predicting output: 
    - This subsection contains summary of all the models we used for training CNN : To train a CNN model we one hot encoded the label column and use 32X32 images for trained . The architecture we followed has 6 convolutional layers and 5 dense layers. We first added 3 convolutional layers with ”relu” activation , same padding with 1 stride length and perfomed maximum pooling with 2X2 matrix. Then added 3 more convolutional layers and a pooling layer. The kernal size is 3X3 throughout . After processing the images this way we made predictions with 5 dense layers. To train the rest of the models we first converted the 32X32 numpy array of pixel values to 1D array of 1024 values .Did this for all the 60,000 images and obtained a dataframe of size 50,000 X 1024 for training , 10,000 X 1024 for validation LDA :Trained a model with LDA from sklearn Before training the rest of models we applied PCA SVM :Used inbuilt SVM module from sklearn and trained a model 100D data (PCA)
     - Random forest : Trained a randon forest classifier on 100D data . Used entropy criterion and 500 estimators Naive bias :
     - Trained a model with GausianNB from sklearn
     - XGboost : Trained a model using XGboost
## 5. model Evaluation: 
    - Evaluated all the models based on accuracy scores and confusion matrices .
This subsection describes the accuracy of each model.
# Results obtained without applying dimensionality reduction.
## CNN : 
70.13 (in percentage)
## Support Vector Classifier : 
Due to incompatibility of the system, it was left after running twice continuously upto 3 hours and
40 minutes.
## LDA : 
28.30 (in percentage)
## XGBoost : 
38.1 (in percentage)
## Random Forest Classifier : 
43.58 (in percentage)
## Naive Bayes(GaussianNB) : 
5.5 (in percentage)
# Results obtained after dimensionality reduction
## SVM : 
Accuracy of 32 percent
## LDA : 
Accuracy of 28 percent
## XG boost : 
Accuracy of 36 percent
## Random forest : 
Accuracy of 37 percent
## Naive bayes: 
Accuracy of 30 percent
# C. Comparison
The decreasing order of performance based on accuracy is
CNN , Random forest , XGboost , SVM , Naive bayes , LDA.
We can tell that CNN outperformed all the other models CNN is correctly able to classify most of the instances whereas other
models are not able to classify and this might be due to lacking some of the advanced featured as compared to CNN.
# II. CONCLUSION
Out of all the models used for training, Convolutional Neural Networks (CNN) fits good for image classification. Reported accuracy is 71%.
By applying dimensionality reduction though there is significantly less change in accuracy of model. For other models, the run time and
computational cost got considerably reduced . Overall, the computational time for processing these images is very high.
Here, CNN is customly designed. We can achieve significantly high accuracy on the CIFAR10 dataset fine tuning with existing CNN models(pretrained on 'Imagenet') such as DenseNet, ResNet etc.
