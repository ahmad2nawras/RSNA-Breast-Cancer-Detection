# RSNA-Breast-Cancer-Detection

This project has been done in collaboration with Mohamad Issam Sayyaf
## Introduction

Breast cancer is a significant cause of cancer-related fatalities among women globally. Timely detection of breast cancer is critical for successful treatment and improved survival rates. In recent times, deep learning techniques have emerged as promising tools for detecting breast cancer from medical images, including mammograms.

The RSNA Breast Cancer Detection competition, hosted on Kaggle, serves as an important initiative to foster the development of machine learning models for breast cancer detection. Participants are provided with a dataset of mammogram images, accompanied by labels indicating the presence or absence of breast cancer. The primary challenge is to create deep learning models capable of accurately detecting breast cancer from these mammogram images.

In this report, we embark on an exploration of the RSNA Breast Cancer Detection competition, where we will closely examine the approaches and techniques employed by the top-performing models. Our objective is to gain valuable insights into the best practices for utilizing deep learning methods in breast cancer detection and understand the advancements made in this critical field.

# Medical Imaging

The RSNA Breast Cancer Detection competition focuses on medical imaging, specifically the utilization of mammograms for breast cancer detection. Mammography serves as the current gold standard for breast cancer screening and is recommended by medical experts for women over the age of 50. However, mammography has its limitations, including false positives and false negatives, leading to potential consequences such as unnecessary biopsies and missed diagnoses.

## a. Dataset Descriptions

The dataset provided for Task 1 of the RSNA Breast Cancer Detection competition plays a pivotal role in the competition. It comprises over 54,000 digital mammography images in DICOM format, encompassing both cancerous and non-cancerous cases.

The dataset poses unique challenges to developers due to its size and complexity. With a size exceeding 300 GB, processing and analyzing the dataset demands substantial computational resources. Additionally, the dataset exhibits an imbalance, containing a higher number of non-cancerous cases compared to cancerous cases (see Figure 1). This imbalance presents a significant hurdle for the development of accurate models that can successfully detect cancerous cases while minimizing false positives.

To overcome these challenges, developers must engage in extensive pre-processing of the dataset, which includes tasks such as resizing, balancing, and normalization. The pre-processing stage aims to standardize image dimensions, balance the dataset, and enhance image contrast, ultimately improving model accuracy.

<p align="center">
  <img src="https://github.com/IssamSayyaf/RSNA-Breast-Cancer-Detection/blob/main/images/Image%20Distribution.png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 1: The Dataset Distribution</em>
</p>

# Methods

To design an efficient model for breast cancer detection, a well-defined pipeline can be established, encompassing pre-processing of the dataset, designing and training the model architecture, and evaluating the model's performance. The pipeline can be described in the following steps:

## i. Preprocessing

The pre-processing pipeline for the RSNA Breast Cancer Detection dataset can be summarized in the following steps:

1. **Read the DICOM images:**
   The dataset is provided in DICOM format, which requires reading and conversion into a format compatible with model processing.

2. **Crop the images:**
   To remove any background noise and reduce redundancy in the images, the breast region must be identified and cropped. Techniques such as thresholding can be employed, as exemplified in Figure 2.

3. **Normalize the pixel values:**
   Normalizing the pixel values of the images to a range of [0, 255] ensures consistent image representation and prevents model saturation.

4. **Enhance contrast:**
   Utilizing contrast enhancement techniques, such as Contrast Limited Adaptive Histogram Equalization (CLAHE), can significantly improve the visibility of breast tissue in the images. CLAHE adjusts image contrast locally while preserving overall contrast and image quality.

5. **Resize the images:**
   Ensuring uniform image dimensions is crucial for consistency and reduced computational load on the model. Therefore, resizing the images to a standardized size is essential.

6. **Save the processed images:**
   After pre-processing the dataset, saving the images in an efficient format for model training is vital. One effective method is using the TFRecord format.

Overall, the pre-processing pipeline aims to standardize image dimensions, enhance contrast, and balance the dataset, ultimately leading to an optimized dataset for model training and improved model performance, as depicted in Figure 3.
 
<p align="center">
  <img src="https://github.com/IssamSayyaf/RSNA-Breast-Cancer-Detection/blob/main/images/Crope%20and%20resize%20images..png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 2: TCrope and resize images</em>
</p>
## ii. Designing and Training the Model Architecture

In this section, we will discuss the steps involved in designing the model architecture and training the model. Once the preprocessed data is downloaded from the TF record, the first crucial step is to address the dataset's imbalance, as illustrated in Figure 1. To achieve balance, we can carefully select all images containing breast cancer and an equal number of non-cancerous images, resulting in a balanced dataset.

Next, the balanced dataset is split into three subsets: the training set, the validation set, and the test set. Typically, the training set constitutes 60% of the dataset, while the validation set and test set account for 38% and 2%, respectively. Figure 4 provides an illustration of the training set.

The design and architecture of the model will involve selecting appropriate deep learning frameworks, defining the layers and parameters of the model, and configuring the optimizer and loss function. The training process will entail feeding the model with the training set, iteratively adjusting the model's weights, and evaluating its performance on the validation set to prevent overfitting.

The final step involves testing the trained model on the test set to obtain an accurate assessment of its performance and its ability to accurately detect breast cancer from mammogram images.


<p align="center">
  <img src="https://github.com/IssamSayyaf/RSNA-Breast-Cancer-Detection/blob/main/images/The%20Processed%20images.png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 3: The Processed images.png</em>
</p>

In this problem will use model for binary classification, which means that it is designed to classify input images into two categories: cancerous or non-cancerous. The model is built using TensorFlow and the EfficientNetB7 architecture, which is a pre-trained convolutional neural network (CNN) that has achieved state-of-the-art performance on many computer vision tasks.
The first step in the model is to define the input shape of the images, which in this case is 1456 by 728 pixels with one channel (grayscale). Then, an inputs layer is defined using the Input function in Keras.
Next, a data augmentation layer is defined using Sequential, which applies a series of random transformations to the input images during training. The transformations include horizontal flipping, zooming, and contrast adjustment. This is done to increase the diversity of the training data and prevent overfitting.

<p align="center">
  <img src="https://github.com/IssamSayyaf/RSNA-Breast-Cancer-Detection/blob/main/images/Images%20after%20processing.png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 4: The trainning Processed images.png</em>
</p>

In this problem will use model for binary classification, which means that it is designed to classify input images into two categories: cancerous or non-cancerous. The model is built using TensorFlow and the EfficientNetB7 architecture, which is a pre-trained convolutional neural network (CNN) that has achieved state-of-the-art performance on many computer vision tasks.
The first step in the model is to define the input shape of the images, which in this case is 1456 by 728 pixels with one channel (grayscale). Then, an inputs layer is defined using the Input function in Keras.
Next, a data augmentation layer is defined using Sequential, which applies a series of random transformations to the input images during training. The transformations include horizontal flipping, zooming, and contrast adjustment. This is done to increase the diversity of the training data and prevent overfitting.

After the augmentation layer, a normalization layer is added to the input to ensure that the pixel values of the images are scaled between 0 and 1. This is important for the training process and can improve the accuracy of the model.
The pre-trained EfficientNetB7 model is loaded without its top layer, which is the layer responsible for classification. The first 800 layers of the pre-trained model are frozen to prevent their weights from being updated during training, as they have already learned valuable features from a large dataset.
Then, several convolutional layers are added to the top of the model. Each convolutional layer is followed by a batch normalization layer, which normalizes the outputs of the previous layer to improve training stability. The convolutional layers are used to learn more complex and abstract features from the input images.
After the convolutional layers, the output is flattened and passed through several dense layers. The dense layers are responsible for learning the final classification decision from the learned features. The number of neurons in the dense layers decreases as the layers get closer to the output layer.
Finally, the output layer has a single neuron with a sigmoid activation function, which outputs a probability value between 0 and 1. The model shown in Figure 5, this value represents the predicted probability that the input image is cancerous. 
 
<p align="center">
  <img src="https://github.com/IssamSayyaf/RSNA-Breast-Cancer-Detection/blob/main/images/The%20Architecture%20of%20the%20proposed%20model.png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 5: The Architecture of the proposed model</em>
</p>
## iii. Results

After training the model for 100 epochs, we achieved a training accuracy of 93.59% and a validation accuracy of 98.7% (see Figure 6). These results suggest that the model has learned to generalize well to unseen data and is likely to perform well on new test data.

To verify the model's performance on test data, we evaluated it on a separate set of previously unseen test samples. The model achieved an accuracy of 98.91% on the test set, which confirms its ability to accurately classify cancer and non-cancer images.

Overall, these results demonstrate the effectiveness of the model in detecting cancer, and suggest that it could be a useful tool for aiding medical professionals in the diagnosis of cancer.
<p align="center">
  <img src="https://github.com/IssamSayyaf/RSNA-Breast-Cancer-Detection/blob/main/images/Training%20Progress%20Evolution%20of%20Accuracy%20and%20Loss%20over%20Epochs.png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 6: Training Progress Evolution of Accuracy and Loss over Epochs</em>
</p>

### Confusion Matrix Analysis

A confusion matrix is a table that is often used to describe the performance of a classification model. It summarizes the predictions made by a classifier against the actual labels of the data being classified.

The confusion matrix is a 2x2 table, where each row represents the instances in a predicted class and each column represents the instances in an actual class as shown in Figure 7. The four possible outcomes are:

- True Positive (TP): the classifier predicted the class correctly and the actual class was positive.
- False Positive (FP): the classifier predicted the class incorrectly and the actual class was negative.
- False Negative (FN): the classifier predicted the class incorrectly and the actual class was positive.
- True Negative (TN): the classifier predicted the class correctly and the actual class was negative.

In the given result of the confusion matrix, the table shows that there were 496 true negative predictions and 7 false negative predictions for the first class. Additionally, there were 8 false positive predictions and 497 true positive predictions for the second class as shown in Figure 8.

These results suggest that the model performed well in classifying both classes, with a high number of true positive and true negative predictions. However, there were a small number of false positive and false negative predictions, indicating that there may be some misclassification errors.

Overall, the confusion matrix provides valuable information about the performance of the classification model and can help identify areas for improvement and optimization.



<p align="center">
  <img src="https://github.com/IssamSayyaf/RSNA-Breast-Cancer-Detection/blob/main/images/Confusion%20Matrix%20Evaluation%20of%20Model%20Performance%20through%20Actual%20vs%20Predicted%20Classifications.png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 7: Confusion Matrix Evaluation of Model Performance through Actual vs Predicted Classifications</em>
</p>





Precision, recall, and F1 score are essential metrics used to evaluate the performance of a binary classification model. They offer valuable insights into the model's ability to identify positive and negative instances and facilitate the comparison of different models.

### Precision

Precision measures the proportion of true positives among the predicted positive instances. A high precision indicates accurate positive predictions, while a low precision suggests an overabundance of false positives. In this case, the precision is 98.6%, signifying a high proportion of true positive predictions.



### Recall (Sensitivity)

Recall, also known as sensitivity or true positive rate, measures the proportion of true positives correctly identified by the model. A high recall indicates correct identification of positive instances, while a low recall implies missed true positive instances. The recall in this case is 98.4%, denoting the model's high ability to correctly identify positive instances.



### Specificity

Specificity is the proportion of true negatives (TN) correctly identified by the model over the total number of actual negative samples. The specificity of the model equals 98.6%, reflecting its capability to correctly identify negative instances.



### F1 Score

The F1 score represents the harmonic mean of precision and recall, providing a balanced measure of the two metrics. It is calculated as the weighted average of precision and recall. A higher F1 score indicates better performance. In this case, the F1 score is 98.5%, indicating a strong balance between precision and recall.


### False Positive Rate (FPR)

The False Positive Rate (FPR) is a statistical measure used to evaluate the performance of a binary classification model. It represents the ratio of false positive predictions made by the model to the total number of actual negative instances. The FPR for the proposed model equals 1.39%.



### AUC - ROC Curve

The AUC - ROC curve is a performance measurement for classification problems at various threshold settings. It is a probability curve, and AUC represents the degree of separability. Higher AUC values indicate better model performance in distinguishing between classes. For example, in a medical diagnosis scenario, higher AUC signifies the model's ability to effectively distinguish between patients with the disease and those without.
Figure 8 displays the AUC (Area Under the Curve) score for the proposed model, which is equal to 0.99. This suggests that the model is able to distinguish between positive and negative instances with high accuracy. However, without further information on the specific problem and the data used, it is difficult to determine the significance of this score.
<p align="center">
  <img src="https://github.com/IssamSayyaf/RSNA-Breast-Cancer-Detection/blob/main/images/AUC%20-%20ROC%20Curve%20for%20the%20proposed%20model.png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 8: AUC - ROC Curve for the proposed model</em>
</p>

Overall, these metrics provide crucial insights into the performance of the classification model, aiding in assessing its accuracy and efficiency in detecting breast cancer from mammogram images.


Based on the results, the model has performed exceptionally well with a high training accuracy of 93.59%, a validation accuracy of 98.7%, and a test accuracy of 98.91%. The confusion matrix shows that the model has made very few misclassifications with only 8 false positive and 7 false negative predictions out of a total of 1008 samples. The precision of the model is 98.6%, which indicates that the model's positive predictions are highly accurate, and the recall is 98.4%, which means that the model has correctly identified a high percentage of positive instances. Additionally, the model's specificity is 98.6%, indicating that the model's negative predictions are also highly accurate. The F1 score is 98.5%, which is a good balance between precision and recall, and the FPR is 1.39%, indicating that the model has made very few false positive predictions. Finally, the AUC is 99%, which is an excellent score and indicates that the model has a high ability to distinguish between positive and negative instances. Overall, the model has performed exceptionally well, and the results are very promising for its practical application
