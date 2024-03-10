# Summer_Research_Internship_programme_IIT_Gandhinagar
ML for Sustainability: Satellite Data Processing for Detecting Pollution Sources Mandatory Task for IIT gandhinagar
**Introduction:**
Air pollution stands as a primary cause of mortality worldwide, with brick kilns significantly contributing to its exacerbation. Unlike other pollution sources, monitoring brick kilns proves challenging. Conventional survey-based methods for kiln identification are both costly and time-consuming. This project proposes the implementation of computer vision and machine learning models to efficiently detect brick kilns using low-label satellite imagery. Through this approach, we aim to revolutionize kiln monitoring, contributing to effective pollution control and public health preservation.
**Real-world examples of the implementation of such technology include:**
**India**: In India, where brick kilns are a significant source of pollution, the government has initiated efforts to monitor and regulate their operations. By utilizing satellite imagery coupled with machine learning algorithms, authorities can identify illegal or unregistered kilns more effectively and enforce environmental regulations.
**Bangladesh**: Bangladesh has one of the highest concentrations of brick kilns globally. The government, in collaboration with international organizations, is exploring the use of remote sensing and AI technologies to monitor kiln emissions and enforce pollution control measures.
**China**: China, as a leading producer of bricks, faces challenges in controlling brick kiln emissions. The country has invested in satellite-based monitoring systems to track kiln activity and enforce emission standards, contributing to improvements in air quality in heavily industrialized regions.

**Mandatory Task Readme File starts here**

DATASET: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals

![image](https://github.com/Prudhvinaraya/Summer_Research_Internship_programme_IIT_Gandhinagar/assets/93300939/9512d2fa-8d25-4775-9e32-b2d938eb9134)

**Introduction**

This project aims to classify animal images into various categories using deep learning techniques. It leverages the power of Convolutional Neural Networks to learn and recognize patterns within the images. The dataset used consists of images belonging to different animal categories.
The dataset comprises 90 different animal images. Initially, we'll structure it for one-vs-rest classification, followed by binary classification and then a 5-class classification problem. We'll evaluate each model's performance using classification matrices.
**Prerequisites**

Before running the code, ensure you have the following dependencies installed:
Python (>= 3.6)
TensorFlow (>= 2.0)
Keras (>= 2.0)
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
OpenCV
PIL (Python Imaging Library)

**One-vs-Rest Classification:**
1. **Data Preparation:**
The dataset contains images of various animals, including snakes.
Images are organized into folders, with each folder representing a different animal category.
2. **Data Loading and Preprocessing:**
Images are loaded using TensorFlow's tf.data.Dataset API.
Each image is resized to a fixed size (128x128 pixels) and normalized to ensure consistency across the dataset.
3. **Label Encoding:**
The LabelEncoder from scikit-learn is used to encode the class labels.
In this case, the label for snakes is encoded as 0, and labels for other animals are encoded as 1.
4. **Model Creation:**
A custom CNN model is defined using Keras' Sequential API.
The model consists of convolutional layers, max-pooling layers, dropout layers, batch normalization, and dense layers.
The final dense layer uses softmax activation to output probabilities for the two classes (snake vs. not snake).

The below snippet shows the Model for one v/s Rest Classification
![image](https://github.com/Prudhvinaraya/Summer_Research_Internship_programme_IIT_Gandhinagar/assets/93300939/4e02c960-d813-4a8d-be1e-e940f2317729)
The below snippet show the Model for 5 class classfication .The only difference is that i changed the output channels from 2 to 5.
![image](https://github.com/Prudhvinaraya/Summer_Research_Internship_programme_IIT_Gandhinagar/assets/93300939/17032489-e858-4a74-8fb2-42de855ce6c9)


6. **Training:**
The model is trained using the binary cross-entropy loss function and the Adam optimizer.
Training is performed using the dataset containing images of snakes and images of other animals.
The below snippets shows the Image of animal and corresponding Feature Map which enables easy for classificaion
![image](https://github.com/Prudhvinaraya/Summer_Research_Internship_programme_IIT_Gandhinagar/assets/93300939/1a3629b4-36c6-4dc3-9e19-224c930bf68d)

![image](https://github.com/Prudhvinaraya/Summer_Research_Internship_programme_IIT_Gandhinagar/assets/93300939/e74dc95e-30a2-4ba5-9e4e-199ea5466438)

8. **Evaluation:**
During evaluation, the trained model predicts whether an input image contains a snake or not.
Performance metrics such as accuracy, precision, recall, and F1-score has been computed using techniques such as classification report and confusion matrix.
9. **Multi-Class Extension:**
The code snippet also extends the OvR approach to a **multi-class classification problem with five different animal categories.**
For each animal category, a separate binary classifier is trained to distinguish it from the rest of the categories.
The model architecture and training process are adjusted accordingly to accommodate the additional classes.
**Summary:**
The provided code demonstrates how to implement OvR classification using a custom CNN model in TensorFlow/Keras. It preprocesses the data, defines the model architecture, trains the model, and evaluates its performance for both binary (snake vs. not snake) and multi-class classification tasks. This approach simplifies the classification task by breaking it down into multiple binary classification sub-problems, making it easier to train and interpret the model's predictions.


**Results**

The project achieves the following results:

Binary classification (Snake vs. Not Snake): 99.4% accuracy

Multi-class classification (5 classes): 98.5% accuracy
Accuracy of Training and Validation of Mulit-classification Model
![image](https://github.com/Prudhvinaraya/Summer_Research_Internship_programme_IIT_Gandhinagar/assets/93300939/eaf4fbdb-c33e-4f19-8fe2-0939d23020ff)

Loss of Training and Validation of Mulit-classification Model
![image](https://github.com/Prudhvinaraya/Summer_Research_Internship_programme_IIT_Gandhinagar/assets/93300939/5992d62b-0402-4021-b808-528d6a50d919)

Confusion matrix :
It seems there is some Bias in my model .i will try to fine_tune my model
![image](https://github.com/Prudhvinaraya/Summer_Research_Internship_programme_IIT_Gandhinagar/assets/93300939/d9d09d89-7048-46f4-89fc-904c94b1eb03)

You can find detailed performance metrics and visualizations in the Results section of the **srip-iit-gandhinagar-submission.ipynb File** .

