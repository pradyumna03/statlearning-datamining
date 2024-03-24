The objective of this project was to build a model from the labeled data and predict the class instances of the prepared model for the unlabeled data.
**Data Description**
Two datasets labeled and unlabeled data was given. Both the dataset contains binary and numerical attributes. Column y is the class attribute (three classes).
The index column is used to order the data and not be used in the model. Missing data are indicated with blank cells and will have to be dropped by using df=df.dropna().

**Preprocesing of the data**
For the preprocessing I followed two steps: 
1) Dropping the na models: There are missing data which is indicated with blank cells in the excel file provided. I dropped all the missing data instances in the data frame with df = df.dropna()
2) Normalize columns: Normalizing the column will help the dataset be in a common scale. Since some are in the range from 0 – 1 and others with more than 150. I filled the na with 0.

**Methods and Parameters**
Baseline Method: 
This method is used to predict values for non-preprocessed data.
After splitting, I fit training data to basic Random Forest model where the number of trees is 10. None of the other hyperparameters have been fine tuned.
After predicting result for sample test data, I found the BER and generated a classification report.
The accuracy I was receiving was 62% and the BER 0.411.

**Random Forest Method**
I normalize the columns to same range which is 0-1, this is done because there are values that are in single digits and some in triple.
After splitting the data, I fine tuned the RandomForest algorithm with the help of RandomSearch this helps us estimate the best values for the various hyperparamters of the randomforest classifier as given below:
(n_estimators= 1783,
 min_samples_split= 10,
 min_samples_leaf= 1,
 max_features= 'sqrt',
 max_depth= 30,
 bootstrap= False)
Similarly, I found the BER = 0.388 Macro Avg f1.score = 0.63
Weighted avg f1.score = 0.65
Accuracy = 0.64

**Neural Network**
I used a different train and test split. I took the validation size = 0.1 Before passing the data through the Neural Network I encoded the features using the Label Encoder.
This is a simple model which consists of two dense layers
Batch Normalization and Final dense layer.
I used the categorical_crossentropy loss function and Adam optimizer.
The BER = 0.448
Accuracy = 0.58
Macro avg f1.score = 0.56 Weighted avg f1.score = 0.59

**Support Vector Machine**
The data was preprocessed the same way and fine tuning has been done using GridSearch.
After running the code, the values generated are:
BER = 0.38
Accuracy = 0.65
Macro avg f1.score = 0.62 Weighted avg f1.score = 0.65

**Result and Evaluation**
First, a comparison was made between deep learning models (the neural network) and machine learning models (SVM and the Random Forest Classifier). 
The machine learning based approaches did better than the deep learning model. From the results and the epoch cycles, it could be seen that the model is overfitting and a possible explanation for the same is that the data size might be too small for the model.
Next, the SVM and the Random Forest Classifier models were trained and tested, achieving similar results. 
The parameters for the Random Forest Classifier were chosen after implementing a Random Search algorithm. As for the SVM model, the fine-tuning was done with the Grid Search algorithm.
After using both the classifier Random Forest and Support Vector I got the BER as 0.38 and accuracy of 64% for Random Forest and similarly got the BER as 0.38 and Accuracy of 65% for the Support Vector.
Since there is a minute difference in the accuracy, I will select SVM as the best model. Because the data provided to us is sparse and can be classified easily with the help of SVM. SVM here works faster and gives a better result.
