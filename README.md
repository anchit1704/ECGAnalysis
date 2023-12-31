# ECGAnalysis

In this project I analyse the diagnostic features extracted from a 12 lead ECG dataset labelled with the patient condition. I train different machine learning classification models to classify the patient condition based on the input features. 

## DataSet Description

- I am using the Diagnostic.xlsx file for the analysis. 
- The features - FileName, Rhythm, Beat, PatientAge, Gender, Ventricular Rate, Atrial Rate, QRSDuration, QTInterval, QTCorrected, RAxis, TAxis, QRSCount, QOffset, TOffset.
- Features used for classification - PatientAge, Gender, Ventricular Rate, Atrial Rate, QRSDuration, QTInterval, QTCorrected, RAxis, TAxis, QRSCount, QOffset, TOffset
- Label used for classification - Rhythm
- Unique Labels  - 11
- Labels and Count - 'AF' -445, 'AFIB'-1780, 'AT'-121, 'AVNRT'-16, 'AVRT' -8,'SA'-399, 'SAAWR'-7, 'SB'-3889, 'SR'-1826, 'ST'-1568, 'SVT' -587

## Data Preprocessing steps
- Convert gender (Male, Female) into 0,1.
- Convert age values to 0,1,2 based on certain criteria. Idea is to have similar number of data points in each bucket. 
- Encode labels into numeric form(0-11).
- As the labels are not evenly distributed create new labels to uniformly distribute the dataset.  Options - 2 labels, 4 labels.

## Algorithms Implemented
- Multinomial Logistic Regression
- Random Forest Classifier Model
- SVM Classifier Model
- XGBoost
- Neural Network

## Results

|Algorithm| Accuracy  | Precision | Recall | F1 Score | ROC_AUC|
|-------------|-------------|-------------|-------------|-------------|-------------|
|Logistic Regression| 0.823  | 0.817  | 0.823 | 0.819|0.958|
|Random Forest Classifier| 0.914  | 0.913  | 0.914 | 0.913 |0.985|
|SVM| 0.830  | 0.832  | 0.830 | 0.816 |0.954|
|XGBoost| 0.923 |  0.922 | 0.923 |  0.922| **0.987**|
|Neural Network| **0.926**  | **0.926**  | **0.926** | **0.926** |**0.987**|

