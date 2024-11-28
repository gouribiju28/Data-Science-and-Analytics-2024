# DAY 20 NOTES

## Datatypes
There are different types of data namely;
 - numeric(discrete, continuous) and
 - non-numeric(grades, categorical) type 
 - discrete and
 - continuous variable

Label columns- which we intend to predict
Feature columns - input based on which predictions are to be made.

## Two types of model:
If continuous datatypes are to be predicted, **REGRESSION MODEL**
If discrete variables are used, **CLASSIFICATION MODEL**
If label is present, it is supervised learning. Two models in supervised learning are classification and regression models.
Eg:
- classification: whether it will rain or not tmrw.
- regression : how much amt of rain can be expected.


## EXPLORATORY DATA ANALYSIS: For Visualization
Plots different graphs and infer
matplotlib, seaborn are some lbraries used for this purpose.
matplotlib: is mainly used for making data easy to visualize by utilizing gra[hs and charts.

## types of plots:
- Line plot: time series nature of variables. how y moves wrt x.
   + import matplotlib.pyplot as plt
   + plt.plot(x,y)
   + plt.xlabel(days)
   + plt.ylabel(counts)
   + plt.show()
x and y values must be sorted to do a line plot.
<p>
 <img src="https://github.com/user-attachments/assets/b08e5dd8-62b2-4f70-896b-63fbdb2ef221" width="800">
</p>


- Bar chart:
<p>
  <img src= "https://github.com/user-attachments/assets/0486a4ea-0270-47e9-87e9-89fa6f17a0fb" width="800">
</p>

- Pie chart:
<p>
 <img src="https://github.com/user-attachments/assets/08c4365e-deb9-4815-9c4b-a2688ced0657" width="800">
</p>

- Histogram
<p>
 <img src="https://github.com/user-attachments/assets/c243094e-a378-40a2-88d6-9329d87af3d1" width="800">
</p>

- Scatter plot:
<p>
 <img src="https://github.com/user-attachments/assets/833e0ded-2e7c-40f1-a7b9-551593331443"width="800">
</p>

- Normal disribution:
<p>
  <img src="https://github.com/user-attachments/assets/3580ada4-16f6-4175-a0a1-f8f20616df6f" width="800">
</p>

- KDE plot:
<p>
 <img src="https://github.com/user-attachments/assets/0a3e4924-8703-485e-9925-ce4455f6536c"width="800">
</p>

- Heatmaps:
<p>
 <img src="https://github.com/user-attachments/assets/10a4bdc0-90a8-43c3-a64c-a4aacd698eee"width="800">
</p>

<p>
 <img src="https://github.com/user-attachments/assets/fa8191fd-dd49-4654-80a0-7fd0d1981b3c"width="800">
</p>

- Boxplot:
<p>
 <img src="https://github.com/user-attachments/assets/ca5d47e0-73c5-4ebb-82f7-3f830a80d629"width="800">
</p>

## FEATURE EXTRACTION ND TRANSFORMATION
Feature Extraction and Transformation refers to a process in machine learning where raw data is converted into meaningful features by extracting relevant information and then further modifying those features to improve their suitability for a machine learning model, often by scaling, encoding, or applying mathematical operations to them.

## FEATURES AND LABELS
Features are data points which have a correlation with each other. They are domain specific vaues which is used to predict the label.
Labels are *answers* or the values which the model is trying to predict.

## FEATURE ENGINEERING
The goal of feature engineering is to create improve model accuracy by providing more meaningful and relevant information.

### Feature orthogonality 
It refers to similarity between features. If the features are orthogonal, then that means they are different. In machine learning, orthogonal features are preferred.
When orthogonal, cosine similarity  of features will be zero.
### Co-linearity
When some part of a feature is present in another feature, it is called co-linear features. For training, co-linear features are not preferred since it does not provide the required accuracy.
Non-colinear features are particularly useful for machine learning. Co-linearity essentially means similarity between features.
- Features should be co-linear with respect to target and orthogonal among each other.
### Feature Slicing
If one feature have higer amount of data points, then two models can be created to effectively study the data. Histogram can be used to check if one or two data models are required. This is used in designing of web applications and mobile apps.
### Feature Binning
It is a technique that groups numerical data into bins. It helps to understand large datasets and observe patterns.
### Mathematical Transformations
- Logarithm is taken to reduce the tailedness of a distribution, to compress larger values to smaller values in order to create a more symmetric graph.
