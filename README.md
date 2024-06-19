<h2 align="center">Crop Recommendation Pipeline</h2>

<h2 align="center">Dataset</h2>

The dataset used was obtained from Kaggle, named "<a href="https://www.kaggle.com/datasets/varshitanalluri/crop-recommendation-dataset">Crop Recommendation Dataset</a>". This dataset contains the following variables:

- Nitrogen: ratio of nitrogen in the soil.
- Phosphorus: ratio of Phosphorus content in the soil.
- Potassium: ratio of Potassium content in the soil.
- Temperature: temperature in degrees Celsius.
- Humidity: relative humidity in %.
- pH_Value: pH value of the soil.
- Rainfall: rainfall in mm.
- Crop: contains 22 unique values of different grown crop types.

In total, there are 2200 cases in this dataset.

During this project, an exploratory data analysis and a machine learning application will be pursued.

<h2 align="center">Exploratory Data Analysis</h2>

First, the dataset was loaded. Then, an overview of the dataset was done. There were 8 variables (7 numeric and 1 object) and 2200 cases. The object variable ```Crop``` was then converted to categorical. No null values or duplicates were found in the dataset. 

The following statistics were calculated:

| | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| Nitrogen | 2200.0 | 50.5518 | 36.9173 | 0.0 | 21.0 | 37.0 | 84.25 | 140.0 |
| Phosphorus | 2200.0 | 53.3627 | 32.9858 | 5.0 | 28.0 | 51.0 | 68.0 | 145.0 |
| Potassium | 2200.0 | 48.1490 | 50.6479 | 5.0 | 20.0 | 32.0 | 49.0 | 205.0 |
| Temperature | 2200.0 | 25.6162 | 5.0637 | 8.8256 | 22.7693 | 25.5986 | 28.5616 | 43.6754 |
| Humidity | 2200.0 | 71.4817 | 22.2638 | 14.2580 | 60.2619 | 80.4731 | 89.9487 | 99.9818 |
| pH_Value | 2200.0 | 6.4694 | 0.7739 | 3.5047 | 5.9716 | 6.4250 | 6.9236 | 9.9350 |
| Rainfall | 2200.0 | 103.4636 | 54.9583 | 20.2112 | 64.5516 | 94.8676 | 124.2675 | 298.5601 |

As we could see from the min and max values, there is a noticeable difference in the range of the variables.

We then plotted the count of each category, observing that there are 22 categories with 100 cases each.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop1.png)

To check the distribution of the variables and visually check outliers we plotted histograms and botplots.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop2.png)

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop3.png)

We observed that the variables did not follow a normal distribution and that outliers could be found in variables such as: Phosphorus, Potassium, Temperature, pH value and Rainfall.

Hence, we proceded to plot each variable by the ```Crop``` category. 

In ```Nitrogen```, we could observe two groups of crops that need either high amount of nitrogen in soil or low.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop4.png)

In ```Phosphorus```, however, both grapes and apples have a high phosphorus requirement; hence, being susceptible of outliers.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop5.png)

As in the previous variable, in ```Potassium```, grapes and apples could be susceptible of being outliers.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop6.png)

For the ```Temperature``` we could observe that all variable range in the same values.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop7.png)

We also observed that chickpeas and kidney beans need a lower ```Humidity``` compared to the other crops.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop8.png)

The ```pH_Value``` range of all crops seemed to be comparable through all of them.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop9.png)

Finally, we observed that rice needed much more ```Rainfall``` than the rest of the crops.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop10.png)

Once we visualized the variables by themselves or compared to all the crops, we proceeded to create a correlation matrix and plot it in a heatmap.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop11.png)

As we could observe in the heatmap, there was a high correlation in between the ```Potassium``` variable and the ```Phosphorus``` variable. This could be explained as phosphorus being found in nature as a potassium salt (K<sub>2</sub>PO<sub>4</sub>). As we could observe in the previous boxplots, both grapes and apples needed a high amount of this salt.

We then proceeded to calculate the Variance Inflation Factor (VIF) of the variables in order to check multicollinearity, obtaining the following values:

| Variable | VIF |
|---|---|
| Nitrogen | 1.0970 |
| Phosphorus | 2.6304 |
| Potassium | 2.7971 |
| Temperature | 1.1111 |
| Humidity | 1.3689 |
| pH_Value | 1.0558 |
| Rainfall | 1.0374 |

As no high values (>5) were observed, it was concluded that there was no multicollinearity.

Afterwards, we calculated the Interquartile Range (IQR) and removed the values that were higher than Q3+1.5xIQR and lower than Q1-1.5xIQR. The following cases of each category were left after removing the outliers.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop12.png)

We could observe that the crops apples and grapes were completely removed. Moreover, the chickpeas lost more than 40% of the original data. Same happened with the mothbeans and papayas. Rice also suffered a high loss of 60% of the data. As this information is crucial to determine which crop is suitable for the soil, the "outliers" were not removed. Hence, 22 crops with 100 cases each were used from here on.

We then calculated the skewness of all the variables to check the normality of the data.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop13.png)

Hence, we needed to transform the data to correct the normality. After testing logarithmic, square root, Yeo Johnson and quantile transformations, we observed that the lowest skewness was observed using the quantile transformation (see data in the <a href="https://github.com/romaniegaa/crop-recommendation/crop_recommendation.ipynb">Jupyter Notebook</a>). Hence, the data was transformed using the quantile transformation.

<h2 align="center">Machine Learning</h2>

We prepared the data into training and testing datasets with their corresponding labels to test different algorithms.

## Support Vector Machine

We first started with the Support Vector Machines, by trying different models (linear, poly, Radial Basis Function and sigmoid) and different C values (1, 10, 100, 1000). After training and evaluating all the models, the best conditions were the next one:

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop14.png)

The best results were obtained with the Radial Basis Function model and a C value of 100, with an accuracy of 95.91%.

## Random Forest

We then tested the Random Forest algorithm with different number of trees (10, 100, 500, 1000), obtaining the next one as the best result.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop15.png)

Hence, we obtained a 97.95% of accuracy with 500 trees.

## Decision Tree

We tested the Decision Tree algorithm for this task, obtaining an accuracy of 95.23%.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop16.png)

## K Nearest Neighbours

For this algorithm we tested different k hyperparameters (1, 3, 5, 7), obtaining the following result:

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop17.png)

With a k value of 5, the best accuracy was obtained, being 94.32%.

## Naive Bayes

Finally we tested the Naive Bayes algorithm with this dataset, obtaining an accuracy of 96.59%.

![](https://github.com/romaniegaa/Portfolio/blob/main/images/crop18.png)

<h2 align="center">Conclusion</h2>

After exploring the dataset, it was observed that it was already clean because no duplicates and no nulls were found. It was also observed that the variables did not follow a normal distribution. Different transformations were tested. However, the best one was the quantile transformation due to having the less median skewness. Afterwards, different machine learning algorithms were tested. High accuracies were obtained overall. However, Random Forest algorithm yielded the highest accuracy with a value of 97.95%.

<h2 align="center">Used libraries</h2>

- ```Pandas```: to work with the data.
- ```NumPy```: to manipulate the numeric data.
- ```Plotly```: to make 2D graphs.
- ```SciPy```: to calculate statistics.
- ```Sklearn```: to implement the machine learning models.
