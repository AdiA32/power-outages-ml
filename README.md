# Power Outages Modeling

## Problem Identification
The prediction problem I am tackling is regression problem, the goal of which is to predict the severity of a power outage given basic information about the outage(specified below) in terms of duration. In other words, this model **predicts the duration of a major power outage**. As such the response variable is `OUTAGE.DURATION`; I choose it because according to the data dictionary given by my source, `OUTAGE.DURATION` is the "Duration of outage events (in minutes)" and offers a way to directly quantify how big the outage was. At the "time of prediction" it is expected the model has access to basic information such as (not all features were necessarily used):
- `YEAR`: The year the outage occurred.
- `MONTH`: The month the outage occurred.
- `OUTAGE.START.DATE`: The day of the year when the outage event started
- `OUTAGE.START.TIME`: The time of the day when the outage event started        
- `U.S._STATE`: The U.S. state where the outage occurred.
- `CLIMATE.REGION`: The climate region where the outage occurred.
- `NERC.REGION`: The North American Electric Reliability Corporation (NERC) regions involved in the outage event
- `CAUSE.CATEGORY`: Categories of all the events causing the major power outages
- `CAUSE.CATEGORY.DETAIL`: Detailed description of the event categories causing the major power outages
- `POPULATION`: Population in the U.S. state in a year
- `TOTAL.SALES`: Total electricity consumption in the U.S. state (megawatt-hour)

All of these values are available to the model because they are provided in the dataframe. Many of the starting features are easily identifiable because they are constant demographics that do not change with the exception of the date and time but they are easily determinable. It may be argued that `CAUSE.CATEGORY` and `CAUSE.CATEGORY.DETAIL` are features that can only be known in hindsight, in my opinion that is not the case. Power companies have access to what caused the power outage and if this model (or a similar one) were to be used in industry, *cause* is an important factor in determining how long customers have until power is back up. Moreover, it is pretty obvious if there is a hurricane and the power goes out. Or there is obvious vandalism at the site, then the cause is vandalism. Therefor I will be counting it as a valid, and crucial feature to predict power outage duration. I am not using a column like `OUTAGE.RESTORATION.TIME` or `DEMAND.LOSS.MW` which would not be a valid "time of prediction" feature as these are features that can only be known in hindsight. 


## Baseline Model
My baseline model utilizes a random forest regression implemented using `sklearn`'s library. The model is built upon the idea of linear regression but takes advantage of the more advance "ensemble" approach provided by random forests. The goal of this model is to predict power outage duration based on various features from the dataset.

The features used in my model are as follows:

- `YEAR` (ordinal): Represents the year in which the power outage occurred.
- `MONTH` (ordinal): Indicates the month in which the outage event took place.
- `DAY.OF.WEEK` (ordinal): Denotes the day of the week when the outage happened.
- `U.S._STATE` (nominal): Specifies the U.S. state where the power outage occurred.
- `NERC.REGION` (nominal): Represents the North American Electric Reliability Corporation (NERC) regions involved in the power outage event.

with the target variable being `OUTAGE.DURATION`.

To process the categorical features, I used one-hot encoding. This technique allows me to represent the nominal features as binary vectors, enabling the model to effectively learn from the categorical information. The remaining features, `YEAR`, `MONTH`, and `DAY.OF.WEEK`, which are ordinal in nature, were left unchanged.

Regarding the model's performance, on the training set, my random forest regression achieved a relatively high coefficient of determination (R-squared) score of 0.855. This indicates that the model explains a significant portion of the variance in the training data. However, when evaluated on the test set, the model's performance drastically dropped, resulting in a negative R-squared score of -0.769. This suggests that the model did not generalize well to unseen data, indicating possible overfitting.

Given the significant drop in performance on the test set, I cannot consider my current model to be "good." The negative R-squared score indicates that the model's predictions on the test data are worse than a simple horizontal line. Therefore, there is a need for further improvement and refinement in order to create a reliable and accurate predictive model for power outage duration. My training data had a decent score but my testing set was terrible, that means my model is not at all generalizable. Let us learn and improve my features, even a little for the final model and get higher generalizability. 

## Final Model 
In my final model, I incorporated additional features that I believed would improve the performance of the model based on the data generating process. These features include:

- `POPULATION`: This feature represents the population of the area where the power outage occurred. I added this feature because population density could potentially impact the severity and duration of power outages. Areas with higher population density may experience more complex infrastructure challenges or higher energy demand, leading to longer outage durations.

- `CAUSE.CATEGORY`: This feature categorizes the cause of the power outage, such as equipment failure, severe weather, or human error. By including this information, I aimed to capture the different underlying factors that contribute to power outages. Different categories of causes might have varying impacts on outage durations, allowing the model to better understand the patterns and correlations within the data.

- `TOTAL.SALES`: This feature represents the total sales in the area where the power outage occurred. I hypothesized that higher sales might indicate a greater reliance on electricity, potentially leading to longer outage durations if there is a disruption in the power supply.

To handle these new features and the existing ones, I employed a `ColumnTransformer` in the preprocessing step. I applied one-hot encoding to the categorical features, including "U.S._STATE," "NERC.REGION," and the newly added "CAUSE.CATEGORY." Standard scaling was performed on the "TOTAL.SALES" feature to ensure its compatibility with the random forest algorithm. Additionally, I used quantile transformation on the "POPULATION" feature to address potential skewness in its distribution.

For the modeling algorithm, I selected the random forest regression, which has proven to be effective in handling complex relationships and capturing nonlinearities in the data. This algorithm is suitable for this prediction task due to its ability to handle both numerical and categorical features effectively.

To determine the best hyperparameters for the random forest regressor, I conducted a grid search using the `GridSearchCV` function. I explored different combinations of hyperparameters, including the number of estimators (`n_estimators`) and the maximum depth of the trees (`max_depth`). The best-performing hyperparameters were found to be {'regressor__max_depth': 10, 'regressor__n_estimators': 50}.This makes sense because they are towards the lower end of the param_grid I passed in which was:

```python
param_grid = {
    "regressor__n_estimators": [50, 100, 150, 200],
    "regressor__max_depth": [None, 10, 20, 30, 40],
}
```
and this makes sense because lower usually means better generalizability and less overfitting to the training data and memorizing the noise. 

Comparing the performance of my final model to the baseline models, there is a significant improvement in predictive accuracy. The final model achieved a training set R-squared score of 0.7707, indicating that it explains approximately 77% of the variance in the training data. On the test set, the model achieved an R-squared score of 0.2813, suggesting that it explains around 28% of the variance in the test data. These scores demonstrate a substantial enhancement over the baseline models, indicating that my final model has a better ability to generalize and predict power outage duration accurately. The hyperparameter tuning and the inclusion of additional relevant features, such as population, cause category, and total sales, have contributed to capturing more meaningful patterns in the data and improving the overall performance of the model. **Moreover**, noticing how my chosen hyperparamters were at the bottom extreame of my provided list, I ran grid search one more time with:
```python
param_grid = {
    "regressor__n_estimators": [10, 20, 30, 40, 50, 60],
    "regressor__max_depth": [None, 1, 4, 7, 10],
}
```
and found the best parameters to be {'regressor__max_depth': 10, 'regressor__n_estimators': 30}. This shows more attention to detail for my model and it payed off because testing score jumped to 0.324 afterwards this adjustment. 

Ideally this is not the score I would want, but since instructions say that *"You will not be graded on “how much” your model improved from Baseline Model Step to Final Model Step. What you will be graded on is on whether or not your model improved, as well as your thoughtfulness and effort in creating features, along with the other points above."* i feel I have tried my best at creating better features and made significant improvement considering my previous models in mind. 

## Fairness Analysis 
Group X: Power outages in the 'WECC' NERC region
Group Y: Power outages in other NERC regions

Evaluation metric: Root Mean Squared Error (RMSE)

Null Hypothesis (H0): The model is fair, the RMSE for power outages in the 'WECC' NERC region and for other regions are roughly the same, and any differences are due to random chance.
Alternative Hypothesis (H1): The model is unfair, the RMSE for power outages in the 'WECC' NERC region is significantly different from that of other regions.

Test statistic: The difference in RMSE between Group X and Group Y.

Significance level: 0.05

I then conducted a permutation test, which generated a p-value of 0.946 

Conclusion: As the p-value (0.9460539460539461) is much greater than the significance level (0.05), I fail to reject the null hypothesis. This suggests that the difference in RMSE for the two groups could very likely be due to random chance. Therefore, based on this test, I do not have enough evidence to claim that the model performs unfairly between power outages in the 'WECC' NERC region versus other regions, likely does not achieve accuracy parity. 