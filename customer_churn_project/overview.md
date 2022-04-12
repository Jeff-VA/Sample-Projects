### Customer Churn Prediction

In most companies that provide a recurring service, the phenomenon of customer attrition is a pervasive issue. If this attrition or “churn” can be predicted based on past and current customer information, then maybe the phenomenon can be addressed and avoided. For example, if a company knows which selections of services tend to be more significantly correlated with customer churn, then perhaps that company can target a selection of services that will reduce the risk of churn for future and current customers.

In [this notebook](https://jeff-va.github.io/Sample-Projects/customer_churn_project/customer_churn_prediction.ipynb), a sample dataset with information about 10,000 customers that have either contributed to churn or not is programmatically used to create logistic regression model that predicts customer churn.

First, the data are prepared through a series of cleaning steps including the numerical transformation of variables, variable screening, and visual inspection. 

Second, a logistic regression model is iteratively reduced of input variables until a most parsimonious or “simple” model is found. A confusion matrix of this model’s predictive accuracy is displayed below:

![reduced model confusion matrix](https://jeff-va.github.io/Sample-Projects/customer_churn_project/confusion_matrix.png)

The resulting logistic regression model from this analysis model successfully predicts customer churn with a predictive accuracy of about 89%. With this code, an analytics pipeline may be easily modularized, recursively reused, and applied to a variety of data. A similar pipeline might also be used to predict other types of binary phenomena in a variety of datasets with little adjustment. For example, a logistic regression model with variable reduction might predict whether a specific customer is likely to buy a certain product or not.

For a detailed report that explains the [source code](customer_churn_prediction.ipynb) in its entirety, [click here](https://jeff-va.github.io/Sample-Projects/customer_churn_project/customer_churn_prediction_report.pdf).
