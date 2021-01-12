# sales_data_project : Overveiw
this a data science project about sales data from an individual company where i created a model that predict if a costumer will or will not buy a certain product and at the end i 
productionized it with a flaskAPI 

## steps of the project
  * Created a tool that help companies estimate if a customer will or will not buy a product
  * Engineered features.
  * Optimized Linear, Lasso, and Random Forest Regressors using GridsearchCV to reach the best model. 
  * Built a client facing API using flask.


## Code and Resources Used 
**Python Version:** 3.8.5 
**Packages:** pandas, numpy, sklearn, matplotlib, flask, json, pickle  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```
**Was a great help for me:** https://www.kaggle.com/fabiot21/sales-prediction-using-xgboost-target-audience
**Dataset:** https://www.kaggle.com/mickey1968/individual-company-sales-data   
**Flask Productionization:** https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

## Data Cleaning
after viewing the dataset, i needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:
*     replaced all unkown values in the dataset with null
*     see if there is a dominating values in the columns so we can either delete the column with a lot of null or replace it with a convient value
*     replace categorial columns with numbers







