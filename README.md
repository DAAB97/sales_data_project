# sales_data_project : Overveiw

this a data science project about sales data from an individual company where i created a model that predict if a costumer will or will not buy a certain product and at the end i productionized it with a flaskAPI 



## steps of the project
  * Viewing the dataset
  * Cleaning 
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

![alt text](https://github.com/DAAB97/sales_data_project/blob/master/agePlot.png "age of customers")
![alt text](https://github.com/DAAB97/sales_data_project/blob/master/childPlot.png "having a child")
![alt text](https://github.com/DAAB97/sales_data_project/blob/master/house_ownerPlot.png "owner or renter of a house")
![alt text](https://github.com/DAAB97/sales_data_project/blob/master/marriagePlot.png "social state of a customer")



## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.   

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret.   

I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit. 


## Productionization 
In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the TDS tutorial in the reference section above. The API endpoint takes in a request with a list of values and returns an estimated number between 0 and 1 which means a customer is gonna highly buy the product if the number is >= 0.75 . 









