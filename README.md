<div align="center">
<img width="674" alt="image" src="https://github.com/user-attachments/assets/f0db51ef-e273-491c-9736-a668a38e991e">
</div>


---
<div align="center">
  
  **In a Nutshell🥜:**
  <p>This project focuses on predicting house prices using advanced regression techniques to achieve accurate results. Key methodologies include thorough data preprocessing, feature engineering, and the application of various machine learning models such as LASSO, Ridge Regression, and Elastic-Net. The dataset, provided by a Kaggle competition, contains a range of features describing house characteristics. After extensive hyperparameter tuning and model evaluation, the LASSO model with squareroot-transformed, capped data (data which the outlier is removed by Winsorizing), provided the best performance, with an _RMSE score of 0.157_ on the test set. The project also includes a detailed model evaluation and an analysis of the most influential features.</p>
  
  ➡️<a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview">Competition Website Here</a>⬅️<br>
  
  **Authors** <br>
  <strong>Yixing Wang</strong> 
  <a href="https://www.linkedin.com/in/yixingwang-ai">
      <img src="https://github.com/user-attachments/assets/a0a78f0c-7366-4fd7-8ed8-560731e543a5" width="20" height="20" alt="LinkedIn Logo">
  </a>
  <a href="https://github.com/BruceLeo99"> 
     <img src="https://github.com/user-attachments/assets/624be0d7-36c8-4704-b406-cb4b335e9da7" width="20" height="20" alt="GitHub Logo">
  </a>

  <br>
  
  <strong>Pei Tong</strong>
  <a href="www.linkedin.com/in/p-p-98608a194">
      <img src="https://github.com/user-attachments/assets/a0a78f0c-7366-4fd7-8ed8-560731e543a5" width="20" height="20" alt="LinkedIn Logo">
  </a>
  <a href="https://github.com/PT-VU"> 
     <img src="https://github.com/user-attachments/assets/624be0d7-36c8-4704-b406-cb4b335e9da7" width="20" height="20" alt="GitHub Logo">
  </a>
</div>  

---


<h1>Intro</h1>

**This Repo contains the code and the result of our participation in an indefinite-period Kaggle Project**🏘️📈

###### Project:

- Goal: Use (advanced) regression algorithm to predict the house price. Train the Regression model on training data, and predict the price of each house in test data.
- Training Data: A set of 1460 house infos from Ames Housing dataset , consisting of 79 features (e.g.: House Area, Number of Neighbours, Built Year, etc.) and 1 _Sales Price_ as the label
- Test Data: A set of 1460 house infos, same features, **Unlabelled** (Labels stored on Kaggle and will be evaluated upon submission).
- Evaluation Metric:  Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.
- Competition Time: Indefinite, maximum 10 submissions/day.

###### Our team:

- Team name: Coconut Coffee (Because our idea and journey began from a Café in Amsterdam, with a cup of freshly-made coconut coffee! 🥥)
- Contributors: <a href="https://github.com/BruceLeo99">Yixing</a>, <a href="https://github.com/PT-VU">PT</a>
- Rank: 2715
- **Best RMSE score: 0.15644**
- Last Submission: 05 Sep 2024

  <div align="center">
    <img width="923" src="https://github.com/user-attachments/assets/a556dfab-3987-4dce-9f5c-3a675352ce2b">
  </div>


# Techniques we used

###### Our Work Pipeline

<div align="center">
  <img width="380" src=https://github.com/BruceLeo99/Kaggle-House-Price-Prediction/blob/main/Kaggle%20Project%20Workflow.png>
</div>

#### Exploratory Data Analysis (EDA):
Before starting implementing the algorithm, we spent some time to explore and visualize the data. <br>
We carefully viewed:

- The datatype of each feature
- Density of Missing Values
- Range of Sales Price
- Distributions of each feature and Sales Price
- Outliers of each feature

#### Feature Engineering
After carefully viewing the data, we made decisions on handling the data by:

1. **Fixing Datatype:** Some categorical data (e.g.: Quality label from 1 to 10 and Material Quality from 1-5) were typed as Integers, which might "confuse" the algorithms. We changed these Datatypes to Categorycal (Object in Pandas DataFrame) to ensure that they won't be treated as real values.
2. Imputing Missing Values:
  - Numerical Data: **Forward fill, Backward fill, Iterative Imputing**
  - Categorical Data (e.g.: Street Type, Alley Type, Swimming Pool Type): We treated Categorical Data as 2 types: **Predictable and Non-Predictable**. Categorical Data is predictable if it does not have too many missing values and can be predicted from the data of non-missing individuals; it is non-predictable if it has many missing values.
    - Predictable Categorical Data: **Use the same imputing technique as Numerical Data**
    - Non-predictable: Treat missing values as a separate value, namely "No Value", and fill it in. For example, there were only 7 houses with a swimming pool, and it is not sensible to "assume" that other houses have a swimming pool as well, therefore, we filled in the missing values of swimming pool type as "No Swimming Pool".
3. Data Transforming: Normalizing the data with different scales of each feature: **min-max, "absolut-max, z-score", log,  yeo-johnson,  "squareroot**
4. Removing outliers: Several outlier removal methods were applied:
  - **Remove by 3 IQR**
  - **Remove by 5 IQR**
  - **Winsorization**: remove _x_ lower percentile and _y_ upper percentile
    - During the training, we realized that **removing by 3 and 5 IQR caused significant data loss (nearly 60 percent!) and resulted in poor model performance, hence we applied Winsorization to preserve the data integrity.**
    - Bounds were adjusted by _Experimenting_ and _observation of the outlier range_.
   
#### Algorithms

###### Baseline and dummy model

- Randomly-generated "fake" Sales Price within the range of the Sales Price in training data
- LASSO Regression Model on **unprocessed data**, $\alpha \in [0.0001, 1]$, 5-fold CV, repeat 5 times with train-validation splitting randomness control (_random_state=1_) to ensure that there is no duplication
  - Initially evaluate the RMSE (before submitting to Kaggle and being evaluated by the Kaggle System) of each $\alpha$.

###### Experimental Models:

**Train LASSO and Elastic-net Model with multiple types of hyperparameters, and training data varying in their outlier removal methods, trainsformation and imputing techniques**. <br><br>

**Hyperarameters**
- alpha: $\alpha \in [0.0001, 100]$
- l1 ratio (for Elastic-net only): $l1 \ ratio \in [0,1]$
  
For both LASSO and Elastic-net if evaluated by Grid Search CV:
- max_iter: $max\\_iter \in \[1000,10000]$, number of iterations to compute maximally
- tolerance: 0.0001, 0.01, 0.1, 0.5, permissible threshold difference in Model's covariances in each iteration - stop training when the covariance is smaller than the threshold to prevent overfitting and wasting of computational power.

#### Evaluation and Hyperparameter Tuning:

We firstly used one of the Validation Methods (5-fold CV with 5 repeats or Grid Search CV) to find out the best-reported hyperparameter combination, then used these hyperparameters to run an _initial test_ (random 70-30 train-test split) and score the RMSE, then visualized the **residual plot** to ensure the result is _Homoscedastic_ (no multicollinearity, which might cause poor model performance)

We selected the best configuration of a model and tuned its (hyper)parameters for Kaggle Submissions **locally** in the following steps:

1. Try different (hyper)parameter combinations, transformation methods, and/or winsorization bounds
2. Use CV to find the best (hyper)parameter combinations
3. Select suitable models (the one with a relatively high Validation score) 
4. Feed the test dataset to the trained model and predict the Sales Prices of the houses in the test dataset
5. Submit and evaluate the result on Kaggle
6. Repeat steps 1-5 by readjusting the model configuration (i.e.: trying a different transformation technique, a different range of alpha, a different Winsorization cap, etc.) to aim for a lower RMSE score (a better result).


# Results

###### On Kaggle:

32 submissions were submitted on Kaggle, and we selected 13 **Notable Model Candidates and RMSE Scores**:

<table>
  <tr>
    <th colspan="3">Model</th>
    <th colspan="3">Data Engineering</th>
    <th colspan="4">Hyperparameters</th>
    <th colspan="2">      </th>
  </tr>

  <tr>
    <td><b>Model No.</b></td>
    <td><b>Algorithm</b></td>
    <td><b>CV method</b></td>
    <td><b>Transformation Method</b></td>
    <td><b>Imputation</b></td>
    <td><b>Outlier Removal</b></td>
    <td><b>Alpha</b></td>
    <td><b>l1 Ratio</b></td>
    <td><b>Max Iter</b></td>
    <td><b>tolerance</b></td>
    <td><b>RMSE Score</b></td>
    <td><b>Note</b></td>
  </tr>

  <tr>
    <td>1</td>
    <td>Dummy Result</td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td> - </td>
    <td>1.33076</td>
    <td>Randomly Generated Sales Price</td>
  </tr>

  <tr>
    <td>2</td>
    <td>LASSO</td>
    <td>5-fold</td>
    <td>yeo-johnson</td>
    <td>Iterative Imputer</td>
    <td>Winsorization, 90% upper percentile</td>
    <td>1</td>
    <td>  -  </td>
    <td>  -  </td>
    <td> -  </td>
    <td> 0.78588  </td>
    <td> -  </td>
  </tr>

  <tr>
    <td>3</td>
    <td>LASSO</td>
    <td>5-fold</td>
    <td>absolute max</td>
    <td>Iterative Imputer</td>
    <td>Winsorization, 95% upper percentile and 1% lower percentile</td>
    <td>1</td>
    <td>  -  </td>
    <td>  -  </td>
    <td> -  </td>
    <td> 0.28879  </td>
    <td> -  </td>
  </tr>

  <tr>
    <td>4</td>
    <td>Elastic Net</td>
    <td>Grid Search</td>
    <td>absolute max</td>
    <td>Iterative Imputer</td>
    <td>Winsorization, 96% upper percentile</td>
    <td>100</td>
    <td>  0.4  </td>
    <td>  1000  </td>
    <td> 0.01  </td>
    <td> 0.41605  </td>
    <td> -  </td>
  </tr>

  <tr>
    <td>5</td>
    <td>Elastic Net</td>
    <td>Grid Search</td>
    <td>absolute max</td>
    <td>Iterative Imputer</td>
    <td>Winsorization, 96% upper percentile</td>
    <td>0.00028</td>
    <td>  0.4  </td>
    <td>  1000  </td>
    <td> 0.01  </td>
    <td> 0.16410  </td>
    <td> -  </td>
  </tr>

  <tr>
    <td>6</td>
    <td>LASSO</td>
    <td>5-fold</td>
    <td>absolute max</td>
    <td>Iterative Imputer</td>
    <td>5IQR</td>
    <td>1</td>
    <td>  -  </td>
    <td>  -  </td>
    <td> -  </td>
    <td> 1.21979  </td>
    <td> -  </td>
  </tr>

  <tr>
    <td>7</td>
    <td>LASSO</td>
    <td>5-fold</td>
    <td>log</td>
    <td>Iterative Imputer</td>
    <td>Winsorization, 95% upper percentile and 1% lower percentile</td>
    <td>15</td>
    <td>  -  </td>
    <td>  -  </td>
    <td> -  </td>
    <td> 0.33224  </td>
    <td> -  </td>
  </tr>


<tr>
    <td>8</td>
    <td>Elastic Net</td>
    <td>Grid Search</td>
    <td>log</td>
    <td>Iterative Imputer</td>
    <td>Winsorization, 96% upper percentile and 1% lower percentile</td>
    <td>0.017</td>
    <td>  0.8  </td>
    <td>  1000  </td>
    <td> 0.01  </td>
    <td> 0.21237  </td>
    <td> -  </td>
  </tr>
  
  <tr>
    <td>9</td>
    <td>LASSO</td>
    <td>5-fold</td>
    <td>log</td>
    <td>Iterative Imputer</td>
    <td>Winsorization, 95% upper percentile and 1% lower percentile</td>
    <td>10</td>
    <td>  -  </td>
    <td>  -  </td>
    <td> -  </td>
    <td> 0.35452  </td>
    <td> -  </td>
  </tr>

<tr>
    <td>10</td>
    <td>LASSO</td>
    <td>5-fold</td>
    <td>squareroot</td>
    <td>Iterative Imputer</td>
    <td>Winsorization, 95% upper percentile</td>
    <td>1</td>
    <td>  -  </td>
    <td>  -  </td>
    <td> -  </td>
    <td> 0.17566 </td>
    <td> -  </td>
  </tr>

<tr>
    <td>11</td>
    <td>Elastic Net</td>
    <td>Grid Search</td>
    <td>squareroot</td>
    <td>Iterative Imputer</td>
    <td>Winsorization, 98% upper percentile</td>
    <td>0.129</td>
    <td>  0.9  </td>
    <td>  1000  </td>
    <td> 0.0001  </td>
    <td> 0.1633 </td>
    <td> Best Result with Elastic-Net </td>
  </tr>

  <tr>
    <td>12</td>
    <td>LASSO</td>
    <td>5-fold</td>
    <td>squareroot</td>
    <td>Iterative Imputer</td>
    <td>Winsorization, 95% upper percentile</td>
    <td>100</td>
    <td>  -  </td>
    <td>  -  </td>
    <td> -  </td>
    <td> 0.15644 </td>
    <td> Best Result among all </td>
  </tr>

  <tr>
    <td>13</td>
    <td>LASSO</td>
    <td>5-fold</td>
    <td>squareroot</td>
    <td>Iterative Imputer</td>
    <td>Winsorization, 95% upper percentile</td>
    <td>15</td>
    <td>  -  </td>
    <td>  -  </td>
    <td> -  </td>
    <td> 0.15697 </td>
    <td> Second Best Result </td>
  </tr>

</table>

###### Barplot of RMSE Scores
<div align="center">
  <img src="https://github.com/BruceLeo99/Kaggle-House-Price-Prediction/blob/main/RMSE%20Barplot.png">
</div>

###### Residual plots of some of the submissions (for demonstration purposes):


<table>
  <tr>
    <td>
      <figure> 
        <img src="https://github.com/user-attachments/assets/cd8a90e3-dd6b-4966-976c-8a6bb920fa24" alt="Figure 1">
        <figcaption>Figure 1: Best Result (LASSO model, sqrt-transformed data, alpha=15, Winsorized (upper bound=0.95, lower bound=0))</figcaption>
      </figure>
    </td>
    <td>
      <figure> 
        <img src="https://github.com/user-attachments/assets/364114be-d3d5-4349-af13-4568ba0de35c" alt="Figure 2">
        <figcaption>Figure 2: Second Best Result (LASSO model, sqrt-root transformed data, alpha=100, Winsorized (upper bound=0.98, lower bound=0))</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure> 
        <img src="https://github.com/user-attachments/assets/d39b5225-54d0-4147-975d-5ed6debf6f74" alt="Figure 3">
        <figcaption>Figure 3: Best result of Elastic-net Model (sqrt-root transformed data, alpha=0.129, l1_ratio=0.9, max_iter=1000, tol=0.0001, Winsorized (upper bound=0.95, lower bound=0))</figcaption>
      </figure>
    </td>
    <td>
      <figure> 
        <img src="https://github.com/user-attachments/assets/372dd54e-8753-4cbc-9f76-8775a6ab9e5c" alt="Figure 4">
        <figcaption>Figure 4: Elastic-net Model (abx-max-transformed data, alpha=100, l1_ratio=0.4, max_iter=1000, tol=0.01, Outliers at IQR=5 removed)</figcaption>
      </figure>
    </td>
  </tr>
  <tr>
    <td>
      <figure> 
        <img src="https://github.com/user-attachments/assets/4321211b-f58f-480e-b71a-c9f40ba84115" alt="Figure 5">
        <figcaption>Figure 5: LASSO model, log transformed data, alpha=15, Winsorized (upper bound=0.95, lower bound=0.01)</figcaption>
      </figure>
    </td>
    <td>
      <figure> 
        <img src="https://github.com/user-attachments/assets/8249fcfb-db1e-4d32-b9fd-e6df21ad9f77" alt="Figure 6">
        <figcaption>Figure 6: LASSO model, abs-max transformed data, alpha=1, Outliers at IQR=5 removed</figcaption>
      </figure>
    </td>
  </tr>
</table>

**This step is crucial to analyze the validity of the model. For instance, we can see clearly that the model of _Figure 4_ is not homoscedastic, meaning that this model will make biased result.**

#### Key Discoveries:

- **Feature Engineering has a more significant influence on the prediction accuracy than the model itself**
- **Traditional Outlier Removal methods cause massive data loss, which causes poor model performance, setting up a removal cap (Winsorization) and testing for a suitable cap threshold is necessary**
  - Best removal cap we found: 95%-98% upper, 0%-1% lower, **any cap out of this scope would cause significant data loss.**
- In this task, Iterative Imputer was tested the most reliable when filling in missing values
- In this task, Normalizing data by square-root significantly improves the model performance, the second-best transformation option was absolute-max, then the log.
- **Best Alpha values and l1 ratio (for elastic-net models)** depend on the other factor and differ in models: there is no rule-of-thumb for this
- There is no significant difference between the performance of the LASSO and the Elastic-Net model.





