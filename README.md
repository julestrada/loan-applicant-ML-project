# Credit Default Risk



Our aim is to create a machine learning model that will predict which Home Credit Group credit applicants will likely default on their credit loans, based on application and credit history features. We are using a (expired) competition from Kaggle named 
'Home Credit Default Risk: Can you predict how capable each applicant is of repaying a loan?'. 

Our data was taken from the competition homepage:

https://www.kaggle.com/competitions/home-credit-default-risk/data

The data from the competition was originally 10 csv files totaling 2.7GB. 2 of these were a training and a testing set. We decided we would create our own datasets to train our own model. We decided to use only 3 of the most relevant files (application, bureau, and previous), taking only those columns that we felt were relevant to each application sample: raw data and not derived or encoded features. After one-hot encoding our 42 raw data features, we ended up with 69 features in our training and testing sets, including 1 labeling target column. 

With our ML modeling, we are essentially trying to achieve the same goal as stated by the instructions of the Kaggle challenge. Our team will begin by exploring different machine lerning models to see which approaches work best for solving this kind of problem.

The labeling data is a binary column (0/1) which describes if a loan applicant (per application/sample) had bad repayment history (1) or good repayment history (0). We decided that for our models, we would also stick to a binary classification output scheme. Since the primary goal is to identify risky credit applicants, 1s (risky) are considered the Positive result and 0s (safe) are the Negative result.

Using this same dataset, each member of our team chose to develop their own ML models based on their area of personal interest. Neural Network, Logistic Regression, and Unsupervised learning. We would then discuss and compare our models in order to discern which strategies and features were most effective.




# Binary Classification Neural Network


I chose to use a Neural Network primarily because of the high dimensionality of the data. Our original dataset had 69 columns including the Target data, but after applying RandomForestClassifier (from SKLearn library) to determine Feature Importance, I reduced my dataset to 61 columns of training data, and 1 Target column of labeled outcomes. 

The first run of my model produced a 92% accuracy score! My teammates also got results around 92%. This made me suspicious that perhaps there might be something in our data that was causing us all to reach the same accuracy so fast, even though we were running completely different models. Sure enough, there were zero scores for Precision and Recall metrics - something wasn't right. Checking the TARGET values, I found that a 1-value  only accounted for about 8% of the samples. Our models were all being trained on highly imbalanced data. I tried underfitting at first, and then overfitting the data to balance the 0 and 1 classes. Oversampling worked very well! My model was then able to train well enough to distinguish between good and bad loans. 

I started with a simple NN structure of 1 input layer with 61 units, 1 hidden layer with 32 units, and 1 output layer of only 1 unit for binary classification. Through trial and error, I found that the model needed a bit more complexity, and eventually I added a few hidden layers, including a dropout layer to help prevent overfitting. I tried many different combinations of hyperparameter settings, and eventually settled on a model that was the most reliable. 


# Binary Classification Logistic Regression


For this section I used linear regression, a form of supervised learning, in order to predict if the applicants would pay back their loan.  In order to start this I imported the necessary library Sklearn.  This is the algorithm I would be using to calculate accuracy score.  To prepare my data, I first had to set X and Y values.  We had prepared a dataframe of useful values such as income and credit that we decided would be used as X values.  For the Y value, we used the ‘target’ column.  This column had values as ‘0’ or ‘1’, indicating whether or not these individuals had paid back their loan.  This would be used to train our algorithm.  This leads us to the test_train_split step, where the training took place.  In order to deal with missing values, I decided to use SimpleImputer.  This is a strategy that fills in missing values with the average value for the respective column.  At this point, a logistic regression model was set in place.  The predicted values were  compared to the test label column to receive an accuracy score.  We were left with a 92% accuracy score using this method.  This indicates that with the specific variables chosen to construct our dataframe, combined with the algorithm methods chosen gave us a 92% accuracy in predicting whether or not an applicant would pay back their loan.  


# Unsupervised Learning


I chose to do Unsupervised Machine Learning, both without and with PCA applied. I chose Unsupervised Learning because I was curious as to whether it would uncover some bias in the way Loan Applications are deemed risky. Whether it was banks overlooking a critical factor, or banks overthinking certain factors, There's no way to tell if any of these kinds of models would work without trying.

I first tested my model without PCA to see what it would assign as risky or safe. Using `StandardScaler` and applying inertia tests, I got an elbow curve that appeared really flat. I chose the number of clusters to be 2, because I could quickly look at the results and not guess which predicted cluster value meant loans were risky and which were safe, plus I could easily compare it to the `y` column of the original Dataset (labeled `TARGET`).

This first model without PCA was extremely inaccurate. It only got `24%` of its predictions right, and they were significantly diffeerent from the actual results. Overall, the first model was extremely risk-averse, and it chose to deny loans left, right, and middle.


For the second Model, I created a copy of my dataset and used PCA on it before finding the number of clusters needed. After using PCA, the Elbow Curve was STILL super flat, and it appeared no difference was made. I ultimately decided to use 2 clusters again, both because it matched what I did previously, and it did have the 'sharpest' Second Derivative of the clusters (Basically k=2 was the pointiest). 

The PCA model got `75.5%` of its predictions right, which I felt was perfect for a machine that I wasn't actively training. I used Baye's Theorem to determine how accurate each individual prediction was. Basically I asked this: If my model said a loan was risky (or safe), what was the probability my Model was right on that decision?

In the end, if the PCA Model predicted a loan was risky then there's a `5.4%` chance that it's right, but if the PCA Model predicted a loan was safe then there's a `91.3%` chance that it's right.

Overall, the accuracy of the PCA Model is much better, but it still has a bit of a loan denial trigger-finger.

While I'm proud of what my models did, I would only change two things; One would be to use pyspark instead of VSCode oor Jupyter Notebook, and the other would be to add more examples of risky loans because most of thee loans in the original dataset were safe.
