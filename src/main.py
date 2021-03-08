import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import pickle
import random
import os
import re
import nltk  

from nltk.corpus import stopwords  
from imblearn.over_sampling import SMOTE
from dtreeviz.trees import dtreeviz
from IPython.display import Image

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
warnings.filterwarnings('ignore')
nltk.download('stopwords') 

#Reading file\
reviews = pd.read_csv('C:\\Users\\Asus\\Desktop\\Amazon-Musical-Instruments-Reviews-main\data\\Musical_instruments_reviews.csv')
reviews.head()

#Data info
print(reviews.shape)
print()
print(reviews.info())
print()
print(reviews.describe())
print()

#Checking Reviews
reviews.isna().sum()

#Dropping NaN values, since we have more than 10k of observations
reviews.dropna(inplace=True)


class Sentiment:
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

#Creating Columns for the year, month and day
def get_year(reviewTime):
    return reviewTime.split(",")[1].strip(" ")

#Function to create 
def get_day(reviewTime):
    return reviewTime.split(" ")[1].strip(",")

#Sentiment Score   
def get_sentiment(score):
    if score <= 2:
        return 'NEGATIVE'
    elif score == 3:
        return 'NEUTRAL'
    else: 
        return 'POSITIVE'
    
def text_clean(text):
    text = str(text).lower()
    text = re.sub(r'@[A-Za-z0-9]','',text)#Removes @mentions
    text = re.sub(r'#','',text)#Removing the Hashtag
    text = re.sub(r'RT[\s]','',text)#Removing RT
    text = re.sub(r'_','',text)#Removing RT
    return text


#Creating Three columns spliiting by year, month and day
reviews['count'] = 1
reviews['year'] = reviews.reviewTime.apply(get_year)
reviews['month'] =  reviews['reviewTime'].str[:2]
reviews['day'] = reviews.reviewTime.apply(get_day)
reviews['Sentiment'] = reviews.overall.apply(get_sentiment)

#Converting Column data type from object to integer
reviews['year'] = reviews['year'].astype('int64')
reviews['month'] = reviews['month'].astype('int64')
reviews['day'] = reviews['day'].astype('int64')

#Cleaning Text
reviews['reviewText'] = reviews['reviewText'].apply(text_clean)


#Top 10 reviewers
top_ten = reviews['reviewerName'].value_counts()
top_ten = top_ten.head(10)
top_ten = pd.DataFrame(top_ten)
top_ten.columns = ['NumberOfReviews']

top_ten.plot(color='darkred',marker='v',linestyle='-')
plt.xticks(rotation=90)
plt.title('TOP 10 REVIEWERS')
plt.grid()
plt.xlabel('Reviewers')
plt.ylabel('Nº of Reviews')
plt.show()


#Product Valuation
sns.countplot(data=reviews, x='overall')
plt.ylabel('Overall Count')
plt.title('PRODUCT OVERALL')
plt.grid()
plt.show()

#Grouping By  year
year_review = reviews.groupby(by='year').sum()
year_review = year_review.drop(columns=['month','day','overall','unixReviewTime'])
year_review.columns = ['trend']

#Creating index
#Creating index for the plot
year_index = year_review.index.unique()

#Plot
plt.figure(figsize=(14,10))
year_review.plot(color='darkred',linestyle='-',marker='o')
plt.xticks(year_index,rotation=90)
plt.grid()
plt.ylabel('Reviews Count')
plt.title('REVIEWS PER YEAR TREND')
plt.show()

#Grouping By  Month
month_review = reviews.groupby(by='month').sum()
month_review = month_review.drop(columns=['year','day','overall','unixReviewTime'])
month_review.columns = ['trend']

#Creating index for the plot
month_index = month_review.index.unique()

#Creating plot
plt.figure(figsize=(14,10))
month_review.plot(color='orange',linestyle='-',marker='o')
plt.xticks(month_index,rotation=90)
plt.grid()
plt.ylabel('Reviews Count')
plt.title('REVIEWS PER MONTH TREND')
plt.show()

# Trend by day of the month
day = reviews
day = day.sort_values(by=['day'],ascending=True)
day =  day.groupby(by=['day']).sum()
day = day.drop(columns=['overall', 'unixReviewTime','year','month'])
day.columns = ['trend']

# Line plot
plt.figure(figsize=(10,8))
sns.lineplot(data=day, x=day.index, y='trend',color='magenta')
plt.xticks(day.index)
plt.grid()
plt.ylabel('Review Count')
plt.xlabel('Days of the Month')
plt.title('REVIEW TREND PER DAYS OF THE MONTHS')
plt.show()

#Reviews Sentiment
size_complete = reviews['Sentiment'].value_counts()
colors_complete = ['lightgreen', 'yellow','darkred']
labels_complete = "Positive", "Neutral","Negative"
explode = [0, 0.01,0.01]

#(0,0) is to create the circle and 0.5 for the width of the circle, white is for the center of the circle
my_circle_complete = plt.Circle((0, 0), 0.5, color = 'white')

plt.figure(figsize=(8,6))
plt.pie(size_complete, colors = colors_complete, labels = labels_complete, shadow = False, explode = explode, autopct = '%.2f%%')
plt.title('PROPORTIONS BY SENTIMENT', fontsize = 15)
p = plt.gcf()
p.gca().add_artist(my_circle_complete)
plt.legend()
plt.show()

# Creating a column to see the len of each message
reviews['Len'] = reviews['reviewText'].apply(lambda x: len(x.split(' ')))

# Senteces Distribution
unique = reviews['Len'].unique()
plt.figure(figsize=(20,8))
sns.distplot(reviews['Len'],color='green')
plt.title('MESSEAGES LEN DISTRIBUTION')
plt.xlabel('Number of words per message')
plt.grid()
plt.xticks(rotation=90)
plt.show()

# Creating random seed 
seed = random.seed(42)

# Creating the inputa Array of values and the target
X = reviews['reviewText']
y = reviews['Sentiment']


# Fitting TF-IDF algorithm
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Plotting Target Class
pos_neg_color = ['lightgreen','gray','darkred']
sns.countplot(data=reviews, x='Sentiment',palette= pos_neg_color)
plt.title('POSITIVE NEUTRAL & NEGATIVE FEEDBACKS', size=18)
plt.ylabel('Feedback Count')
plt.grid()
plt.show()


# we are facing an umbalanced target variable. Let´s apply **SMOTE**. This stands for Synthetic Minority Oversampling Technique, or SMOTE for short.
# Applying Smote
smote = SMOTE(random_state= seed)
X_smote, y_smote = smote.fit_resample(X,y)

# Counting Target Variable
y_smote.value_counts()


# Traind and test split
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size= 0.3, random_state= seed)

print('X_train shape ', X_train.shape)
print('y_train shape ', y_train.shape)
print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)


#Creating Classifier
logistic_regression = LogisticRegressionCV(random_state= seed, verbose= 1) 

#Fitting Model
logistic_regression.fit(X_train,y_train)

#Prediction
prediction = logistic_regression.predict(X_test)

#Model Accuracy
print('LOGISTIC REGRESSION')
plot_confusion_matrix(logistic_regression, X_test, y_test)  
plt.show()

print()
print('Classification Report')
print(classification_report(y_test, prediction))
print('Model Accuracy: ',accuracy_score(y_test,prediction))


# Checking Predictions with random reviews
test_set = ['very bad cable, didn´t expected it had a blue light, and also to expensive', #Negative Comment
            'this really rocks!, product was like i expected, power cable is perfect', # Positive Comment
            'wasn´t expecting Fender guitar, but is ok, next time i will check the comments'] # Neutral Comment  

# fitting TF-IDF
new_test = vectorizer.transform(test_set)
          
print('Logistic Regression prediction',logistic_regression.predict(new_test))

#Creating classifier
random_forest_classifier = RandomForestClassifier(random_state=seed, n_estimators=200,verbose=1)
random_forest_classifier.fit(X_train, y_train)

# Prediction
random_prediction = random_forest_classifier.predict(X_test)

# Model Accuracy
print('RANDOM FOREST CLASSIFIER')
plot_confusion_matrix(random_forest_classifier, X_test, y_test)  
plt.show() 

print()
print('Classification Report')
print(classification_report(y_test, random_prediction))
print('Model Accuracy: ',accuracy_score(y_test, random_prediction))

# Checking Predictions with random reviews
test_set = ['very bad cable, didn´t expected it had a blue light, and also to expensive', #Negative Comment
            'this really rocks!, product was like i expected, power cable is perfect', # Positive Comment
            'wasn´t expecting Fender guitar, but is ok, next time i will check the comments'] # Neutral Comment  

# fitting TF-IDF
new_test = vectorizer.transform(test_set)
          
print('Random forest prediction',random_forest_classifier.predict(new_test))
# It can be seen that our random forest model has not been very effective, since all the texts with which we have tested have been recognized as positive. For what our model has a bias towards positive feedback, the SMOT has not worked correctly. 

#Saving Logistic Regression Model.
filename = 'logistic_sentiment_model.pkl'
pickle.dump(logistic_regression, open(filename, 'wb'))

