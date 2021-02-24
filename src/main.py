import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
from sklearn import metrics
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn import tree
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.neighbors import KNeighborsClassifier
import warnings
import joblib

warnings.filterwarnings('ignore')


# Reading file
reviews = pd.read_csv('C:\\Users\\Asus\\Desktop\\AmazonAnalysis\\Musical_instruments_reviews.csv')
reviews.head()

# Data info
print(reviews.shape)
print()
print(reviews.info())
print()
print(reviews.describe())
print()

# Checking NaN values
reviews.isna().sum()

# Dropping NaN values, since we have more than 10k of observations
reviews.dropna(inplace=True)


class Sentiment:
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


# Creating Columns for the year, month and day
def get_year(reviewTime):
    return reviewTime.split(",")[1].strip(" ")


# Function to create
def get_day(reviewTime):
    return reviewTime.split(" ")[1].strip(",")


# Sentiment Score
def get_sentiment(score):
    if score <= 2:
        return 'NEGATIVE'
    elif score == 3:
        return 'NEUTRAL'
    else:
        return 'POSITIVE'


def text_clean(text):
    text = re.sub(r'@[A-Za-z0-9]', '', text)  # Removes @mentions
    text = re.sub(r'#', '', text)  # Removing the Hashtag
    text = re.sub(r'RT[\s]', '', text)  # Removing RT
    text = re.sub(r'_', '', text)  # Removing RT
    return text


def balanced_sentiment(score):
    if score > 3:
        return 'POSITIVE'
    else:
        return 'NEGATIVE'


# Creating Three columns spliiting by year, month and day
reviews['count'] = 1
reviews['year'] = reviews.reviewTime.apply(get_year)
reviews['month'] = reviews['reviewTime'].str[:2]
reviews['day'] = reviews.reviewTime.apply(get_day)
reviews['Sentiment'] = reviews.overall.apply(get_sentiment)

# Converting Column data type from object to integer
reviews['year'] = reviews['year'].astype('int64')
reviews['month'] = reviews['month'].astype('int64')
reviews['day'] = reviews['day'].astype('int64')

# Top 10 reviewers
top_ten = reviews['reviewerName'].value_counts()
top_ten = top_ten.head(10)
top_ten = pd.DataFrame(top_ten)
top_ten.columns = ['NumberOfReviews']

top_ten.plot(color='darkred', marker='v', linestyle='-')
plt.xticks(rotation=90)
plt.title('TOP 10 REVIEWERS')
plt.xlabel('Reviewers')
plt.ylabel('Nº of Reviews')
plt.show()

# Product Valuation
sns.countplot(data=reviews, x='overall')
plt.ylabel('Overall Count')
plt.title('PRODUCT OVERALL')
plt.show()


# Grouping By  year
year_review = reviews.groupby(by='year').sum()
year_review = year_review.drop(columns=['month', 'day', 'overall', 'unixReviewTime'])
year_review.columns = ['trend']
plt.figure(figsize=(14, 10))
year_review.plot(color='darkred', linestyle='-', marker='o')
plt.xticks(rotation=90)
plt.ylabel('Reviews Count')
plt.title('REVIEWS PER YEAR TREND')
plt.show()


# Grouping By  Month
month_review = reviews.groupby(by='month').sum()
month_review = month_review.drop(columns=['year', 'day', 'overall', 'unixReviewTime'])
month_review.columns = ['trend']
plt.figure(figsize=(14, 10))
month_review.plot(color='orange', linestyle='-', marker='o')
plt.xticks(rotation=90)
plt.ylabel('Reviews Count')
plt.title('REVIEWS PER MONTH TREND')
plt.show()

# Trend by day of the month
day = reviews
day = day.sort_values(by=['day'], ascending=True)
day = day.groupby(by=['day']).sum()
day = day.drop(columns=['overall', 'unixReviewTime', 'year', 'month'])
day.columns = ['trend']
# Line plot
plt.figure(figsize=(10, 8))
sns.lineplot(data=day, x=day.index, y='trend', color='magenta')
plt.xticks(day.index)
plt.ylabel('Review Count')
plt.xlabel('Days of the Month')
plt.title('REVIEW TREND PER DAYS OF THE MONTHS')
plt.show()

# Reviews Sentiment
size_complete = reviews['Sentiment'].value_counts()
colors_complete = ['green', 'yellow', 'red']
labels_complete = "Positive", "Neutral", "Negative"
explode = [0, 0.01, 0.01]

# (0,0) is to create the circle and 0.5 for the width of the circle, white is for the center of the circle
my_circle_complete = plt.Circle((0, 0), 0.5, color='white')

plt.figure(figsize=(8, 6))
plt.pie(size_complete, colors=colors_complete, labels=labels_complete, shadow=False, explode=explode, autopct='%.2f%%')
plt.title('PROPORTIONS BY SENTIMENT', fontsize=15)
p = plt.gcf()
p.gca().add_artist(my_circle_complete)
plt.legend()
plt.show()

# Creating the inputa Array of values and the target
X = reviews['reviewText']
y = reviews['Sentiment']
X = pd.DataFrame(X)
y = pd.DataFrame(y)


# Let's clean the text of the reviews
X['reviewText'] = X['reviewText'].apply(text_clean)


# TF-IDF
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=20
)
tfidf.fit(X.reviewText)
text = tfidf.transform(X.reviewText)

# Data Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(text, y, test_size=0.25, random_state=0)

# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
text_classifier.fit(X_train, y_train)

# Creating Model
predictions = text_classifier.predict(X_test)

# Model Accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

# Model check
test_set = ["very good", 'this guitar rocks', "I didn't expect this cable to be so thin", 'bad bad bad bad no way']
new_test = tfidf.transform(test_set)

text_classifier.predict(new_test)


# Create a new dataset with balanced data
undersample = reviews.sort_values(by=['Sentiment'], ascending=True)
# Create a dataframe with only positive sentiment
positive = reviews[reviews['Sentiment'] == 'POSITIVE']
# Extracting only 800 rows to balance POSITIVE CLASS
positive = positive.iloc[0:700, ]
# Drop the positive sentiment from the undersample dataframe
undersample = undersample[undersample['Sentiment'] != 'POSITIVE']
# Concat dataframe
undersample = pd.concat([undersample, positive])
# Shuffle Rows to randomize
undersample = undersample.sample(frac=1)

# Data Prep
X_undersample = undersample['reviewText']
y_undersample = undersample['Sentiment']
X_undersample = pd.DataFrame(X_undersample)
y_undersample = pd.DataFrame(y_undersample)


# Let's clean the text of the reviews
X_undersample['reviewText'] = X_undersample['reviewText'].apply(text_clean)


# Fitting TF-IDF
tfidf_undersample = TfidfVectorizer(
    stop_words='english',
    max_features=20
)
tfidf_undersample.fit(X_undersample.reviewText)
corpus = tfidf_undersample.transform(X_undersample.reviewText)

# Data Split
X_train, X_test, y_train, y_test = train_test_split(corpus, y_undersample, test_size=0.3, random_state=0)

# RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=10, random_state=0)
text_classifier.fit(X_train, y_train)

# Creating Model
predictions = text_classifier.predict(X_test)

# Model Accuracy
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

# Decision Tree
tree_classifier = tree.DecisionTreeClassifier()
tree_text_classifier = tree_classifier.fit(X_train, y_train)

tree_predictions = tree_text_classifier.predict(X_test)

# Model Accuracy
print(confusion_matrix(y_test, tree_predictions))
print(classification_report(y_test, tree_predictions))
print(accuracy_score(y_test, tree_predictions))

model_test = ['bad guitar', 'no words', 'very good guitar', 'the suction cup does not work']
model_test = tfidf_undersample.transform(model_test)

tree_text_classifier.predict(model_test)

# Model Check
model_test = ['bad guitar', 'no words', 'very good guitar', 'the suction cup does not work']
model_test = tfidf_undersample.transform(model_test)

print(text_classifier.predict(model_test))


# Create a dataframe based on the original one
balanced = reviews
# Drop Sentiment Column to create a new one with onle positive and negative columns
balanced = balanced.drop(columns=['Sentiment'])
# Applying function
balanced['Sentiment'] = balanced['overall'].apply(balanced_sentiment)


# Let´s visualize the Sentiments now
sns.countplot(data=balanced, x='Sentiment', palette='Set1')
plt.title('NEGATIE VS POSITIVE SENTIMENT FEEDBACKS')
plt.ylabel('Feedback Count')
plt.show()

# Around 1.2k are negative feedbacks, so let´s undersample again the positive feedbacks, since we don´t have
# any additional data

# Creating a dataframe with only  positive feedbacks
balanced_positive = balanced[balanced['Sentiment'] == 'POSITIVE']
# Let`s extract 1300 observations
balanced_positive = balanced_positive.iloc[0:1300, ]
# Let´s drop negative Sentiments from the original dataframe
balanced = balanced[balanced['Sentiment'] != 'POSITIVE']
# Let's concat both dataset and compare target variable
balanced = pd.concat([balanced, balanced_positive])

sns.countplot(data=balanced, x='Sentiment', palette='Set1')
plt.title('NEGATIVE VS POSITIVE SENTIMENT FEEDBACKS')
plt.ylabel('Feedback Count')
plt.show()

# Data Prep
X_balanced = balanced['reviewText']
y_balanced = balanced['Sentiment']
X_balanced = pd.DataFrame(X_balanced)
y_balanced = pd.DataFrame(y_balanced)


# Let's clean the text of the reviews
X_balanced['reviewText'] = X_balanced['reviewText'].apply(text_clean)

# Fitting TF-IDF
tfidf_balance = TfidfVectorizer(
    stop_words='english',
    max_features=20
)
tfidf_balance.fit(X_balanced.reviewText)
texts = tfidf_balance.transform(X_balanced.reviewText)

# Data Split
X_train, X_test, y_train, y_test = train_test_split(texts, y_balanced, test_size=0.3, random_state=0)

# RandomForestClassifier
balanced_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
balanced_classifier.fit(X_train, y_train)

# Creating Model
predictions_balanced = balanced_classifier.predict(X_test)

# Model Accuracy

print(confusion_matrix(y_test, predictions_balanced))
print(classification_report(y_test, predictions_balanced))
print(accuracy_score(y_test, predictions_balanced))

# Model Check
model_test = ['nice guitar', 'bad', 'very good guitar', 'not fine']
model_test = tfidf_balance.transform(model_test)

print(balanced_classifier.predict(model_test))

# K-nearest Neighbor classifier, let´s use 2 neighbors since we only have to classes
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)

knn_prediction = knn.predict(X_test)
print(confusion_matrix(y_test, knn_prediction))
print(classification_report(y_test, knn_prediction))
print(accuracy_score(y_test, knn_prediction))

# try K=1 through K=25 and record testing accuracy
k_range = range(1, 20)

# We can create Python dictionary using [] or dict()
scores = []

# We use a loop through the range 1 to 20
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)


plt.plot(k_range, scores, marker='s', markerfacecolor='black')
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

# Model Check
model_test = ['nice guitar, and excellent buy', 'bad cable, not plugged in', 'very good guitar',
              'not fine, but it works']
model_test = tfidf_balance.transform(model_test)

print(knn.predict(model_test))

# Saving Model
joblib.dump(balanced_classifier, 'amazon_sentiment_recommender.joblib')
