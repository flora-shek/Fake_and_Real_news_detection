!pip install pydrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

id = "1rseY1d16ZtKLDV13JgI6jH6ne3kqmT9B"

imported = drive.CreateFile({'id':id})
imported.GetContentFile('Fake.csv')

id ="1XaiL21qWBJjZGVm85__KuhBy_a0Dpp6y"

imported = drive.CreateFile({'id':id})
imported.GetContentFile('True.csv')

# Data Cleaning and Visualization

import pandas as pd

df_true_news = pd.read_csv('True.csv')

df_fake_news = pd.read_csv('Fake.csv')

df_fake_news.head(20)

df_true_news.head(20)

df_fake_news.count()

df_true_news.count()

# find missing data if at any cell
def find_missing_vals(data):
  total = len(data)
  for column in data.columns:
    if data[column].isna().sum() !=0:
      print("{} has : {:,} ({:.2}%) missing values ".format(column , data[column].isna().sum(),(data[column].isna().sum()/total)*100))
    else:
      print("{} has no missing value".format(column))
  print("\n Missing Value Summary\n{}".format("-"*35))
  print("\n ndf_db\n".format("-"*15))
  print(data.isnull().sum(axis=0))
  
  # find duplicate data if it has
 def remove_duplicates(data):
   print("\n Cleaning Summary\n{}".format("-"*35))
   size_before = len(data)
   data.drop_duplicates(subset = None , keep ="first", inplace = True )
   size_after = len(data)
   print("...removed {} duplicate rows in db data".format(size_before-size_after))
    
 find_missing_vals(df_fake_news)
 
 find_missing_vals(df_true_news)
 
 remove_duplicates(df_fake_news)
 
 remove_duplicates(df_true_news)
 
 df_merged = pd.merge(df_fake_news,df_true_news ,how='outer')
 
import seaborn as sn
import matplotlib.pyplot as plt
sn.set(style ="ticks", color_codes = True)

fig_dims = (20 , 4.8)
fig,ax = plt.subplots(figsize=fig_dims)
sn.countplot(df_merged['subject'], ax=ax , data = df_merged);

# Data Labelling And Feature Extraction

df_fake_news['label']=0
df_true_news['label']=1

df_train = pd.merge(df_fake_news,df_true_news ,how='outer')

!pip install sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import string

import nltk
nltk.download('stopwords')

def text_process(text):
  no_punctuation =[char for char in text if char not in string.punctuation]
  no_punctuation = ''.join(no_punctuation)
  return [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]
      
# Model Creation And Training

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df_train['title'] , df_train['label'] , test_size=0.2 , random_state=42)

#Deep learning Multi-perception neural network binary classifier

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

news_classifier = Pipeline([('vectorizer', CountVectorizer(analyzer= text_process)),('tfidf', TfidfTransformer()),('classifier',MLPClassifier(solver='adam',activation='tanh',random_state=1, max_iter=200, early_stopping= True))])

from nltk.corpus import stopwords

news_classifier.fit(X_train, y_train)

# Model Evaluation

predicted =  news_classifier.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(predicted,y_test))

# Saving and Downloading the Model

from sklearn.externals import joblib

joblib.dump(news_classifier , 'model.pkl')

from googleapiclient.discovery import build
drive_service = build('drive' , 'v3')

from googleapiclient.http import MediaFileUpload

file_metadata ={'name':'model.pkl'}

file_metadata ={'mimeType':'text/plain'}

media = MediaFileUpload('model.pkl', mimetype='text/plain', resumable=True)

created = drive_service.files().create(body = file_metadata , media_body = media , fields='id').execute()

print('File ID: {}'.format(created.get('id')))


# Model Deployment

news_title = ['Man has landed on the Mars']
prediction = news_classifier.predict(news_title)
print(prediction)

# NOTE: if the output is "0" and "1" is "No" and "Yes" respectively.









































































































































          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
























 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  













