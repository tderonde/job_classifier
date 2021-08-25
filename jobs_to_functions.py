import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

# importing random forest classifier
from sklearn.ensemble import RandomForestClassifier

# Spliting arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split

# metrics are used to find accuracy or error
from sklearn import metrics

df_jobs = pd.read_excel('./job_titles.xlsx', sheet_name='jobs', index_col='job_title')
df_functions = pd.read_excel('./job_titles.xlsx', sheet_name='functional_groups', index_col='functional_group_code')
df_levels = pd.read_excel('./job_titles.xlsx', sheet_name='management_levels', index_col='management_level_code')

job_titles = list(df_jobs.index)
job_functions = list(df_jobs['functional_group_code'])
job_levels = list(df_jobs['management_level_code'])

# create train and test sets
X_train, X_test, y_train, y_test = train_test_split(job_titles, job_functions, random_state=42)

stop_list = ['sr', 'senior', 'staff', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'exempt']

# Create a Vectorizer Object
vectorizer = CountVectorizer(stop_words=stop_list, ngram_range=(1,2))
  
vectorizer.fit(X_train)
  
# Encode the train and test Documents
X_train_vector = vectorizer.transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# creating a RF classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train_vector, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test_vector)

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

df_results = pd.DataFrame({'job': X_test, 'predicted': y_pred, 'actual': y_test})
df_results = df_results.merge(df_functions, how='left', left_on='predicted', right_index=True)
df_results.rename(columns={'functional_group': 'functional_group_pred'}, inplace=True)
df_results = df_results.merge(df_functions, how='left', left_on='actual', right_index=True)
df_results.rename(columns={'functional_group': 'functional_group_act'}, inplace=True)

print(df_results)

df_results.to_csv('./results.csv')