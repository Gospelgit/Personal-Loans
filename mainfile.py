!pip install numpy==1.25.2 pandas==1.5.3 matplotlib==3.7.1 seaborn==0.13.1 scikit-learn==1.2.2 sklearn-pandas==2.2.0 -q --user 

# import libraries for data manipulation
import numpy as np
import pandas as pd

# import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

#import libraries for Machine Learning 
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#new dataset 
df = '/content/sample_data/Loan_Modelling.csv'
df = pd.read_csv(df)

#data overview 
df.shape 

df.head()

df.info()

df.describe()


#exploratory data analysis 

plt.figure(figsize = (15,5))
sns.boxplot(data = df, x = 'Mortgage');


plt.figure(figsize = (15,5))
sns.histplot(data = df, x = 'Mortgage');


sns.histplot(data=df,x='CCAvg')
plt.show()

sns.boxplot(data=df,x='CCAvg')
plt.show()

sns.histplot(data=df,x='Income')
plt.show()

df['Personal_Loan'].value_counts()

df['CreditCard'].value_counts()

df['Online'].value_counts()


#bivariate EDA
sns.pairplot(data = df)

sns.jointplot(x='Age', y='Personal_Loan', data=df, kind='hex', gridsize=50, cmap='Blues')
plt.show()

sns.scatterplot(data = df, x = 'Education', y = 'Personal_Loan')

sns.scatterplot(data = df, x = 'Securities_Account', y = 'Personal_Loan')

sns.scatterplot(data = df, x = 'Income', y = 'Personal_Loan')

sns.scatterplot(data = df, x = 'Experience', y = 'Personal_Loan')


#Data preprocessing 

df.isnull.sum()

df.duplicated.sum() 

df['ZIPCode'].nunique() 

# Bin the zip codes using qcut, converting them to 4 categorical values
df['ZipCode_Binned'] = pd.qcut(df['ZIPCode'], q=4, labels=['1', '2', '3', '4'])
print(df)

#drop the ID and Zip code columns
df = df.drop(['ID', 'ZIPCode'], axis =1)
print(df)


#converting the Zipcode binned column to an int
df['ZipCode_Binned'] = df['ZipCode_Binned'].astype(int)

# Function to identify outliers using IQR
def find_outliers_iqr(df):
    outliers = pd.DataFrame()
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers = pd.concat([outliers, col_outliers], axis=0)
    return outliers.drop_duplicates()

# Identify outliers
outliers_iqr = find_outliers_iqr(df)

print("Outliers detected by IQR method:")
print(outliers_iqr)

#calculating the percentage of outliers
percentage_outliers = (len(outliers_iqr) / len(df)) * 100
print(f"Percentage of outliers: {percentage_outliers:.2f}%")

''' Model Building '''
# split data
X = df.drop('Personal_Loan', axis=1)
y = df.pop('Personal_Loan')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state = 5)

#build decision tree model
dtPerLoan = DecisionTreeClassifier(criterion = 'gini', random_state = 5)
dtPerLoan.fit(X_train, y_train)

#score the decision tree
print(dtPerLoan.score(X_train, y_train))
print(dtPerLoan.score(X_test, y_test))

#checking number of positives
y.sum(axis=0)

'''
What does the bank want?
Prediction of customers who would buy personal loans so they can be targetted with Ads. 
So, there is no harm if the model predicts customers who may not want the loan. So there are two losses here:

      1. Targeting customers who wouldn't buy the loan with Ads
      2. Missing out on customers who would buy the loan
We can afford the first loss but not the second. We want to capture as much customers who would buy the loan as possible. 
So we'll also try Recall for Model Evaluation Metric
'''

''' Model performance improvement'''

def confusion_matrix(model,y_actual,labels=[1,0]):
    '''
    model: classifier to predict values of x
    y_actual = ground truth

    '''

    y_predict = model.predict(X_test)
    cm = metrics.confusion_matrix(y_actual, y_predict, labels=[0,1])
    df.cm = pd.DataFrame(cm, index = [i for i in ["Actual -No", "Actual -Yes"]], columns = [i for i in ["Predicted -No", "Predicted - Yes"]])
    group_counts =["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.array(labels).reshape(2,2)
    plt.figure(figsize = (10,7))
    sns.heatmap(df.cm, annot = labels, fmt = '')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def cal_recall_score(model):
    '''
    model : classifier to predict the values of X

    '''

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    print("Recall on training data set: ", metrics.recall_score(y_train, pred_train))
    print("Recall on test data set: ", metrics.recall_score(y_test, pred_test))

#making confusion matrix for this model
confusion_matrix(dtPerLoan, y_test)

#recall on train and test sets
cal_recall_score(dtPerLoan)

#visualizing the Decision Tree
feature_names = list(X.columns)
print(feature_names)


plt.figure(figsize=(20,30))
tree.plot_tree(dtPerLoan, feature_names = feature_names, filled = True, fontsize=9, node_ids=True, class_names=True)
plt.show()

#showing the result in text form
print(tree.export_text(dtPerLoan,feature_names = feature_names, show_weights= True))

#checking the importance of various features
print(pd.DataFrame(dtPerLoan.feature_importances_, columns = ["importance"], index = X_train.columns).sort_values(by = 'importance', ascending=False))

#plotting importance as a barchart

imp = dtPerLoan.feature_importances_
indices = np.argsort(imp)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='blue', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


'''
Pre-pruning with a depth of 3

'''

dtPerLoan1 = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 5)
dtPerLoan1.fit(X_train, y_train)

#the confusion matrix for the new model
confusion_matrix(dtPerLoan1, y_test)

#accuracy on dtPerLoan1
print("Accuracy on training set:", dtPerLoan1.score(X_train, y_train))
print("Accuracy on testing set:", dtPerLoan1.score(X_test, y_test))

#recall on dtPerLoan1 test and train
cal_recall_score(dtPerLoan1)

#visualizing the dtPerLoan1 tree

plt.figure(figsize=(15,10))
tree.plot_tree(dtPerLoan1, feature_names = feature_names, filled = True, fontsize=9, node_ids=True, class_names=True)
plt.show()

#checking the importance of various features
print(pd.DataFrame(dtPerLoan1.feature_importances_, columns = ["importance"], index = X_train.columns).sort_values(by = 'importance', ascending=False))

#plotting importance as a barchart

imp = dtPerLoan1.feature_importances_
indices = np.argsort(imp)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='blue', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


'''
Pre-pruning using GridSearchCV for Hyperparamter
'''
#choosing classifier
estimator = DecisionTreeClassifier(random_state = 5)
parameters = {'max_depth': np.arange(1,10),
              'min_samples_leaf': [1,3,5,9,11,13,17],
              'max_leaf_nodes': [2,3,5,10],
              'min_impurity_decrease': [0.001, 0.01, 0.1]}

#type of scorer for parameter combination
acc_scorer = metrics.make_scorer(metrics.recall_score)

#run the grid search  estimator
grid_obj = GridSearchCV(estimator, parameters, scoring = acc_scorer, cv=5)
grid_obj= grid_obj.fit(X_train, y_train)

#set clf to the best parameter combination
estimator = grid_obj.best_estimator_


#fit the best algorithm to the data
estimator.fit(X_train, y_train)

#confusion matrix for the Gridsearch
confusion_matrix(estimator, y_test)


#accuracy on estimator
print("Accuracy on training set:", estimator.score(X_train, y_train))
print("Accuracy on testing set:", estimator.score(X_test, y_test))


#recall on estimator test and train
cal_recall_score(estimator)

#visualizing the estimator tree

plt.figure(figsize=(15,10))
tree.plot_tree(estimator, feature_names = feature_names, filled = True, fontsize=9, node_ids=True, class_names=True)
plt.show()

#showing the result in text form
print(tree.export_text(estimator,feature_names = feature_names, show_weights= True))

#checking the importance of various features
print(pd.DataFrame(estimator.feature_importances_, columns = ["importance"], index = X_train.columns).sort_values(by = 'importance', ascending=False))

#plotting importance as a barchart

imp = estimator.feature_importances_
indices = np.argsort(imp)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='blue', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


'''
post-pruning with CCAlpha 
'''

clf = DecisionTreeClassifier(random_state= 5)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = abs(path.ccp_alphas), path.impurities

pd.DataFrame(path)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(ccp_alphas[:-1], impurities[:-1], marker= 'o', drawstyle = 'steps-post')
ax.set_xlabel('effective alphas')
ax.set_ylabel('total impurities of leaves')
ax.set_title('Total Impurties vs effective alphas for training set')
plt.show()

#training the DT with effective alphas
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state= 5, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

print('Number of nodes in the last tree is: {} with ccp_alpha: {}'.format(
    clfs[-1].tree_.node_count, ccp_alphas[-1]))

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax= plt.subplots(2,1,figsize=(10,7))
ax[0].plot(ccp_alphas, node_counts, marker = 'o', drawstyle = 'steps-post')
ax[0].set_xlabel('alpha')
ax[0].set_ylabel('number of nodes')
ax[0].set_title('Number of nodes vs alphas')
ax[1].plot(ccp_alphas, depth, marker = 'o', drawstyle = 'steps-post')
ax[1].set_xlabel('alpha')
ax[1].set_ylabel('depth of tree')
ax[1].set_title('depth  vs alphas')
fig.tight_layout()

#accuracy vs alpha for training and testing sets

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title('Accuracy vs alpha  for training  and testing set')
ax.plot(ccp_alphas, train_scores, marker = 'o', label='train', drawstyle = 'steps-post')
ax.plot(ccp_alphas, test_scores, marker = 'o', label='test', drawstyle = 'steps-post')
ax.legend()
plt.show()

#accuracy of the model
index_best_model = np.argmax(test_scores)
best_model = clfs[index_best_model]
print(best_model)
print('Training accuracy for best model:', best_model.score(X_train, y_train))
print('Test accuracy for best model:', best_model.score(X_test, y_test))

# but accuracy is not our best metric, recall is, so we'll use recall

recall_train = []
for clf in clfs:
    pred_train3 = clf.predict(X_train)
    values_train= metrics.recall_score(y_train, pred_train3)
    recall_train.append(values_train)

recall_test = []
for clf in clfs:
    pred_test3 = clf.predict(X_test)
    values_test= metrics.recall_score(y_test, pred_test3)
    recall_test.append(values_test)

fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlabel('alpha')
ax.set_ylabel('recall')
ax.set_title('recall vs alpha  for training  and testing set')
ax.plot(ccp_alphas, recall_train, marker = 'o', label='train', drawstyle = 'steps-post')
ax.plot(ccp_alphas, recall_test, marker = 'o', label='test', drawstyle = 'steps-post')
ax.legend()
plt.show()

#recall of the model
index_best_model = np.argmax(recall_test)
best_model = clfs[index_best_model]
print(best_model)

#confusion matrix for recall
confusion_matrix(best_model, y_test)


#recall on train and test set
cal_recall_score(best_model)

#visualizing
plt.figure(figsize=(15,10))
tree.plot_tree(best_model, feature_names = feature_names, filled = True, fontsize=9, node_ids=True, class_names=True)
plt.show()

#text report showing the rules of the DT
print(tree.export_text(best_model,feature_names = feature_names, show_weights= True))

#checking the importance of various features
print(pd.DataFrame(best_model.feature_importances_, columns = ["importance"], index = X_train.columns).sort_values(by = 'importance', ascending=False))

#plotting importance as a barchart

imp = best_model.feature_importances_
indices = np.argsort(imp)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='blue', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


'''
Model Comparison and Final Model Selection
'''

comparison_frame =pd.DataFrame({'Model':['Initial decision tree model', 'Decision tree with restricted maximum depth', 'Decision tree with hyperparameter tuning', 'Decision tree with post-pruning'], 'Train_Recall': [1, 0.977, 0.918, 0.936], 'Test_Recall': [0.899, 0.818, 0.912, 0.912]})
comparison_frame

