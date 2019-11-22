import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, make_scorer, precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import warnings
warnings.filterwarnings('ignore')


wine = pd.read_csv('/home/FRACTAL/brajith.dr/Downloads/27-7-19_RandomForests_SanketKakade-master/wine_quality_classification.csv')
wine.head()

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine)


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = wine)

fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)

f, ax = plt.subplots(figsize=(10, 8))
corr = wine.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)



label_quality = LabelEncoder()

wine['quality'] = label_quality.fit_transform(wine['quality'])
wine['quality'].value_counts()

sns.countplot(wine['quality'])

X = wine.drop('quality', axis = 1)
y = wine['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify= wine['quality'], random_state = 42)


print (y_train.value_counts())
print (y_test.value_counts())

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


lr= LogisticRegression()
lr.fit(X_train, y_train)
pred_lr= lr.predict(X_test)


print (accuracy_score(y_test, pred_lr))


print (accuracy_score(y_test, pred_lr))
rfc = RandomForestClassifier(n_estimators=10, random_state=2)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)


rfc = RandomForestClassifier(n_estimators=10, random_state=2)
rfc_model=rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

data_feature_names=list(X.columns)


from sklearn import tree
import collections
from sklearn import tree
import pydotplus
import pydot



forest_clf=rfc.fit(X_train, y_train)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

dot_data = tree.export_graphviz(clf,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)




graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))


for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])


graph.write_png('tree.png')

parameters = {'n_estimators': [30, 40, 50,100,200,300], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy','gini'],
              'max_depth': [10, 15, 18, 20, 25, 30], 
              'min_samples_split': [2, 3, 5, 7],
              'min_samples_leaf': [1,5,8]
             }
acc_scorer = make_scorer(f1_score)



parameters = {'n_estimators': [30, 40, 50,100,200,300], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy','gini'],
              'max_depth': [10, 15, 18, 20, 25, 30], 
              'min_samples_split': [2, 3, 5, 7],
              'min_samples_leaf': [1,5,8]
             }
acc_scorer = make_scorer(f1_score)
