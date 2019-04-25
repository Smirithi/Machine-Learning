#importing libraries
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#importing dataset
df = pd.read_csv("creditcard.csv")
df.info()
df.describe()
df.isnull().sum()

#exploratory data analysis(EDA)
# the target variable (Class)
count_classes = pd.DataFrame(pd.value_counts(df['Class'], sort = True).sort_index())
sb.countplot(x = 'Class' , data = df, hue = 'Class')

#distribution of target variable according to time
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
bins = 50
ax1.hist(df.Time[df.Class == 1], bins = bins)
ax1.set_title('Fraud')
ax2.hist(df.Time[df.Class == 0], bins = bins)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Number of Transactions')
plt.show()

#distribution of target variable according to amount
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))
bins = 30
ax1.hist(df.Amount[df.Class == 1], bins = bins)
ax1.set_title('Fraud')
ax2.hist(df.Amount[df.Class == 0], bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.show()

"""Most transactions are small amounts, less than 100.
Fraudulent transactions have a maximum value far less than normal 
transactions, 2,125.87 vs $25,691.16."""

#amount vs time (nothing specifically new is found from this graph)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,6))
ax1.scatter(df.Time[df.Class == 1], df.Amount[df.Class == 1])
ax1.set_title('Fraud')
ax2.scatter(df.Time[df.Class == 0], df.Amount[df.Class == 0])
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

#Select only the anonymized features.
v_features = df.iloc[:,1:29].columns

plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
# orange (Normal) blue (Fraud)
# enumerate gives an index that can be used to iterate in the for loops
for i, cn in enumerate(df[v_features]):
    ax = plt.subplot(gs[i])
    sb.distplot(df[cn][df.Class == 1], bins=50)
    sb.distplot(df[cn][df.Class == 0], bins=50)
    ax.set_xlabel('')
    plt.legend(df["Class"])
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()

# preparing train and test data
x = df.iloc[:,0:30]
y = df.iloc[:,30:]
#scaling the x_train and x_test data
sc = StandardScaler()
x.iloc[:,[0,29]] = sc.fit_transform(x.iloc[:,[0,29]])

#splitting into training and testing data
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=0)

#Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(xtrain,ytrain)
y_lr_model = lr_model.predict(xtest)

print("Logistic Regression metrics:\n")
print(f"Classification_report = \n{classification_report(ytest,y_lr_model)}")
print(f"Confusion matrix = \n{confusion_matrix(ytest,y_lr_model)}")
print(sb.heatmap(confusion_matrix(ytest,y_lr_model),annot=True,cmap="Blues"))
print(f"Accuracy score = {accuracy_score(ytest,y_lr_model)}")

# Linear SVC
svc_model = LinearSVC()
svc_model.fit(xtrain,ytrain)
y_svc = svc_model.predict(xtest)

print("Linear SVC metrics:\n")
print(f"Classification_report = \n{classification_report(ytest,y_svc)}")
print(f"Confusion matrix = \n{confusion_matrix(ytest,y_svc)}")
print(sb.heatmap(confusion_matrix(ytest,y_svc),annot=True,cmap="Blues"))
print(f"Accuracy score = {accuracy_score(ytest,y_svc)}")

#Naive Bayes
nb_model = GaussianNB()
nb_model.fit(xtrain,ytrain)
y_nb = nb_model.predict(xtest)

print("Naive Bayes metrics:\n")
print(f"Classification_report = \n{classification_report(ytest,y_nb)}")
print(f"Confusion matrix = \n{confusion_matrix(ytest,y_nb)}")
print(sb.heatmap(confusion_matrix(ytest,y_nb),annot=True,cmap="Blues"))
print(f"Accuracy score = {accuracy_score(ytest,y_nb)}")

#Artificial Neural Network
X = dataset.iloc[:, 1:30].values
y = dataset.iloc[:, 30].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# let's make the ANN!

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))

# Adding the second hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the fifth hidden layer
classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10000, epochs = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10000, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=3)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [15000, 25000],
              'epochs': [75, 125],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
