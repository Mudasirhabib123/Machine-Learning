
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold,cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import datasets

def get_scores(model, x_train,x_test, y_train, y_test ):
    model.fit(x_train, y_train)
    return model.score(x_test,y_test)

# Load dataset

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# dataset = read_csv(url , names=names)
iris = datasets.load_iris()
# print(dataset.groupby('sepal-length').size())

# dataset.plot(kind = 'box', subplots=True, layout=(2,2))
# dataset.hist()
# scatter_matrix(dataset)
# pyplot.show()


data = dataset.values
# x=data[:,0:4]
# y=data[:,4]

x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.20, random_state=1)

models = []
models.append(('Logistic Regression',LogisticRegression(solver='liblinear', multi_class = 'ovr')))
models.append(('Linear Discriminant Analysis',LinearDiscriminantAnalysis()))
models.append(('KNeighbors Classifier',KNeighborsClassifier()))
models.append(('Decision Tree Classifier',DecisionTreeClassifier()))
models.append(('GaussianNB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring= 'accuracy')
    # cv_results = get_scores(model, x_train,x_test, y_train, y_test)
    names.append(name)
    results.append(cv_results)
    print('%s: %f' % (name, cv_results.mean()))
    # print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


pyplot.boxplot(results)
pyplot.title("Accuracy Measure")

model = SVC()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print("Accuracy -->",accuracy_score(y_test, pred))
print("Confussion Matrix --> \n",confusion_matrix(y_test, pred))
print("Classification Report --> \n",classification_report(y_test, pred))


# pyplot.show()
