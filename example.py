#def gender_features(word):
#	return {'last_letter': word[-1]}
#import nltk
#from nltk.corpus import names
#labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
#	[(name, 'female') for name in names.words('female.txt')])
#import random
#random.shuffle(labeled_names)
#featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
#train_set, test_set = featuresets[500:], featuresets[:500]
#classifier = nltk.NaiveBayesClassifier.train(train_set)

#print(classifier.classify(gender_features('Neo')))
#print(classifier.classify(gender_features('Trinity')))

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
##############################
iris = datasets.load_iris()
x=iris.data
y=iris.target
############################
test_size=0.3
random_state=0
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=test_size,random_state=random_state)
################
sc=StandardScaler()
sc.fit(x_train)

x_train_std=sc.transform(x_train)
x_test_std=sc.transform(x_test)

#####
print("unique labels: {0}".format(np.unique(y)))

x_train_std=x_train_std[:,[2,3]]
x_test_std=x_test_std[:,[2,3]]

n_iter=40
eta0=0.1

##################################

ppn=Perceptron(n_iter=n_iter,eta0=eta0,random_state=random_state)

#fit the model to the standardized data

ppn.fit(x_train_std,y_train)

#make predictions
y_pred=ppn.predict(x_test_std)

print("accuracy: {0:.2f}%".format(accuracy_score(y_test,y_pred)*100))
