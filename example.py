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
data=iris.data
target=iris.target
############################
test_size=0.3
random_state=0
data_train,data_test,target_train,target_test =train_test_split(data,target,test_size=test_size,random_state=random_state)
################
sc=StandardScaler()
sc.fit(data_train)

data_train=sc.transform(data_train)
data_test=sc.transform(data_test)

#####
print("unique labels: {0}".format(np.unique(target)))

data_train=data_train[:,[2,3]]
data_test=data_test[:,[2,3]]

n_iter=40
eta0=0.1

##################################

ppn=Perceptron(n_iter=n_iter,eta0=eta0,random_state=random_state)

#fit the model to the standardized data
print("data train")
print(data_train)
print("\n\n\n")

print("target train")
print(target_train)
print("\n\n\n")
counter=0
for i in data_train:
    print(i,target_train[counter])
    counter+=1
ppn.fit(data_train,target_train)

#make predictions
print("data test")
print(data_test)
target_pred=ppn.predict(data_test)

print("accuracy: {0:.2f}%".format(accuracy_score(target_test,target_pred)*100))
