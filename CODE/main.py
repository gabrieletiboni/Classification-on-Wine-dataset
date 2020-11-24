import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

import math

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

#
# Class for homework #1
# Look at the "assignment" function as the main controller of the homework, that calls out functions
#
class HW1:
	
	def __init__(self):
		return

	# MAIN CONTROLLER OF THE HOMEWORK
	def assignment(self, random_state=None):

		self.random_state = random_state

		dataset = load_wine()

		X = dataset.data
		y = dataset.target
		N = len(y)

		# --- Explore the dataset and print some statistics
		self.dataExploration(X, y)
		
		# Reduce dataset to only a subset of features
		featuresConsidered = [0, 1]
		X = np.array(X[:, featuresConsidered])

		# --- Plot the dataset on a 2D graph
		# self.plotSet(X, y)

		# Split dataset into train, val and test sets with proportions 5:2:3
		X_intermediate, X_test, y_intermediate, y_test = train_test_split(X, y, test_size=0.3, stratify=None, random_state=self.random_state)
		X_train, X_val, y_train, y_val = train_test_split(X_intermediate, y_intermediate, test_size=0.285, stratify=None, random_state=self.random_state)

		# --- Check the split of the dataset
		print('X train:', len(X_train[:,0]), len(X_train[:,0])/N)
		print('X val:', len(X_val[:,0]), len(X_val[:,0])/N)
		print('X test:', len(X_test[:,0]), len(X_test[:,0])/N)

		# unique, counts = np.unique(y_test, return_counts=True)
		# print(np.asarray((unique, counts)).transpose())
		# print('---')

		# --- 1) KNN
		self.KNNassignment(X_train, y_train, X_val, y_val, X_test, y_test)

		# --- 2) Linear SVM
		# self.LinearSVMAssignment(X_train, y_train, X_val, y_val, X_test, y_test)

		# --- 3) RBF-Kernel SVM
		# self.RBFSVMAssignment(X_train, y_train, X_val, y_val, X_test, y_test)

		# --- 4) RBF-Kernel GRIDSEARCH SVM
		# self.RBFSVMGammaAssignment(X_train, y_train, X_val, y_val, X_test, y_test)

		# --- 5) RBF-Kernel K-FOLD GRIDSEARCH SVM
		# self.RBFSVMKFoldAssignment(X_train, y_train, X_val, y_val, X_test, y_test)

		return

	# Methods

	def KNNassignment(self, X_train_original, y_train, X_val_original, y_val, X_test_original, y_test):

		# Very important to standardize data before using KNN
		scaler = StandardScaler()
		scaler.fit(X_train_original)

		X_train = scaler.transform(X_train_original)
		X_val = scaler.transform(X_val_original)

		# pca = PCA(n_components=2)
		
		# pca.fit(X_train)
		# X_train = pca.transform(X_train)
		# X_val = pca.transform(X_val)
		# ---------------------------------------------------

		# self.plotSet(X_train_original, y_train)
		# self.plotSet(X_train, y_train)

		Ks = [1,3,5,7]
		bestK = Ks[0]
		bestAccuracy = 0
		accuracies = list()

		nRows=2
		nCols=2
		fig, ax = plt.subplots(nrows=nRows, ncols=nCols)

		for i, k in enumerate(Ks):
			currAx = ax[math.floor(i/nCols), i%nCols]

			clf = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto')
			clf.fit(X_train, y_train)

			y_pred = clf.predict(X_val)
			accuracy = accuracy_score(y_val, y_pred)

			if accuracy > bestAccuracy:
				bestK = k

			print('Accuracy on val set:', accuracy, '| with k =', k)
			accuracies.append(accuracy)

			self.print_decision_boundaries(clf, X_train, y_train, ax=currAx, title='K='+str(k))

		plt.show()

		# --- Plot accuracy graph
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(Ks, accuracies, c='blue', linestyle='-', marker='o', label='Accuracy')

		ax.set_xlabel('Value of K', labelpad=12, fontweight='bold')
		ax.set_ylabel('Accuracy', labelpad=32, rotation=0, fontweight='bold')

		plt.grid(b=True, axis='y', alpha=0.3)
		plt.show()
		# -----------------------

		# test final model on the test set
		print('FINAL EVALUATION OF THE MODEL')
		print('BEST K FOUND:', bestK)

		X_final = np.concatenate((X_train_original, X_val_original), axis=-2)
		y_final = np.concatenate((y_train, y_val))

		scaler.fit(X_final)
		X_final = scaler.transform(X_final)
		X_test = scaler.transform(X_test_original)

		# X_final = pca.transform(X_final)
		# X_test = pca.transform(X_test)

		clf = KNeighborsClassifier(n_neighbors=bestK, weights='uniform', algorithm='auto')
		clf.fit(X_final, y_final)

		y_pred = clf.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)

		print('FINAL ACCURACY GOT:', accuracy)

		return accuracies


	def LinearSVMAssignment(self, X_train_original, y_train, X_val_original, y_val, X_test_original, y_test):

		# Important to standardize data before using SVM (https://scikit-learn.org/stable/modules/svm.html)
		scaler = StandardScaler()
		scaler.fit(X_train_original)

		X_train = scaler.transform(X_train_original)
		X_val = scaler.transform(X_val_original)

		# pca = PCA(n_components=2)
		# pca.fit(X_train)
		# X_train = pca.transform(X_train)
		# X_val = pca.transform(X_val)
		# ---------------------------------------------------

		# self.plotSet(X_train_original, y_train)
		# self.plotSet(X_train, y_train)

		Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
		bestC = Cs[0]
		bestAccuracy = 0
		accuracies = list()

		nRows=2
		nCols=4
		fig, ax = plt.subplots(nrows=nRows, ncols=nCols)

		for i, c in enumerate(Cs):
			currAx = ax[math.floor(i/nCols), i%nCols]

			clf = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.0001, C=c, max_iter=10000)
			clf.fit(X_train, y_train)

			y_pred = clf.predict(X_val)
			accuracy = accuracy_score(y_val, y_pred)

			if accuracy >= bestAccuracy:
				bestC = c
				bestAccuracy = accuracy

			print('Accuracy on val set:', accuracy, '| with C =', c)
			accuracies.append(accuracy)

			self.print_decision_boundaries(clf, X_train, y_train, ax=currAx, title='C='+str(c))

		plt.show()

		# --- Plot accuracy graph
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(Cs, accuracies, c='blue', linestyle='-', marker='o', label='Accuracy')

		ax.set_xlabel('Value of C', labelpad=12, fontweight='bold')
		ax.set_ylabel('Accuracy', labelpad=32, rotation=0, fontweight='bold')
		
		ax.set_xticks(Cs)
		ax.set_xscale('log')

		plt.grid(b=True, axis='y', alpha=0.3)
		plt.show()
		# -----------------------

		# evaluate final model on the test set
		print('FINAL EVALUATION OF THE MODEL')
		print('BEST C FOUND:', bestC)

		X_final = np.concatenate((X_train_original, X_val_original), axis=-2)
		y_final = np.concatenate((y_train, y_val))

		scaler.fit(X_final)
		X_final = scaler.transform(X_final)
		X_test = scaler.transform(X_test_original)

		# X_final = pca.transform(X_final)
		# X_test = pca.transform(X_test)

		clf = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.0001, C=bestC, max_iter=10000)
		clf.fit(X_final, y_final)

		y_pred = clf.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)

		print('FINAL ACCURACY GOT:', accuracy)

		return

	def RBFSVMAssignment(self, X_train_original, y_train, X_val_original, y_val, X_test_original, y_test):

		# Important to standardize data before using SVM (https://scikit-learn.org/stable/modules/svm.html)
		scaler = StandardScaler()
		scaler.fit(X_train_original)

		X_train = scaler.transform(X_train_original)
		X_val = scaler.transform(X_val_original)

		# pca = PCA(n_components=2)
		# pca.fit(X_train)
		# X_train = pca.transform(X_train)
		# X_val = pca.transform(X_val)
		# ---------------------------------------------------

		# self.plotSet(X_train_original, y_train)
		# self.plotSet(X_train, y_train)

		unique, counts = np.unique(y_val, return_counts=True)
		print(np.asarray((unique, counts)).transpose())

		Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
		bestC = Cs[0]
		bestAccuracy = 0
		accuracies = list()

		nRows=2
		nCols=4
		fig, ax = plt.subplots(nrows=nRows, ncols=nCols)

		for i, c in enumerate(Cs):
			currAx = ax[math.floor(i/nCols), i%nCols]

			clf = SVC(C=c, kernel='rbf', gamma='scale')	
			clf.fit(X_train, y_train)

			y_pred = clf.predict(X_val)
			accuracy = accuracy_score(y_val, y_pred)

			if accuracy >= bestAccuracy:
				bestC = c
				bestAccuracy = accuracy

			print('Accuracy on val set:', accuracy, '| with C =', c)
			accuracies.append(accuracy)

			print('Number of support vectors with C =', c, 'is: ', clf.n_support_.sum(), '-', clf.n_support_)

			self.print_decision_boundaries(clf, X_train, y_train, ax=currAx, title='C='+str(c))

		plt.show()

		# --- Plot accuracy graph
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(Cs, accuracies, c='blue', linestyle='-', marker='o', label='Accuracy')

		ax.set_xlabel('Value of C', labelpad=12, fontweight='bold')
		ax.set_ylabel('Accuracy', labelpad=32, rotation=0, fontweight='bold')
		
		ax.set_xticks(Cs)
		ax.set_xscale('log')

		plt.grid(b=True, axis='y', alpha=0.3)
		plt.show()
		# -----------------------

		# evaluate final model on the test set
		print('FINAL EVALUATION OF THE MODEL')
		print('BEST C FOUND:', bestC)

		X_final = np.concatenate((X_train_original, X_val_original), axis=-2)
		y_final = np.concatenate((y_train, y_val))

		scaler.fit(X_final)
		X_final = scaler.transform(X_final)
		X_test = scaler.transform(X_test_original)

		# X_final = pca.transform(X_final)
		# X_test = pca.transform(X_test)

		clf = SVC(C=bestC, kernel='rbf', gamma='scale')	
		clf.fit(X_final, y_final)

		y_pred = clf.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)

		print('FINAL ACCURACY GOT:', accuracy)
		# print('predicted:', y_pred)

		return

	def RBFSVMGammaAssignment(self, X_train_original, y_train, X_val_original, y_val, X_test_original, y_test):

		# Important to standardize data before using SVM (https://scikit-learn.org/stable/modules/svm.html)
		scaler = StandardScaler()
		scaler.fit(X_train_original)

		X_train = scaler.transform(X_train_original)
		X_val = scaler.transform(X_val_original)

		# pca = PCA(n_components=2)
		# pca.fit(X_train)
		# X_train = pca.transform(X_train)
		# X_val = pca.transform(X_val)
		# ---------------------------------------------------

		params = {
			'C': [10, 100, 1000],
			'gamma': [0.01, 0.1, 1.0, 10.0]
		}

		bestParam = None
		bestAccuracy = 0.0
		accuracies = list()
		tries = list()
		i=0

		nRows=3
		nCols=4
		fig, ax = plt.subplots(nrows=nRows, ncols=nCols)
		
		for configuration in ParameterGrid(params):
			currAx = ax[math.floor(i/nCols), i%nCols]

			clf = SVC(kernel='rbf', **configuration)	
			clf.fit(X_train, y_train)

			y_pred = clf.predict(X_val)
			accuracy = accuracy_score(y_val, y_pred)

			if accuracy >= bestAccuracy:
				bestParam = configuration
				bestAccuracy = accuracy

			print('Accuracy on val set:', accuracy, '| with params:', configuration)
			accuracies.append(accuracy)
			tries.append(i)
			i=i+1

			self.print_decision_boundaries(clf, X_train, y_train, ax=currAx, title='')

		plt.show()

		# --- Plot accuracy graph
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(tries, accuracies, c='blue', linestyle='-', marker='o', label='Accuracy')

		ax.set_xlabel('Params', labelpad=12, fontweight='bold')
		ax.set_ylabel('Accuracy', labelpad=32, rotation=0, fontweight='bold')
		
		# ax.set_xticks(Cs)
		# ax.set_xscale('log')

		plt.grid(b=True, axis='y', alpha=0.3)
		plt.show()
		# -----------------------

		# evaluate final model on the test set
		print('FINAL EVALUATION OF THE MODEL')
		# print('BEST C FOUND:', bestC)
		print('BEST CONFIGURATION FOUND:', bestParam)

		X_final = np.concatenate((X_train_original, X_val_original), axis=-2)
		y_final = np.concatenate((y_train, y_val))

		scaler.fit(X_final)
		X_final = scaler.transform(X_final)
		X_test = scaler.transform(X_test_original)

		# X_final = pca.transform(X_final)
		# X_test = pca.transform(X_test)

		# clf = SVC(kernel='rbf', **bestParam)
		# clf = SVC(kernel='rbf', C=10, gamma=0.01)
		clf.fit(X_final, y_final)

		y_pred = clf.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)

		print('FINAL ACCURACY GOT:', accuracy)
		# print('predicted:', y_pred)

		self.print_decision_boundaries(clf, X_final, y_final, ax=None, title='Decision boundaries')

		unique, counts = np.unique(y_test, return_counts=True)
		print(np.asarray((unique, counts)).transpose())

		return

	def RBFSVMKFoldAssignment(self, X_train_original, y_train, X_val_original, y_val, X_test_original, y_test):
# 
		# Important to standardize data before using SVM (https://scikit-learn.org/stable/modules/svm.html)
		scaler = StandardScaler()
		scaler.fit(X_train_original)

		X_train = scaler.transform(X_train_original)
		X_val = scaler.transform(X_val_original)

		# --- REMEMBER to scale features before applying PCA (tfidf is already scaled)
		# --- REMEMBER rows are not normalized anymore after applying PCA
		#
		pca = PCA(n_components=2)

		pca.fit(X_train)
		# X_train = pca.transform(X_train)
		X_val = pca.transform(X_val)

		# self.plotSet(X_train, y_train)
		# ---------------------------------------------------

		params = {
			'C': [0.1, 1, 10, 100, 1000],
			'gamma': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
		}

		bestParam = None
		bestAccuracy = 0.0
		accuracies = list()
		tries = list()
		i=0

		# nRows=3
		# nCols=4
		# fig, ax = plt.subplots(nrows=nRows, ncols=nCols)

		X_final = np.concatenate((X_train_original, X_val_original), axis=-2)
		y_final = np.concatenate((y_train, y_val))

		scaler.fit(X_final)
		X_final = scaler.transform(X_final)
		X_test = scaler.transform(X_test_original)
		
		for configuration in ParameterGrid(params):
			# currAx = ax[math.floor(i/nCols), i%nCols]

			clf = SVC(kernel='rbf', **configuration)


			score_on_each_fold = cross_val_score(clf, X_final, y_final, cv=20, scoring='accuracy')
			accuracy = score_on_each_fold.mean()

			if accuracy >= bestAccuracy:
				bestParam = configuration
				bestAccuracy = accuracy

			print('Accuracy on val set:', accuracy, '| with params:', configuration)
			accuracies.append(accuracy)
			tries.append(i)
			i=i+1

			# self.print_decision_boundaries(clf, X_train, y_train, ax=currAx, title='')

		# plt.show()

		# --- Plot accuracy graph
		fig, ax = plt.subplots(nrows=1, ncols=1)
		ax.plot(tries, accuracies, c='blue', linestyle='-', marker='o', label='Accuracy')

		ax.set_xlabel('Params', labelpad=12, fontweight='bold')
		ax.set_ylabel('Accuracy', labelpad=32, rotation=0, fontweight='bold')

		plt.grid(b=True, axis='y', alpha=0.3)
		plt.show()
		# -----------------------

		# evaluate final model on the test set
		print('FINAL EVALUATION OF THE MODEL')
		# print('BEST C FOUND:', bestC)
		print('BEST CONFIGURATION FOUND:', bestParam)

		X_final = np.concatenate((X_train_original, X_val_original), axis=-2)
		y_final = np.concatenate((y_train, y_val))

		scaler.fit(X_final)
		X_final = scaler.transform(X_final)
		X_test = scaler.transform(X_test_original)

		X_final = pca.transform(X_final)
		X_test = pca.transform(X_test)

		# X_final = Normalizer(copy=True).transform(X_final)
		# X_test = Normalizer(copy=True).transform(X_test)

		clf = SVC(kernel='rbf', **bestParam)
		# clf = SVC(kernel='rbf', C=10, gamma=0.5)
		clf.fit(X_final, y_final)

		y_pred = clf.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)

		print('FINAL ACCURACY GOT:', accuracy)
		# print('predicted:', y_pred)

		self.print_decision_boundaries(clf, X_final, y_final, ax=None, title='Decision boundaries')

		unique, counts = np.unique(y_test, return_counts=True)
		print(np.asarray((unique, counts)).transpose())

		return

	def plotSet(self, X, y):

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
		ax.scatter(X[y==0, 0], X[y==0, 1], s=10, alpha=0.5, label='class 0')
		ax.scatter(X[y==1, 0], X[y==1, 1], s=10, alpha=0.5, label='class 1')
		ax.scatter(X[y==2, 0], X[y==2, 1], s=10, alpha=0.5, label='class 2')

		ax.set_xlabel('Alcohol %', labelpad=12, fontweight='bold')
		ax.set_ylabel('Malic acid', labelpad=32, rotation=0, fontweight='bold')

		ax.legend()
		plt.show()

		return

	# --- FUNCTIONS FOR PLOTTING THE DECISION BOUNDARIES
	def print_decision_boundaries(self, clf, X, y, ax=None, title='Decision surface'):
		if ax==None: createFigure=True
		else: createFigure=False

		if createFigure:
			fig, ax = plt.subplots()

		# Set-up grid for plotting.
		X0, X1 = X[:, 0], X[:, 1]
		xx, yy = self.make_meshgrid(X0, X1)

		self.plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.6)
		ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, alpha=0.8, edgecolor='k')

		# ax.set_xlabel('Alcohol')
		# ax.set_ylabel('Malic acid')

		ax.set_title(title)
		# ax.legend()

		if createFigure:
			plt.show()

		return

	def make_meshgrid(self, x, y, h=.02):
	    x_min, x_max = x.min() - 1, x.max() + 1
	    y_min, y_max = y.min() - 1, y.max() + 1
	    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	    return xx, yy

	def plot_contours(self, ax, clf, xx, yy, **params):
	    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	    Z = Z.reshape(xx.shape)
	    out = ax.contourf(xx, yy, Z, **params)
	    return out
	# --------------------------------------------------

	def dataExploration(self, X, y):

		print('--------- DATA EXPLORATION ---------')
		print('lenght of X: ', len(X))
		print('lenght of y: ', len(y))
		print()

		# print('First 10 records:')
		# for i in range(0, 10):
		# 	print(X[i,:], ' - class', y[i])


		unique, counts = np.unique(y, return_counts=True)
		print('Number of classes: ', len(unique))
		print('Number of samples per classes: ')
		print(np.asarray((unique, counts)).transpose())

		print('Number of features:', len(X[0,:]))

		print()
		#Descrizione statistica delle prime due features
		print(stats.describe(X[:, [0, 1]]))

		print(np.var(X[:,0]))
		print(np.var(X[:,1]))

		print('------------------------------------')
		print()

		return

	def plotGraphWithMoreRandomDatasets(self):

		iniziali = self.assignment(random_state=30)

		# Ks = [1,3,5,7]
		Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
		others = list()

		for i in range(0, 5):
			newAccuracies = self.assignment(random_state=None)
			others.append(newAccuracies)

		
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
		ax.plot(Cs, iniziali, c='blue', linestyle='-', marker='o')

		for i in range(0, 5):
			ax.plot(Cs, others[i], linestyle='-', marker='o', alpha=0.3)

		ax.set_xlabel('Value of C', labelpad=12, fontweight='bold')
		ax.set_ylabel('Accuracy', labelpad=32, rotation=0, fontweight='bold')

		ax.set_xticks(Cs)
		ax.set_xscale('log')

		plt.grid(alpha=0.25)

		plt.show()


		return



def main():

	hw1 = HW1()
	hw1.assignment(random_state=30)

	# hw1.plotGraphWithMoreRandomDatasets()

if __name__ == '__main__':
	main()