# CS 155 Project3 Matrix Factorization

import numpy as np
import pandas as pd
import random
import time
import math as m
from collections import Counter
import matplotlib.pyplot as plt

K = 20
l = 0.1
eta = 0.04
epsilon = 0.00001
maxIterations = 1.5 * (10 ** 6)

def loss(df, U, V, Y, M, N):
	''' Training objective '''
	total = 0
	UV = np.dot(np.transpose(U), V)
	for i in range(M + 1):
		for j in range(N + 1):
			if Y[i, j] != 0:
				total += (Y[i, j] - UV[i, j]) ** 2

	return total / 2.

def createYMatrix(df, M, N):
	''' Creates the matrix of ratings, with
	matrix[i][j] = y_ij '''
	matrix = np.zeros((M + 1, N + 1))
	for idx, row in df.iterrows():
		i = row['i']
		j = row['j']
		y = row['y']
		matrix[i, j] = y

	return matrix

def meanCenter(U, V):
	''' Centers V around its mean, and shifts the rows of
	U by the same amount as the rows of V, and returns the new
	matrices and the means '''
	copyU = np.copy(U)
	copyV = np.copy(V)
	means = [0. for i in range(K)]

	for i in range(K):
		means[i] = np.mean(copyV[i])
		copyU[i] -= means[i]
		copyV[i] -= means[i]

	return copyU, copyV, means

def load_movies():
	''' Loads the movie names. The index of list
	matches the index of the movie '''
	lst = [0]
	f = open('movies.txt', 'r', encoding = "ISO-8859-1")
	for line in f:
		lst.append(line.split('\t')[1])
	f.close()
	return lst

def get_top_movies(data):
	# Object for tallying
	c = Counter(data['j']) 
	# Ten movies with the most ratings
	topTenPopular = [x[0] for x in c.most_common(10)]

	# List of all movie IDs with how many times they're rated
	completeList = c.most_common(len(c))

	# Take the median of the numbers of ratings
	median = np.percentile([x[1] for x in completeList], 50)

	# Only take (indices of) movies with more than the median number of ratings
	aboveMedian = [x[0] for x in completeList if x[1] >= median]

	# Create a dictionary mapping each movie in aboveMedian to the mean of its list of ratings
	dic = {x : np.mean(list(data[data['j'] == x]['y'])) for x in aboveMedian}

	# Sort the keys and take the top 10 best rated
	topTenRated = []
	for w in sorted(dic, key=dic.get, reverse=True):
		topTenRated.append(w)
		if len(topTenRated) == 10:
			break

	return topTenPopular, topTenRated

def main():
	data = pd.read_table('data.txt', names=['i', 'j', 'y']) # Load ratings info
	movieData = load_movies()								# Load movie names
	M = max(data['i'])										# Largest user index
	N = max(data['j'])										# Largest movie index								
	topTenPopular, topTenRated = get_top_movies(data)       # Get the most and best rated movies
	L = len(data)											# The number of ratings total (100000)
	errors = []												# List of errors
	yMatrix = createYMatrix(data, M, N)						# yMatrix[i, j] = y_ij


	# Initialize the U and V matrices
	U = np.random.uniform(low=-0.5, high=0.5, size=(K, M + 1))
	V = np.random.uniform(low=-0.5, high=0.5, size=(K, N + 1))

	# Get initial loss, so the stopping condition works on first iteration
	errors.append(loss(data, U, V, yMatrix, M, N))

	# SGD loop
	iterations = 0
	while True:
		iterations += 1

		# Get a random user,rating pair from the data
		pt = data.iloc[random.randint(0, L - 1)]
		i = pt[0]
		j = pt[1]

		# Update the U and V at the i-th and j-th columns respectively by subtracting the gradient
		U[:, i] -= eta * (l * U[:, i] - (V[:, j] * (yMatrix[i, j] - np.dot(np.transpose(U[:, i]), V[:, j]))))
		V[:, j] -= eta * (l * V[:, j] - (U[:, i] * (yMatrix[i, j] - np.dot(np.transpose(U[:, i]), V[:, j]))))

		# Check for error roughly once an epoch
		if iterations % 100000 == 0:
			errors.append(loss(data, U, V, yMatrix, M, N))
			print(errors[-1])
			if iterations == maxIterations or m.fabs(errors[-1] - errors[-2]) / m.fabs(errors[1] - errors[0]) <= epsilon:
				print(iterations)
				break

	# Transform V so that the mean of each of its rows is 0
	newU, newV, means = meanCenter(U, V)

	# Perform SVD on V
	A, s, B = np.linalg.svd(newV)

	# Project the first two columns of A onto U and V
	vTilde = np.dot(np.transpose(A[:, :2]), newV)
	uTilde = np.dot(np.transpose(A[:, :2]), newU)

	##
	## Plot top ten most popular movies
	##
	# Each column defines a particular movies' coordinates
	# xCommon = [vTilde[0][idx] for idx in topTenPopular]
	# yCommon = [vTilde[1][idx] for idx in topTenPopular]

	# # Define the range to plot
	# xMax = max(np.abs(xCommon))
	# yMax = max(np.abs(yCommon))
	# xRange = xMax + 0.1 * xMax
	# yRange = yMax + 0.1 * yMax

	# # Scatter plot
	# plt.figure(1)
	# plt.scatter(xCommon, yCommon)
	# # Set x and y ranges
	# axes = plt.gca()
	# axes.set_xlim([-xRange, xRange])
	# axes.set_ylim([-yRange, yRange])
	# # Get movie names by their index
	# xy = list(zip(xCommon, yCommon))
	# for i in range(len(xy)):
	# 	axes.annotate('%s' % movieData[topTenPopular[i]], xy=xy[i], textcoords='data')
	# plt.margins(y = 0.02)
	# # Draw gridlines on x and y axes.
	# plt.axhline(0)
	# plt.axvline(0)
	# plt.show()

	##
	## Plot top ten highest rated moves
	##
	# Each column defines a particular movies' coordinates
	xBest = [vTilde[0][idx] for idx in topTenRated]
	yBest = [vTilde[1][idx] for idx in topTenRated]

	# Define the range to plot
	xMax = max(np.abs(xBest))
	yMax = max(np.abs(yBest))
	xRange = xMax + 0.1 * xMax
	yRange = yMax + 0.1 * yMax

	# Scatter plot
	plt.figure(2)
	plt.scatter(xBest, yBest)
	# Set x and y ranges
	axes = plt.gca()
	axes.set_xlim([-xRange, xRange])
	axes.set_ylim([-yRange, yRange])
	# Get movie names by their index
	xy = list(zip(xBest, yBest))
	for i in range(len(xy)):
		axes.annotate('%s' % movieData[topTenRated[i]], xy=xy[i], textcoords='data')
	plt.margins(y = 0.02)
	# Draw gridlines on x and y axes.
	plt.axhline(0)
	plt.axvline(0)
	plt.show()

	##
	## Plot ten chosen movies
	##
	# Each column defines a particular movies' coordinates
	# chosenMovies = [820, 227, 228, 229, 230, 50, 127, 187, 182, 214]
	# chosenMovies = [177, 661, 246, 203, 435, 589, 519, 510, 241, 1454]
	# chosenMovies = [200, 288, 219, 185, 675, 671, 98, 447, 176, 201]
	# chosenMovies = [588, 423, 1, 95, 465, 71, 132, 151, 419, 143]
	# xChosen = [vTilde[0][idx] for idx in chosenMovies]
	# yChosen = [vTilde[1][idx] for idx in chosenMovies]


	# # Define the range to plot
	# xMax = max(np.abs(xChosen))
	# yMax = max(np.abs(yChosen))
	# xRange = xMax + 0.1 * xMax
	# yRange = yMax + 0.1 * yMax

	# # Scatter plot
	# plt.figure(2)
	# plt.scatter(xChosen, yChosen)
	# # Set x and y ranges
	# axes = plt.gca()
	# axes.set_xlim([-xRange, xRange])
	# axes.set_ylim([-yRange, yRange])
	# # Get movie names by their index
	# xy = list(zip(xChosen, yChosen))
	# for i in range(len(xy)):
	# 	axes.annotate('%s' % movieData[chosenMovies[i]], xy=xy[i], textcoords='data')
	# plt.margins(y = 0.02)
	# # Draw gridlines on x and y axes.
	# plt.axhline(0)
	# plt.axvline(0)
	# plt.show()

if __name__ == '__main__':
	main()