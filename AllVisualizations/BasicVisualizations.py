import numpy as np
import csv
import matplotlib.pyplot as plt
import operator
import seaborn as sns
import pandas as pd
from beeswarm import *


def loadRatings(fileName):
    '''
    Data comes in form User ID, Movie ID, Rating.
    Returns all data as 2d array
    '''
    ratings = []
    f = open(fileName, "r")
    
    for line in f:
        ratings.append(line.split())
        
    return np.asarray(ratings, dtype = int)    

def loadRatingsPandas(fileName):
    df = pd.read_table(fileName, header=None, names=['User', 'Movie', 'Rating'])
    return df

def loadMovies(fileName):
    '''
    Input data comes in form:
    
    Movie Id, Movie Title, Unknown, Action, Adventure, Animation, 
    Childrens, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, 
    Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western
    
    Will return all data as 2d matrix
    '''
    movies = []
    with open(fileName,'r') as f:
        reader=csv.reader(f,delimiter='\t')
        for movieData in reader:
            movies.append(movieData)
    return modifyMovieData(np.array(movies))

def returnThreeGenres(lstOfGenres, ratings, moviesArr):
    '''
    Returns data for three genres ... 'Western', 'Horror', 'Childrens'
    '''
    western = []
    horror = []
    childrens = []
    westernData = []
    horrorData = []
    childrensData = []
    
    #countsMap = dict()
    #for i in moviesArr:
        #currentMovie = i[1]
        #if currentMovie not in countsMap:
            #countsMap[currentMovie] = i[0]
            
    for i in moviesArr:
        if i[2] == 'Western':
            western.append(i[0])
        elif i[2] == 'Horror':
            horror.append(i[0])
        elif i[2] == 'Childrens':
            childrens.append(i[0])

    for i in ratings:
        if str(i[1]) in western:
            westernData.append(i[2])
        elif str(i[1]) in horror:
            horrorData.append(i[2])
        elif str(i[1]) in childrens:
            childrensData.append(i[2])
    return westernData, horrorData, childrensData
        
def returnMapOfIDtoMovie(moviesArr):
    '''
    Returns a mapping of ID to movie name
    '''
    
    mapping = dict()
    for i in moviesArr:
        if i[0] not in mapping:
            mapping[i[0]] = i[1]
    return mapping
        
    
def modifyMovieData(moviesArr):
    '''
    This function will take the movie input data and convert it to form:
    
    [Movie ID, Movie Title, Genre (string)]
    '''
    genresMap= {2: "Unknown", 3: "Action", 4: "Adventure", 5: "Animation", 
            6: "Childrens", 7: "Comedy", 8: "Crime", 9: "Documentary", 
            10: "Drama", 11: "Fantasy", 12: "Film-Noir", 13: "Horror",
            14: "Musical", 15: "Mystery", 16: "Romance", 17: "Sci-Fi", 
            18: "Thriller", 19: "War", 20:"Western"}    
    modifiedMovies = []
    for movie in moviesArr:
        lastIndex = -999
        for i, el in enumerate(movie):
            if el == '1':
                lastIndex = i
        modifiedMovies.append([movie[0], movie[1], genresMap[lastIndex]])
    return modifiedMovies

def getDataBestMovies(ratingsArr):
    '''
    This returns the data for the highest rated movies
    '''
    movieIDs = getBestMovies(ratingsArr, 10)
    hashLst = dict()
    for movie in movieIDs:
        hashLst[movie] = []
    
    for i in ratingsArr:
        if i[1] in movieIDs:
            hashLst[i[1]].append(i[2])
    topTenMovies = []
    topTenRatings = []
    for i in hashLst:
        topTenMovies.append(i)
        topTenRatings.append(hashLst[i])
    return topTenMovies, topTenRatings    
    
def getBestMovies(ratingsArr, num):
    '''
    This returns the IDs of the top highest rated movies
    '''
    ratingsMap = dict()
    countsMap = dict()
    allNumRatings = []
    lstOfValidMovies = []
    bestMovies = []
    for i in ratingsArr:
        currentMovie = i[1]
        if currentMovie not in ratingsMap:
            ratingsMap[currentMovie] = [i[2]]
            countsMap[currentMovie] = 1
        else:
            ratingsMap[currentMovie].append(i[2])
            countsMap[currentMovie] += 1
    for i in countsMap:
        allNumRatings.append(countsMap[i])
    threshold = np.percentile(allNumRatings, 50)
    for i in countsMap:
        if countsMap[i] > threshold:
            lstOfValidMovies.append(i)
    averageRatingMap = dict()
    for i in lstOfValidMovies:
        averageRatingMap[i] = np.mean(ratingsMap[i])
    for w in sorted(averageRatingMap, key=averageRatingMap.get, reverse=True):
        bestMovies.append(w)
        if len(bestMovies) == num:
            break  
    return bestMovies
        
        
    
    
def getMostPopularMovies(ratingsArr, num):
    '''
    This returns data for the top n movies which have been rated the most 
    '''
    topMovies = []
    countsMap = dict()
    for i in ratingsArr:
        currentMovie = i[1]
        if currentMovie not in countsMap:
            countsMap[currentMovie] = 1
        else:
            countsMap[currentMovie] += 1
    for w in sorted(countsMap, key=countsMap.get, reverse=True):
        topMovies.append(w)
        if len(topMovies) == 10:
            break
    
    return topMovies


    
def getDataForMostPopular(ratingsArr):
    '''
    Takes in ratings data, and returns data for top 10 most popular 
    '''
    hashLst = dict()
    movieIDs = getMostPopularMovies(ratingsArr, 10)
    for movie in movieIDs:
        hashLst[movie] = []
    
    for i in ratingsArr:
        if i[1] in movieIDs:
            hashLst[i[1]].append(i[2])
    topTenMovies = []
    topTenRatings = []
    for i in hashLst:
        topTenMovies.append(i)
        topTenRatings.append(hashLst[i])
    return topTenMovies, topTenRatings

def makeViolinPlotGenres(data, labels, title):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 4)) 
    fig.suptitle(title, fontsize=20)
    axes[0].violinplot(data[0], 
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[0].set_ylim([1,5])
    axes[0].set_title(labels[0])
    axes[0].set_xticklabels([])
    
    axes[1].violinplot(data[1],
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[1].set_title(labels[1])
    axes[1].set_xticklabels([])
    axes[1].set_ylim([1,5])
    
    axes[2].violinplot(data[2],
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[2].set_title(labels[2])
    axes[2].set_xticklabels([])
    axes[2].set_ylim([1,5])    
    return

def makeBoxPlot(labels, data_to_plot, title):
    '''
    Plots a boxplot with labels for data with title 
    '''

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    
    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    
    bp = ax.boxplot(data_to_plot, showmeans = True)
    ax.set_title(title, fontsize = 25)
    ## change outline color, fill color and linewidth of the boxes
    #for box in bp['boxes']:
        ## change outline color
        #box.set( color='#7570b3', linewidth=2)
        ## change fill color
        #box.set( facecolor = '#1b9e77' )
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    
    ax.set_xticklabels([labels[0], labels[1], labels[2], labels[3], labels[4],labels[5],
                            labels[6], labels[7], labels[8], labels[9]], rotation =30, fontsize = 12)
    plt.xlabel("Movie", fontsize = 15)
    plt.ylabel("Rating", fontsize = 15)

        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()   
    plt.tight_layout()
    return
    
def makeViolinPlot(data, labels, title):
    '''
    Makes violin plot from data given labels 
    '''
    
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(9, 4)) 
    fig.suptitle(title, fontsize=20)
    axes[0,0].violinplot(data[0], 
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[0,0].set_ylim([1,5])
    axes[0,0].set_title(labels[0])
    axes[0,0].set_xticklabels([])
    
    axes[0, 1].violinplot(data[1],
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[0,1].set_title(labels[1])
    axes[0,1].set_xticklabels([])
    axes[0,1].set_ylim([1,5])
    
    axes[1,0].violinplot(data[2],
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[1,0].set_title(labels[2])
    axes[1,0].set_xticklabels([])
    axes[1,0].set_ylim([1,5])
    
    axes[1,1].violinplot(data[3],
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[1,1].set_title(labels[3])
    axes[1,1].set_xticklabels([])
    axes[1,1].set_ylim([1,5])
    
    axes[2,0].violinplot(data[4],
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[2,0].set_title(labels[4])
    axes[2,0].set_xticklabels([])
    axes[2,0].set_ylim([1,5])
    
    
    axes[2,1].violinplot(data[5],
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[2,1].set_title(labels[5])
    axes[2,1].set_xticklabels([])
    axes[2,1].set_ylim([1,5])
    
    axes[3,0].violinplot(data[6],
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[3,0].set_title(labels[6])
    axes[3,0].set_xticklabels([])
    axes[3,0].set_ylim([1,5])
    
    axes[3,1].violinplot(data[7],
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[3,1].set_title(labels[7])
    axes[3,1].set_xticklabels([])
    axes[3,1].set_ylim([1,5])
    
    axes[4,0].violinplot(data[8],
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[4,0].set_title(labels[8])
    axes[4,0].set_xticklabels([])
    axes[4,0].set_ylim([1,5])
    
    axes[4,1].violinplot(data[9],
                       showmeans=True,
                       showmedians=False, showextrema = False)
    axes[4,1].set_title(labels[9])
    axes[4,1].set_xticklabels([])
    axes[4,1].set_ylim([1,5])    
    return

def makeBoxPlotGenres(labels, data_to_plot, title):
    data_to_plot = genresData
    labels = genresLabels
    fig = plt.figure(1, figsize=(9, 6))
    
    # Create an axes instance
    ax = fig.add_subplot(111)
    
    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    
    bp = ax.boxplot(data_to_plot, showmeans = True)
    ax.set_title(title, fontsize = 25)
    ## change outline color, fill color and linewidth of the boxes
    #for box in bp['boxes']:
        ## change outline color
        #box.set( color='#7570b3', linewidth=2)
        ## change fill color
        #box.set( facecolor = '#1b9e77' )
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    
    ax.set_xticklabels([labels[0], labels[1], labels[2]], rotation =30, fontsize = 15)
    plt.xlabel("Movie", fontsize = 18)
    plt.ylabel("Rating", fontsize = 18)
    
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()   
    plt.tight_layout()

def plotBarGenres(genres, title):
    ratingsArr = list(genres)
    barWidth = 0.8
    index = np.arange(5)
    x = ['1','2','3','4','5']
    y = [ratingsArr.count(1), ratingsArr.count(2), ratingsArr.count(3),
         ratingsArr.count(4), ratingsArr.count(5)]
    plt.bar(index,y, barWidth, color='#d62728')
    plt.title(title)
    plt.xlabel("Rating")
    plt.ylabel("Number of Movies")
    plt.xticks(index + barWidth/2, ('1', '2', '3', '4', '5'))
    
   
def plotAllRatings(ratingsArr):
    '''
    Takes in a list of numerical ratings and plots a bar graph of 
    its distribution 
    '''
    ratingsArr = list(ratingsArr)
    barWidth = 0.8
    index = np.arange(5)
    x = ['1','2','3','4','5']
    y = [ratingsArr.count(1), ratingsArr.count(2), ratingsArr.count(3),
         ratingsArr.count(4), ratingsArr.count(5)]
    plt.bar(index,y, barWidth, color='#d62728')
    plt.title("All Ratings of MovieLens Dataset")
    plt.xlabel("Rating")
    plt.ylabel("Number of Movies")
    plt.xticks(index + barWidth/2, ('1', '2', '3', '4', '5'))

    plt.show()
    return

def main():
    ratings = loadRatings("data.txt")
    movies = loadMovies("movies.txt")
    
    topMovies, topRatings= getDataForMostPopular(ratings)
    bestMovies, bestRatings= getDataBestMovies(ratings)
    
    IDtoMovie = returnMapOfIDtoMovie(movies)
    topMoviesNames = []
    bestMoviesNames = []
    
    for i in topMovies:
        topMoviesNames.append(IDtoMovie[str(i)])
    assert(len(topMoviesNames) == 10)
    
    for i in bestMovies:
        bestMoviesNames.append(IDtoMovie[str(i)])
    assert(len(bestMoviesNames) == 10)
    
    
    westernData, horrorData, childrensData = returnThreeGenres(["Western", "Children", "Horror"], ratings, movies)
    genresData = [westernData, horrorData, childrensData]
    genresLabels = ["Western", "Horror", "Childrens"]
    
    #makeBoxPlot(topMoviesNames, topRatings, "Top Ten Most Popular Movies")
    #makeBoxPlot(bestMoviesNames, bestRatings, "Top Ten Highest Rated Movies")
    #makeBoxPlotGenres(genresLabels, genresData, "Genre Comparison")
    #makeViolinPlot(topRatings, topMoviesNames, "Top Ten Most Popular Movies")
    #makeViolinPlot(bestRatings, bestMoviesNames, "Top Ten Highest Rated Movies")
    makeViolinPlotGenres(genresData, genresLabels, "Comparing Genres")
    #plotBarGenres(genresData[0], "Western Movies Ratings")
    #plotBarGenres(genresData[1], "Horror Movies Ratings")
    #plotBarGenres(genresData[2], "Childrens Movies Ratings")
    
    
main()

            



 