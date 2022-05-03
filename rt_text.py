# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 02:07:18 2021

@author: Richard Pincus

Rotten Tomatoes Reviews Text Analytics
"""

#Load libraries
import string
from datetime import datetime as dt
import nltk
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from pandasql import sqldf  ##Based on SQLite
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
# from nltk.sentiment import vader


#Display options
pd.set_option('display.max_column', 100)
pd.set_option('display.max_row', 24000)


#Set directory
os.getcwd()
os.chdir('C:\\Users\\Richard Pincus\\Documents\\Classes - MSA\\AA502\\Text Analytics\\Project\\rt_reviews')


#Read in data
data1 = pd.read_csv('rt_reviews.csv')
data1.columns


#Check out data
pd.crosstab(index=data1['genres'], columns='count')
pd.crosstab(index=data1['review_type'], columns='count')
print(data1[0:10])


#Subset data for code running purposes, comment out this line later
# data = data[0:100]



#Subset data to movies released Jan 1 2020 and after
current_movies = []
for i in range(len(data1['original_release_date'])):
    current_movies.append(dt.strptime(data1['original_release_date'][i], "%Y-%m-%d") >= dt.strptime("2019-01-01", "%Y-%m-%d"))
    # print(dt.strptime(data['original_release_date'][i], "%Y-%m-%d") >= dt.strptime("2020-01-01", "%Y-%m-%d"))

#Subset data
data = data1.loc[current_movies]

#Subset even smaller for debugging code
# data = data[1:100]

#Reset indeces
data['rownum'] = (range(len(data)))
data = data.set_index('rownum')



#Get list of unique titles across all movies
movie_list = []
for i in range(len(data['movie_title'])):
    if data['movie_title'][i].strip() not in movie_list:
        movie_list.append(data['movie_title'][i].strip())
# print(movie_list)

#Combine all reveiws per movie
#Initialize objects
comb_reviews = '' #to hold all reviews for a movie combined
col1_list = []    #to hold all movie titles going in col1
col2_list = []    #to hold all movie reviews going in col2
#loop through movies
for title in movie_list:
    #loop through rows
    for j in range(len(data['movie_title'])):
        #check if we have the current movies
        if data['movie_title'][j] == title:
            if comb_reviews == '':
                comb_reviews = data['review_content'][j]
            else:
                comb_reviews = comb_reviews + ', ' + data['review_content'][j]
    
    #save reviews data
    col2_list.append(comb_reviews)
    #reinitialize reveiws
    comb_reviews = ''
    #save movie title data
    col1_list.append(title)


#Create new data frame that is structured by movie
movies_df = pd.DataFrame(col1_list, columns=['movie_title'])    
movies_df['review_content'] = col2_list  
# print(movies_df)
            
            
            

#Get list of unique genres across all movies
genre_complex_list = []
for i in range(len(data['genres'])):
    if data['genres'][i].strip() not in genre_complex_list:
        # print(data['genres'][i].split(',')[j].strip())
        genre_complex_list.append(data['genres'][i].strip())
# print(genre_complex_list)


#Get list of single unique genres across all movies
genre_list = []
for i in range(len(data['genres'])):
    for j in range(len(data['genres'][i].split(','))):
        if data['genres'][i].split(',')[j].strip() not in genre_list:
            # print(data['genres'][i].split(',')[j].strip())
            genre_list.append(data['genres'][i].split(',')[j].strip())
# print(genre_list)

#Create dictionary of lists that contain the indicator variables for each genre for each movie
movie_list2 = movie_list.copy()
# genre_df = pd.DataFrame(genre_list, columns=['genres'])
genre_dict = {}
movie_genres = []
movie_genres_final = np.zeros(len(genre_list))
for i in range(len(data)): 
    if data['movie_title'][i].strip() in movie_list2:
        print(data['movie_title'][i])
        movie_list2.remove(data['movie_title'][i])
        print('Removed!')
        movie_genres = []
        for g in data['genres'][i].split(','):
            this_genre = []
            for j in range(len(genre_list)):
                if g.strip() == genre_list[j]:
                    gcount = 1
                else:
                    gcount = 0
                this_genre.append(gcount)
                # movie_genres.append(this_genre)
                # print(movie_genres)
                if j==(len(genre_list)-1):
                    movie_genres.append(this_genre)
                    movie_genres_final = np.sum(movie_genres, axis=0)
    genre_dict[str(data['movie_title'][i])] = list((movie_genres_final))


print(len(movie_list2)==0)
# print(genre_dict)

genre_vars = pd.DataFrame.from_dict(genre_dict)
#Add genres as a column to use as an index
genre_vars['genres'] = genre_list
genre_vars = genre_vars.set_index('genres')
#tranpose to get genres as indicator vars
genre_vars_final = genre_vars.transpose()
# print(genre_vars_final)



#Collapse genre column to the first genre in the list for each
# genres = []
# for i in range(len(data['genres'])):
#     genres.append(data['genres'][i].split(',')[0])


#Create first_genre column in data
# data['first_genre'] = genres

#Get number of genres
# set(genres)
# n_clust = len(set(genres))


#Get reviews in a list
reviews = list(movies_df['review_content'])
type(reviews)

# for i in range(10):
#     print(reviews[i])
    

#make lower case and remove punctuation
reviews1=[]
reviews2=[]
for i in range(len(reviews)):
    reviews1.append(reviews[i])
    reviews2 = reviews1
    reviews1[i] = reviews[i].lower()
    reviews2[i] = reviews1[i].translate(str.maketrans('', '', string.punctuation))


#Initialize term vector
rev_vec = [ ]

# #Create review vector
# for r in reviews2:
#     rev_vec.append( ( r.split() ) )



# #Remove stop words
# stop_words = nltk.corpus.stopwords.words( 'english' )

# for i in range( 0, len( reviews2 ) ):
#     term_list = [ ]

#     for term in term_vec[ i ]:
#         if term not in stop_words:
#             term_list.append( term )

#     reviews[i] = term_list



#define Porter stemming
porter = nltk.stem.porter.PorterStemmer()
    

#Stem terms in reviews
# for i in range(len(reviews2)):
#     for j in range(len(term_vec[i])):
#         term_vec[i][j] = porter.stem(term_vec[i][j])
        
stems = { }

for i in range( 0, len( reviews2 ) ):
    tok = reviews2[ i ].split()
    for j in range( 0, len( tok ) ):
        if tok[ j ] not in stems:
            stems[ tok[ j ] ] = porter.stem( tok[ j ] )
        tok[ j ] = stems[ tok[ j ] ]

    reviews2[ i ] = ' '.join( tok )


# Remove empty reviews after stop word removal
i = 0
while i < len( reviews2 ):
    if len( reviews2[ i ] ) == 0:
        del reviews2[ i ]
    else:
        i += 1


# Convert frequencies to TF-IDF values, get cosine similarity
# of all pairs of documents
tfidf = TfidfVectorizer( stop_words='english', max_df=0.8, max_features=1000 )
term_vec = tfidf.fit_transform( reviews2 )
X = cosine_similarity( term_vec )



#Try k=1 to 18 (number of genres) for KMEANS clusters
distortions = []
silhs = []
for k in range(1,18):
    clust = KMeans( n_clusters=k, random_state=1 ).fit( X )
    distortions.append(clust.inertia_)
    silhs.append(clust)
    # print( clust.labels_ )


#Create Elbow plot to pick K clusters
plt.figure(figsize=(16,8))
plt.plot(range(1,18), distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#3 probably looks the best to me from this plot


#Create Silhouette Plot to pick K clusters
silh_scores = []
l1 = []
l2 = []
for i in range(1,17):
    # l1 = []
    print("---------------------------------------")
    print(silhs[i])
    print("Silhouette score:",silhouette_score(X, silhs[i].predict(X)))
    score = silhouette_score(X, silhs[i].predict(X))
    l1.append(i)
    l2.append(score)


plt.plot(l1, l2)
plt.xlabel('K Clusters')
plt.ylabel('Silhouette Score')
plt.title('Sihlouette Scores (Pick local Maxima)')

#3 still looks the best to me from this plot



##
##
## Final Clustering
##
##



clust = KMeans( n_clusters=3, random_state=1 ).fit( X )
# print( clust.labels_ )

# for i in range( 0, len( set( clust.labels_ ) ) ):
#     print( f'Cluster {i}:' )

#     for j in range( 0, len( clust.labels_ ) ):
#         if clust.labels_[ j ] == i:
#             print( movies_df[ j ].replace( '"', '' ).strip() )

#     # for j in range( 0, len( clust.labels_ ) ):
#     #     if clust.labels_[ j ] == i:
#     #         print( genres[ j ].replace( '"', '' ).strip() )
#     print()
    
    
#Append cluters to final dataset
genre_vars_final['cluster'] = clust.labels_

#Get clusters by movie
genre_vars_final['movie_title'] = genre_vars_final.index
movie_cluster = genre_vars_final[['movie_title','cluster']]
movie_cluster = movie_cluster.reset_index(drop=True)
    
#Get clusters by genre
#Group by cluster and aggregate genres in each cluster
cluster_genres = genre_vars_final.groupby(by='cluster').sum()
genre_cluster = cluster_genres.transpose()
genre_cluster['genre'] = genre_cluster.index
genre_cluster['cluster1'] = genre_cluster[0]
genre_cluster['cluster2'] = genre_cluster[1]
genre_cluster['cluster3'] = genre_cluster[2]


#Read in sentiment data from Kari
os.getcwd()
os.chdir('C:\\Users\\Richard Pincus\\Documents\\Classes - MSA\\AA502\\Text Analytics\\Project')
#Read in sentiment data
by_reveiw = pd.read_csv('sentimentvals.csv')
by_reveiw.columns
by_movie = pd.read_csv('bymovie.csv')
by_movie.columns
by_genre = pd.read_csv('bygenre.csv')
by_genre.columns
    
#Merge sentiment from Kari onto clusters
genre_cluster_sent = sqldf("""
      select a.genre, a.cluster1, a.cluster2, a.cluster3, b.genre_mean
          from genre_cluster as a
          inner join by_genre as b
          on a.genre = b.genres
          ;
      """)
      
      
movie_cluster_sent = sqldf("""
      select a.movie_title, a.cluster, b.movie_sentiment
          from movie_cluster as a
          inner join by_movie as b
          on a.movie_title = b.movie_title
          ;
      """)










##
##
## Try 18 clusters for fun
##
##



clust18 = KMeans( n_clusters=18, random_state=1 ).fit( X )


#Append cluters to final dataset
genre_vars_final18 = genre_vars_final
genre_vars_final18['cluster'] = clust18.labels_

#Get clusters by movie
genre_vars_final18['movie_title'] = genre_vars_final18.index
movie_cluster18 = genre_vars_final18[['movie_title','cluster']]
movie_cluster18 = movie_cluster18.reset_index(drop=True)

#Get clusters by genre
#Group by cluster and aggregate genres in each cluster
cluster_genres18 = genre_vars_final18.groupby(by='cluster').sum()
genre_cluster18 = cluster_genres18.transpose()
genre_cluster18['genre'] = genre_cluster18.index
for i in range(18):
    genre_cluster18[f"cluster{i+1}"] = genre_cluster18[i]
    genre_cluster18 = genre_cluster18.drop([i], axis=1).reset_index(drop=True)
genre_cluster18







#Visualize reviews in PCA world
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(X)

#Get X and Y axes
x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

#Plot color by cluster and label with movie title
plt.scatter(x_axis, y_axis, c=clust.labels_)
plt.title('Clustered Reviews in 2D PCA')

for i, label in enumerate(movie_list):
    plt.annotate(label, (x_axis[i], y_axis[i]))

plt.show()




###Get dataset with only movie and genre
movie_genre_df = sqldf("""
                          select distinct movie_title, genres
                              from data;
                        """)
                        
random.seed(18)
for i in range(len(movie_genre_df)):
    n = len(movie_genre_df['genres'][i].split(','))
    gen = random.randint(0,n-1)
    movie_genre_df['genres'][i] = movie_genre_df['genres'][i].split(',')[gen].strip()
    
        
#Assign genres numbers for coloring
genre_list.sort()
genre_color_df = pd.DataFrame({'genre': genre_list, 'color': range(0,len(genre_list))})
    
#Merge colors onto movie_genre_df
movie_genre_df = sqldf("""
                       select a.movie_title, b.genre, b.color
                           from movie_genre_df as a
                           inner join genre_color_df as b
                           on a.genres = b.genre
                       """)
                       
#Plot color by genres
colors = movie_genre_df['color']
plt.scatter(x_axis, y_axis, c=colors)
plt.title('Clustered Reviews in 2D PCA')

# for i, label in enumerate(movie_list):
#     plt.annotate(label, (x_axis[i], y_axis[i]))

plt.show()

pd.crosstab(movie_genre_df['color'], columns='count')






##
##
## Create 2 clusters to see if sentiment gets separated
##
##



clust2 = KMeans( n_clusters=2, random_state=1 ).fit( X )


#Append cluters to final dataset
genre_vars_final2 = genre_vars_final
genre_vars_final2['cluster'] = clust2.labels_

#Get clusters by movie
genre_vars_final2['movie_title'] = genre_vars_final2.index
movie_cluster2 = genre_vars_final2[['movie_title','cluster']]
movie_cluster2 = movie_cluster2.reset_index(drop=True)

#Get clusters by genre
#Group by cluster and aggregate genres in each cluster
cluster_genres2 = genre_vars_final2.groupby(by='cluster').sum()
genre_cluster2 = cluster_genres2.transpose()
genre_cluster2['genre'] = genre_cluster2.index
for i in range(2):
    genre_cluster2[f"cluster{i+1}"] = genre_cluster2[i]
    genre_cluster2 = genre_cluster2.drop([i], axis=1).reset_index(drop=True)
    
    
    

#Merge Kari's sentiment onto 2 clustered data
sent_cluster2 = sqldf("""
                          select a.movie_title, a.cluster, b.movie_sentiment
                              from movie_cluster2 as a
                              inner join by_movie as b
                              on a.movie_title = b.movie_title
                      """)
                      
#Group by cluster and summarize sentiment
sent_cluster2_summary = sent_cluster2.groupby(by='cluster').mean('movie_sentiment')
sent_cluster2_summary





##
##
## Create 2 clusters to see if sentiment gets separated in PCA space
##
##



clust2_pca = KMeans( n_clusters=2, random_state=1 ).fit( scatter_plot_points )


#Append cluters to final dataset
genre_vars_final2_pca = genre_vars_final
genre_vars_final2_pca['cluster'] = clust2_pca.labels_

#Get clusters by movie
genre_vars_final2_pca['movie_title'] = genre_vars_final2_pca.index
movie_cluster2_pca = genre_vars_final2_pca[['movie_title','cluster']]
movie_cluster2_pca = movie_cluster2_pca.reset_index(drop=True)

#Get clusters by genre
#Group by cluster and aggregate genres in each cluster
cluster_genres2 = genre_vars_final2.groupby(by='cluster').sum()
genre_cluster2 = cluster_genres2.transpose()
genre_cluster2['genre'] = genre_cluster2.index
for i in range(2):
    genre_cluster2[f"cluster{i+1}"] = genre_cluster2[i]
    genre_cluster2 = genre_cluster2.drop([i], axis=1).reset_index(drop=True)
    
    
    

#Merge Kari's sentiment onto 2 clustered data
sent_cluster2_pca = sqldf("""
                          select a.movie_title, a.cluster, b.movie_sentiment
                              from movie_cluster2_pca as a
                              inner join by_movie as b
                              on a.movie_title = b.movie_title
                      """)
                      
#Create marker columns for plot
# marker = []
# for i in range(len(sent_cluster2_pca)):
#     if sent_cluster2_pca['movie_sentiment'][i] >= 0:
#         marker.append('o')
#     elif sent_cluster2_pca['movie_sentiment'][i] < 0:
#         marker.append('x')
                      
#Append sentiment scores to scatter_plot_points
sc_pts = pd.DataFrame(scatter_plot_points)
sc_pts[3] = by_movie['movie_sentiment']

#Separate positive and negative movies
pos_pts = sc_pts[sc_pts[3] >= 0]
neg_pts = sc_pts[sc_pts[3] < 0]
        
# pd.crosstab(marker, columns='count')

#Group by cluster and summarize sentiment
sent_cluster2_summary_pca = sent_cluster2_pca.groupby(by='cluster').mean('movie_sentiment')
sent_cluster2_summary_pca


model = KMeans(n_clusters = 2, init = "k-means++")
label = model.fit_predict(scatter_plot_points)
centers = np.array(model.cluster_centers_)
plt.figure(figsize=(10,10))
uniq = np.unique(label)

#Append clusters to PCA points
sc_pts[4] = model.labels_
sc_pts1 = np.array(sc_pts)


#Separate positive and negative movies
pos_pts = sc_pts[sc_pts[3] >= 0]
pos_pts = pos_pts.reset_index(drop=True)
pos_colors = []
for i in range(len(pos_pts)):
    if pos_pts[4][i] == 1:
        pos_colors.append('blue')
    elif pos_pts[4][i] == 0:
        pos_colors.append('orange')
        
neg_pts = sc_pts[sc_pts[3] < 0]
neg_pts = neg_pts.reset_index(drop=True)
neg_colors = []
for i in range(len(neg_pts)):
    if neg_pts[4][i] == 1:
        neg_colors.append('blue')
    elif neg_pts[4][i] == 0:
        neg_colors.append('orange')

# for i in uniq:
#     plt.scatter(sc_pts1[label == i, 0] , sc_pts1[label == i, 1] , label = i, marker='o')
#     # plt.scatter(sc_pts1[label == i and movie_sentiment<0, 0] , sc_pts1[label == i and movie_sentiment<0, 1] , label = i, marker='x')
# plt.scatter(centers[:,0], centers[:,1], marker="x", color='k')
# #This is done to find the centroid for each clusters.
# plt.legend()
# plt.show()

#scatter plot positive sentiments
plt.scatter(pos_pts[0], pos_pts[1], c=pos_colors, marker='o', label='Positive')
#scatter plot negative sentiments
plt.scatter(neg_pts[0], neg_pts[1], c=neg_colors, marker='x', label='Negative')
plt.title('Clustered Reviews in 2D PCA')
plt.legend()
plt.show()



#Export scatter plot data for viz in Tableau
sc_pts_out = pd.DataFrame(sc_pts)
for i in range(len(sc_pts_out)):
    if sc_pts_out[3][i] >= 0:
        sc_pts_out['shape'][i] = 1
    elif sc_pts_out[3][i] < 0:
        sc_pts_out['shape'][i] = 0
        
    

os.chdir('C:\\Users\\Richard Pincus\\Documents\\Classes - MSA\\AA502\\Text Analytics\\Project')

sc_pts_out.to_csv('cluster_sent.csv')

data.columns
