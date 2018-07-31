# Abstract

Poker is a popular game which is played both online and in casinos globally. Online poker games are increasing on a daily basis and it has turn out to be a very profitable business. Unlike chess, GO and other similar board games, Poker is stochastic in nature which means that it doesn’t offers complete information at a given point in the game. Poker strategies vary from place, person, mood, strengths, stakes and other non-verbal and non-indicative factors. 

Due to this  nature of Poker it has always been a real challenging problem to the artificial intelligence research. Over the years many different approaches has been applied to create a perfect Poker player which can win over any human opponent but to this data there is no artificial Poker player that matches the best human players. 



# Motivation

The motivation behind opponent modeling lies in the notion that once we know how to read and understand the profile of an opponent we can easily figure out their game play and strategy. Knowledge about the opponent can boost ones’ win rate significantly and help increase the profits.  Opponent modeling can be rule based or learning based. More games we have against one player more we can know its strategy. Most players use a static win strategy for their games and data logs coupled with AI techniques can help find his/her playing style making it easier to defeat.  



# Introduction

Poker is a test bed for AI research since it is a game of imperfect knowledge. Imperfect information since the cards for other players are not known. Makes it more difficult to analyze and predict compared with chess,GO etc. Usually a Poker game consists of more than 2 players. Element of deception is very common in Poker and any experienced player using bluff can easily defies fixed strategies.
For this project we worked with Texas’Holdem variation of Poker for this application. Texas Holdem is the most frequently offered in casinos and is often played in Poker tournaments. In this variant five cards (community cards) are placed in the middle of the table by each player to form his poker hand, while each player but no longer can use as two of his hand cards. 

The first part of the project revolves around importing, extracting, parsing and storing data obtained from various freely available sources. Then general rules, statistics and recognized metrics are applied on this data which counts for the opponent modeling part. Finally, based on the data from the previous stages next moves in a particular part of the game is predicted. The prediction takes in consideration the state of the game, players involved the the player who has to take the next move. 

# Goals

The primary goal of this application is to test AI methods against predicting task in Poker and predict the outcome or the next move of any given poker hand. Currently this feature is only available with casino logs data obtained from HandsHQ but using data parsing techniques the same model can be applied to predict next move given Hand data from any arbitary source. 

Other than that, this application also informs the user about the player type and player style of all opponents which can be used to create a dashboard or a stats menu that can aid in playing Poker.  



# Technology
Python (Django framework v2.0.3)
Postgresql v9.5.12
Pandas, matplotlib & seaborn(datavisualization), scikit learn (for ML)
Random Forest Tree Classifiers
Python PostgresSQL stack is used for this application. Django allows to push bulk datasets in databases and has easily integrable solution with relational databases. 
Details of the python libraries used:
NumPy
NumPy is a Numeric Python module. It provides fast mathematical functions. Numpy provides robust data structures for efficient computation of multi-dimensional arrays & matrices.We used numpy to read data files into numpy arrays and data manipulation.
Pandas
Provides DataFrame Object for data manipulation. Provides reading & writing data b/w different files.
DataFrames can hold different types data of multidimensional arrays.
 Scikit-Learn
It’s a machine learning library. It includes various machine learning algorithms.
DecisionTreeClassifier,
accuracy_score algorithms.
