import os
import pickle
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from urllib.request import urlopen
from sklearn.ensemble import RandomForestRegressor

import psycopg2 as psycopg2
from contextlib import closing
from django.db import connection
from django.utils import timezone
import psycopg2

#from django.db import models
#from pokerApp.models import *

#conn = psycopg2.connect("host=localhost dbname=poker user=hani password=hanihani1 ")
#cur = conn.cursor()
#def insert_game():
#    Game.objects.create(number_of_hands=10, game_type="Springsteen",duration= "long")
#cur.execute("INSERT INTO pokerApp_game VALUES (%s, %s, %s)", ('1', 'hello', 'long'))
#conn.commit()
#insert_game()

# for p in Game.objects.raw('SELECT * FROM myapp_person'):
#     print (p)

def countNumberOfHands(freshData,filename):

    #newFile = open("/home/syed/Documents/AdvDI/PokerDataFromHandsHQ/ong NLH handhq_1-OBFUSCATED.txt",
    #           'r')
    #print(filename)
    Hands = {}
    Hands["File_Name"] = filename.split('-')[-1].split('.txt')[0]
    #freshData = newFile.read()
    newData = freshData.split("\n\n\n\n")
    j = 0
    test = []
    count= 0
    for i in newData:
        if i: #and count < 500:
            count +=1
            moves = []
            listy = {}
            positions = []
            winner = {}
            showdown = False
            preflop = flop = turn = river = False
            phase = ""

            hand = i.split("\n")

            #print(i)
            for k in range(len(hand)):
                                if "pocket cards" in hand[k]:
                                    preflop = True
                                    flop = False
                                    turn = False
                                    river = False
                                elif "Dealing flop" in hand[k]:
                                    #listy["Phase"] = "Flop"
                                    flop = True
                                    preflop = False
                                    turn = False
                                    river = False
                                elif "Dealing turn" in hand[k]:
                                    #listy["Phase"] = "Turn"
                                    turn = True
                                    preflop = False
                                    flop = False
                                    river = False
                                elif "Dealing river" in hand[k]:
                                    #listy["Phase"] = "River"
                                    river = True
                                    preflop = False
                                    flop = False
                                    turn = False

                                if preflop == True:
                                    listy["Phase"] = "Preflop"
                                    phase = "Preflop"
                                elif flop == True:
                                    listy["Phase"] = "Flop"
                                    phase = "Flop"
                                elif turn == True:
                                    listy["Phase"] = "Turn"
                                    phase = "Turn"
                                elif river == True:
                                    listy["Phase"] = "River"
                                    phase = "River"
                                else:
                                    listy["Phase"] = "None"
                                    phase = "PreFlop"
                                #listy["Phase"] = {"Preflop": preflop, "Flop": flop, "Turn": turn, "River": river}

                                if not hand[0] == " ":
                                    listy["HandID"] = hand[0].split(' ')[4]
                                    listy["TimeStamp"] = hand[1].split('hand:')[1]
                                    listy["GameType"] = hand[2].split('(')[1].split(' ')[0] + " " +hand[2].split('(')[1].split(' ')[1]
                                    listy["TableName"] = hand[2].split(' ')[1]
                                    listy["Button"] = hand[3].split(' ')[2]
                                    listy["Stakes"] = hand[2].split(" ")[-3]
                                    listy["PlayersInRound"] = hand[4].split(" ")[4]
                                if "posts small blind" in hand[k]:
                                           listy["SmallBlind"] = hand[k].split(' ')[0]
                                if "posts big blind" in hand[k]:
                                           listy["BigBlind"] = hand[k].split(' ')[0]
                                if hand[k].split(' ')[0] == "Seat":
                                         if "net" not in hand[k]:
                                             positions.append({"Seat": hand[k].split(' ')[1], "Player": hand[k].split(' ')[2],
                                                               "StackSize": hand[k].split(' ')[3][1:-2]})
                                         elif "net" in hand[k]:
                                             pass
                                if "folds" in hand[k] or "checks" in hand[k]:
                                         moves.append({"Player": hand[k].split(' ')[0], "Action": hand[k].split(' ')[1],
                                                  "Phase": phase})
                                elif "bets" in hand[k]  or  "calls" in hand[k]:
                                         moves.append({"Player": hand[k].split(' ')[0], "Action": hand[k].split(' ')[1],
                                                  "Phase": phase, "Amount": hand[k].split(' ')[2]})
                                elif "raises" in hand[k]:
                                         moves.append({"Player": hand[k].split(' ')[0], "Action": hand[k].split(' ')[1],
                                                  "Phase": phase, "Amount": hand[k].split(' ')[4], "Previous Amount":hand[k].split(' ')[2]})
                                if "flop" in hand[k]:
                                          listy["Flop"] = hand[k].split(" [")[1][:-1]
                                if "turn" in hand[k]:
                                          listy["Turn"] = hand[k].split(" ")[3][1:-1]
                                if "river" in hand[k]:
                                    listy["River"] = hand[k].split(" ")[3][1:-1]
                                    showdown = True
                                if "won" in hand[k]:
                                          #if hand[k].split(" ")[5]:
                                          winner["Player"] = hand[k].split(" ")[5]
                                          winner["Amount"] = hand[k].split(" ")[2]

            listy["Showdown"] = showdown
            listy["Positions"] = positions
            listy["Moves"] = moves
            listy["Winner"] = winner
            test.append(listy)

    #print (len(test))
    Hands["Hands"] = test
    print(Hands)
    return Hands

#countNumberOfHands()

def totalCounts():
        path = "/home/syed/Documents/AdvDI/PokerDataFromHandsHQ/"
        totalFiles = 0
        totalHands = 0
        count = 0
        for filename in os.listdir(path):
            if count < 1:
                totalFiles = totalFiles + 1
                newData = open(path + filename, 'r')
                tempFileData = newData.read()
                result = countNumberOfHands(tempFileData, filename)
                totalHands = totalHands + len(result["Hands"])
                count +=1

        print("Total Hands: ",totalHands)
        print("Total Files: ",totalFiles)
        return result

result = totalCounts()

### Player Table ####  Everything from here also to do in source one #######################################################
count = 0
Players = []
PlayersGameCount = []
PlayersHandCount = []

for i in result["Hands"]:
    for j in i["Positions"]:
        #print(j["Player"])
        if j["Player"] not in Players:
            Players.append(j["Player"])
            PlayersGameCount.append({'Name':j["Player"],'games':1})
        elif j["Player"] in Players:
            for k in range(len(Players)):
                if j["Player"] == PlayersGameCount[k]["Name"]:
                    PlayersGameCount[k]["games"] +=1

# print(PlayersGameCount)
#print(Players)
# count = 0
# for i in result["Hands"]:
#     for j in i["Positions"]:
#         if j["Player"] in PlayersGameCount[count]["Name"]:
#                 PlayersGameCount[count]["games"] +=1
#         count +=1

# {Player:PhasesSeen,Preflop,Flop,Turn,River,Winner,bets,calls,raise,check,fold,moneywon,avrgInitialStakes,AvrgNetProfitORLoss}

for i in range(len(Players)):
        PlayersHandCount.append({"Name":Players[i],"Bets":0,"Calls":0,"Raise":0,"Check":0,"Fold":0,"Win":0,"MoneyWon":0})
for k in result["Hands"]:
    for j in k["Moves"]:
        for l in range(len(PlayersHandCount)):
            if j["Player"] == PlayersHandCount[l]["Name"]:
                if j["Action"] == "folds":
                    PlayersHandCount[l]["Fold"] += 1
                if j["Action"] == "raises":
                    PlayersHandCount[l]["Raise"] += 1
                if j["Action"] == "calls":
                    PlayersHandCount[l]["Calls"] += 1
                if j["Action"] == "bets":
                    PlayersHandCount[l]["Bets"] += 1
                if j["Action"] == "checks":
                    PlayersHandCount[l]["Check"] += 1

#print(PlayersHandCount)
# print(len(Players))
# print(PlayersHandCount)
# {Player:PhasesSeen,Preflop,Flop,Turn,River,Winner,bets,calls,raise,check,fold,moneywon,avrgInitialStakes,AvrgNetProfitORLoss}

PlayerPhases = []
PhasesSeen = []
for i in range(len(Players)):
    PlayerPhases.append({"Name":Players[i],"Preflop":0,"Flop":0,"Turn":0,"River":0,"Win":0,"MoneyWon":0,
                                 "AvrgInitialStakes":0,"AvrgNetProfitOrLoss":0, "slansky":"", "Percentage_of_hands_played":0})
    PhasesSeen.append({"Name":Players[i],"Preflop":"0","Flop":"0","Turn":"0","River":"0","Phase":""})

#print(len(PlayerPhases))

phase = []
for j in result["Hands"]:
        for l in range(len(PlayersHandCount)):
            if j["Winner"]["Player"] == PlayersHandCount[l]["Name"]:
                PlayersHandCount[l]["Win"] += 1
                PlayersHandCount[l]["MoneyWon"] += int(j["Winner"]["Amount"].split("$")[1].split(".")[0])
                break

        for m in j["Moves"]:
            p = m["Player"]
            a = m["Phase"]
            #print(m)
            #print("Player: " + str(count) + " " + p)
            #print("Phase: " + a)
            if p not in phase:
                phase.append(p)
                phase.append(1)
                phase.append({"Preflop":0,"Flop":0,"Turn":0,"River":0})
            else:
                phase[phase.index(p)+1] += 1
                pass

        for o in range(len(phase)):
            if phase[o] == 1:
                    phase[o+1]["Preflop"] +=1
                    phase[o] = 0
            if phase[o] == 2:
                    phase[o + 1]["Preflop"] += 1
                    phase[o + 1]["Flop"] += 1
                    phase[o] = 0
            if phase[o] == 3:
                    phase[o + 1]["Preflop"] += 1
                    phase[o + 1]["Flop"] += 1
                    phase[o + 1]["Turn"] += 1
                    phase[o] = 0
            if phase[o] == 4:
                    phase[o + 1]["Preflop"] += 1
                    phase[o + 1]["Flop"] += 1
                    phase[o + 1]["Turn"] += 1
                    phase[o + 1]["River"] += 1
                    phase[o] = 0

        #print(phase)
#print(phase)

### Slansky Classification ####
#   Slansky Classification: tight: 28% or less hands played, loose > 28%,

for j in range(len(phase)):

            if type(phase[j]) is dict and phase[j]["Preflop"] is not 0:
                slansky = (phase[j]["Flop"]/phase[j]["Preflop"])*100
                if slansky > 28:
                    phase[j]["HandPlayed%"] = slansky
                    phase[j]["Slansky"] = "Loose"
                else:
                    phase[j]["HandPlayed%"] = slansky
                    phase[j]["Slansky"] = "Tight"

###        Aggression Factor ###
#AF: (num_bets + num_raises)/(num_calls), AF> 1 == Agressive, AF< 1 == Passive

for meow in range(len(PlayersHandCount)):
            if PlayersHandCount[meow]["Calls"] == 0:
                PlayersHandCount[meow]["Agression"] = 0
            else:
                PlayersHandCount[meow]["Agression"] = (PlayersHandCount[meow]["Bets"] + PlayersHandCount[meow][
                    "Raise"]) / PlayersHandCount[meow]["Calls"]

#print(len(phase)/3)
#print(PlayersHandCount)
#print(PlayerPhases)

### Hand Table ###
### Move Table ###
### Game Table ###
### Players in Game ###
### Player ###

### Game Variables ###

# (For N-1 Move)
# Position,
# Hand Strength
GameVariables =  []
# Games won % in PlayersHandCount
for i in range(len(PlayersHandCount)):
    if PlayersHandCount[i]["Name"] == PlayersGameCount[i]["Name"]:
        PlayersHandCount[i]["Games_won_%"] = float("{0:.2f}".format((PlayersHandCount[i]["Win"] / PlayersGameCount[i]["games"]) * 100))

# Pecentage of check/calls, bets/raises, folds in PlayersHandCount
for i in range(len(PlayersHandCount)):
    Total = PlayersHandCount[i]["Bets"] + PlayersHandCount[i]["Calls"] + PlayersHandCount[i]["Raise"] + PlayersHandCount[i]["Check"] + PlayersHandCount[i]["Fold"]
    if Total > 0:
        PlayersHandCount[i]["Check/Calls_%"] = float("{0:.2f}".format((PlayersHandCount[i]["Calls"] + PlayersHandCount[i]["Check"])/ Total *100))
        PlayersHandCount[i]["Bets/Raises_%"] = float("{0:.2f}".format((PlayersHandCount[i]["Raise"] + PlayersHandCount[i]["Bets"] )/ Total *100))
        PlayersHandCount[i]["Fold_%"] = float("{0:.2f}".format(PlayersHandCount[i]["Fold"]/ Total *100))

# Current Move, Last Move
for j in result["Hands"]:
    for i in j["Moves"]:

            # Percentage of Actions #
            # Win Percentage #
            # Slansky #
            # Aggressive #

    # Move # Fold = 1, Bet,Raise = 2, Check,Calls = 3
            if i["Action"] == "folds":
                i["Action"] = 1
            elif i["Action"] == "bets" or i["Action"] == "raises":
                i["Action"] = 2
            elif i["Action"] == "calls" or i["Action"] == "checks":
                i["Action"] = 3
    # Phase # Preflop= 1, Flop= 2, Turn= 3, River= 4
            if i["Phase"] == "Preflop":
                i["Phase"] = 1
            elif i["Phase"] == "Flop":
                i["Phase"] = 2
            elif i["Phase"] == "Turn":
                i["Phase"] = 3
            elif i["Phase"] == "River":
                i["Phase"] = 4

    # Slansky # (Loose =0, Tight =1)
            #countz = -1
            for l in range(len(phase)):
                if i["Player"] == phase[l]:
                    if 'Slansky' in phase[l+2].keys():
                        if phase[l+2]['Slansky'] == "Loose":
                            #slanskies = 0
                            i["Slansky"] = 0
                        elif phase[l+2]['Slansky'] == "Tight":
                            #slanskies = 1
                            i["Slansky"] = 1
                    else:
                        i["Slansky"] = 0


    # Percentage of Actions #
    # Win Percentage #
    # Aggressive # Aggression =1, Passive =0
            for k in range(len(PlayersHandCount)):
                if i["Player"] == PlayersHandCount[k]["Name"]:
                    i["Check/Calls_%"] =  PlayersHandCount[k]["Check/Calls_%"]
                    i["Bets/Raises_%"] =  PlayersHandCount[k]["Bets/Raises_%"]
                    i["Fold_%"] = PlayersHandCount[k]["Fold_%"]
                    i["Win"] = PlayersHandCount[k]["Games_won_%"]
                    if PlayersHandCount[k]["Agression"] < 1:
                        i["Aggression"] = 0
                    if PlayersHandCount[k]["Agression"] >= 1:
                        i["Aggression"] = 1

    for i in j["Moves"]:
        if j["Moves"].index(i) < len(j["Moves"]) - 1:
                GameVariables.append({"Current_Move":j["Moves"][j["Moves"].index(i)]["Action"],"Last_Move":j["Moves"][j["Moves"].index(i)-1]["Action"], "Output":j["Moves"][j["Moves"].index(i)+1]["Action"], "Phase":j["Moves"][j["Moves"].index(i)]["Phase"],
                                      "Last_Player_Slansky":j["Moves"][j["Moves"].index(i)-1]["Slansky"],"Current_Player_Slansky":j["Moves"][j["Moves"].index(i)]["Slansky"],
                                      "Player_Check/Calls":i["Check/Calls_%"],"Player_Bets/Raises":i["Bets/Raises_%"],"Player_Fold":i["Fold_%"],
                                      "Player_Wins":i["Win"], "Aggression":i["Aggression"]
                                      })

# Experimenting values for most common move
countAll = countFold = 0
for i in GameVariables:
    countAll +=1
    if i["Output"] == 1:
        countFold+=1

print("Most Common Move", float(countFold/countAll)*100)

# Experimenting values with random move

countRand = 0
for i in GameVariables:
    randomVal = random.randint(1,4)
    if randomVal == 1:
        countRand+=1

print("Random Move", float(countRand/countAll)*100)
#
#
# print(PlayersHandCount)
# Current Player Slansky(Loose =1, Tight =0) and Aggression(Aggressive = 1, Passive =0), Last Player Slansky and Aggression
# Slansky in phase
# Aggressive in PlayersHandCount
# Phase (Preflop = 1, Flop = 2, Turn =3, River =4)
# (N Move)
# Output Action
#
# Insert into Player
# m_ratio, player_name, hands_played, avrg_profit, avrg_stake, avrg_ROI, total_stake, player_since

### Random Forest ###
df = pd.DataFrame(GameVariables)
print("Game Variabbles Data Frame Shape: ", df.shape)
print(df)

# Labels are the values we want to predict
labels = np.array(df['Output'])

# Remove the labels from the features
# axis 1 refers to the columns
features = df.drop('Output', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

#Convert to numpy array
features = np.array(features)

print("features",feature_list)

#### Training and Testing Sets ###
# Split the data into training and testing sets
#train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

#print('Training Features Shape:', train_features.shape)
#print('Training Labels Shape:', train_labels.shape)

#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)

### Establish Baseline ### Think about it

### Train Model ###
# Instantiate model with 1000 decision trees
#rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
#rf.fit(train_features, train_labels);

### Make Predictions on the Test Set ###

# Use the forest's predict method on the test data
#predictions = rf.predict(test_features)
# Calculate the absolute errors
#errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
#print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

########################################################################################################################
fit_rf = RandomForestClassifier(max_depth=2, random_state=0 ,n_estimators=100)

training_set, test_set, class_set, test_class_set = train_test_split(features, labels, test_size = 0.25, random_state = 42)

np.random.seed(42)
start = time.time()

param_dist = {'max_depth': [2, 3, 4],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

#cv_rf = GridSearchCV(fit_rf, cv = 10,
#                     param_grid=param_dist,
#                     n_jobs = 3)



fit_rf.fit(training_set, class_set)
print (sorted(zip(map(lambda x: round(x, 4), fit_rf.feature_importances_), feature_list),
             reverse=True))
#print('Best Parameters using grid search: \n',
#      fit_rf.best_params_)

#fit_rf.set_params(criterion = 'gini',
#                  max_features = None,
#                  max_depth = 2)


y_testpredclf = fit_rf.predict(test_set)

print("Predictions Array: ",y_testpredclf)

# f = open('store', 'wb')
# pickle.dump(cv_rf, f)
# f.close()

#f = open('store', 'rb')
#obj = pickle.load(f)
#f.close()

importances = fit_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in fit_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(training_set.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(training_set.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(training_set.shape[1]), indices)
plt.xlim([-1, training_set.shape[1]])
plt.show()

print(y_testpredclf)

#print("accuracy: {}".format(fit_rf.score(test_set, test_class_set)))

# Accuracy
print ("Accuracy is ", accuracy_score(test_class_set, y_testpredclf)*100)














