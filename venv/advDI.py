import os

import psycopg2 as psycopg2
import utils
from contextlib import closing
from django.db import connection
from django.utils import timezone
import psycopg2
from django.db import models
from AdvDI.venv.myproject.pokerApp.models import Game

#conn = psycopg2.connect("host=localhost dbname=poker user=hani password=hanihani1 ")
#cur = conn.cursor()
#def insert_game():
#    Game.objects.create(number_of_hands=10, game_type="Springsteen",duration= "long")
#cur.execute("INSERT INTO pokerApp_game VALUES (%s, %s, %s)", ('1', 'hello', 'long'))
#conn.commit()

#insert_game()

newFile = open("/home/syed/Documents/AdvDI/Hands1/Part1/HH20170728 Alamak - $0.01-$0.02 - USD No Limit Hold'em.txt",
               'r')
with newFile as myfile:
    # for line in myfile:
    data = myfile.read()


def countNumberOfHands(freshData, listy,filename):
    countOfHands = 0
    #newData = One Hand
    newData = freshData.split("\n\n\n")
    positions = []
    lineNo = 0
    line = 0
    Hands1 = {}
    Hands1["tableID"] = filename
    notes = []
    moves = []
    phase = ""
    hand = []
    test = []
    flop = False
    for i in newData:
        hand = i.split("\n")
        #Limiting hands data
        #if lineNo <5:
        if hand[0] == "" and hand[1] != "":
                    listy = {}
                    listy["HandID"] = hand[1].split(' ')[2]
                    listy["TimeStamp"] = hand[1].split('-')[1]
                    listy["GameType"] = hand[1].split(':')[1].split('-')[0]
                    listy["TableName"] = hand[2].split("'")[1]
                    listy["MaxNoOfPlayers"] = hand[2].split(' ')[2][0]
                    listy["Button"] = hand[2].split('#')[1].split(' ')[0]
                    for line in range(30):
                        if hand[line].split(' ', 1)[0] == "Seat":
                            positions.append({"Seat": hand[line].split(' ')[1], "Player": hand[line].split(' ')[2], "StackSize": hand[line].split(' ')[3][1:]})
                            listy["Positions"] = positions
                        if "posts small blind" in hand[line]:
                            listy["SmallBlind"] = hand[line].split(':')[0]
                        if "posts big blind" in hand[line]:
                            listy["BigBlind"] = hand[line].split(':')[0]
                        if ":" in hand[line].split(' ')[0]:
                            moves.append({"Player": hand[line].split(':')[0], "Action": hand[line].split(':')[1], "Phase": "Pre-Flop"})
                            listy["moves"] = moves
            #lineNo += 1
        test.append(listy)
    Hands1["Hands"] = test

    print (Hands1)
    return countOfHands

def totalCounts():
    path = '/home/syed/Documents/AdvDI/Hands1/Part1/'
    totalFiles = 0
    totalHands = 0
    count = 0
    for filename in os.listdir(path):
        listy = {}
        if count == 0 :
            totalFiles = totalFiles + 1
            newData = open(path + filename, 'r')
            tempFileData = newData.read()
            totalHands = totalHands + countNumberOfHands(tempFileData, listy,filename)
            count +=1


    print("Total Hands:", totalHands)
    print("Total Files:", totalFiles)


totalCounts()
