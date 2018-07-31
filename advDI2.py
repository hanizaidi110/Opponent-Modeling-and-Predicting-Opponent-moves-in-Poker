import os
#
# import psycopg2 as psycopg2
# from contextlib import closing
# from django.db import connection
# from django.utils import timezone
# import psycopg2
#
# #from django.db import models
# from pokerApp.models import *

#conn = psycopg2.connect("host=localhost dbname=poker user=hani password=hanihani1 ")
#cur = conn.cursor()
#def insert_game():
#    Game.objects.create(number_of_hands=10, game_type="Springsteen",duration= "long")
#cur.execute("INSERT INTO pokerApp_game VALUES (%s, %s, %s)", ('1', 'hello', 'long'))
#conn.commit()
#insert_game()

# for p in Game.objects.raw('SELECT * FROM myapp_person'):
#     print (p)

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
    Hands1["game_name"] = filename.split('-')[0]
    Hands1["game_type"] = filename.split('-')[-1].split('.txt')[0]

    notes = []
    moves = []
    phase = ""
    hand = []
    test = []
    meow = []

    for i in newData:
        hand = i.split("\n")
        #Limiting hands data
        #if lineNo <5:
        #print (hand)
        moves = []
        positions = []
        listy = {}

        showdown = "False"
        turn = "False"
        river = "False"
        flop = "False"

        if hand[0] == "" and hand[1] != "":
            for i in range(len(hand)):
                    listy["HandID"] = hand[1].split(' ')[2]
                    listy["TimeStamp"] = hand[1].split('-')[1]
                    listy["GameType"] = hand[1].split(':')[1].split('-')[0]
                    listy["TableName"] = hand[2].split("'")[1]
                    listy["MaxNoOfPlayers"] = hand[2].split(' ')[2][0]
                    listy["Button"] = hand[2].split('#')[1].split(' ')[0]
                    if hand[i].split(' ', 1)[0] == "Seat":
                        if "in chips" in hand[i]:
                            positions.append({"Seat": hand[i].split(' ')[1], "Player": hand[i].split(' ')[2], "StackSize": hand[i].split(' ')[3][1:]})
                    if "posts small blind" in hand[i]:
                            listy["SmallBlind"] = hand[i].split(':')[0]
                    if "posts big blind" in hand[i]:
                            listy["BigBlind"] = hand[i].split(':')[0]
                    if ":" in hand[i].split(' ')[0]:
                            moves.append({"Player": hand[i].split(':')[0], "Action": hand[i].split(':')[1], "Phase": "Pre-Flop"})
                    if "collected" in hand[i] and "Seat" not in hand[i]:
                            listy["Winner"] = hand[i].split(' ')[0]
                    if "SHOW DOWN" in hand[i]:
                            showdown =  "True"
                    if "TURN" in hand[i]:
                            turn =  "True"
                    if "RIVER" in hand[i]:
                            river = "True"
                    if "FLOP" in hand[i]:
                            flop = "True"

            listy["flop"] = flop
            listy["showdown"] = showdown
            listy["turn"] = turn
            listy["river"] = river
            listy["Positions"] = positions
            listy["moves"] = moves
            test.append(listy)

    Hands1["Hands"] = test
    return Hands1

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
            result = countNumberOfHands(tempFileData, listy, filename)
            totalHands = totalHands + len(result["Hands"])
            count +=1

    #print("Total Hands:", totalHands)
    #print("Total Files:", totalFiles)
    return result

result = totalCounts()
#print(result)
#p = Game(number_of_hands=len(test), game_type= Hands1["game_type"] ,  game_name=Hands1["game_name"])
#p.save()

#print(result)

for i in range(len(result["Hands"])):
    if result["Hands"][i]:
        if len(result["Hands"][i]["moves"]) > len(result["Hands"][i]["Positions"]):
                    #print("Bahi Waah")
            pass

    #print(result["Hands"][i]["showdown"])
    #print(result["Hands"][i]["flop"])
    #print(result["Hands"][i]["river"])
    #print(result["Hands"][i]["turn"])

for i in range(len(result["Hands"])):
        if result["Hands"][i]:
            #print(len(result["Hands"][i]["Positions"]))
            for j in range(len(result["Hands"][i]["Positions"])):
                pass
                #print(result["Hands"][i]["Positions"][j]["Player"])
                #print(result["Hands"][i]["Positions"][j]["Seat"])

        #print (result["Hands"][i]["TimeStamp"])
        #print(result["Hands"][i])
        #print(result["Hands"])
        #print (result["Hands"][i]["Positions"])
        #print(result["Hands"][i]["moves"])
        #if len(result["Hands"][i]["moves"]) >
        # s = Hand(game_ID = result["game_name"], number_of_orbit = 1, winner = "Hani",\
        #      showdown = "True", rake = "0", timestamp = result["Hands"][i]["TimeStamp"], number_of_players = 0,\
        #      button ="none", big_blind ="none", small_blind="none")
        # s.save()

for i in range(len(result["Hands"])):
    if result["Hands"][i]:
        for j in range(len(result["Hands"][i]["moves"])):
            new = result["Hands"][i]["moves"][j]["Action"].split('$')
            if len(new) > 2:
                #print(new[0], new[2].split(' ')[0])
                pass
            if len(new) > 1 and len(new) < 3:
                #print(new[0], new[1].split(' ')[0])
                pass
            elif len(new) < 2:
                print(new[0])

             #print(result["Hands"][i]["Positions"][j]["Player"])



