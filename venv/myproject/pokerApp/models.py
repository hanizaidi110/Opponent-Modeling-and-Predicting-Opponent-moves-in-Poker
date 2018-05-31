# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.db import models

# Create your models here.

#CREATE TABLE game(id int, noOfHands int, gameType varchar(255), duration varchar(255))

class Game(models.Model):
    number_of_hands = models.IntegerField(default = 0)
    game_name = models.CharField(max_length=200)
    game_type = models.CharField(max_length=200)

#CREATE TABLE hand(id int, gameId int, NoOforbit int, winner varchar(255), showdown varchar(255),
                # rake varchar(255), timeStamp varchar(255), noOfPlayers int, button varchar(255), bigblind varchar(255),
                # smallblind) varchar(255))

class Hand(models.Model):
    game_ID = models.CharField(max_length=200)
        #models.ForeignKey(Game)
    number_of_orbit = models.IntegerField(default= 0)
    winner = models.CharField(max_length= 200)
    showdown = models.CharField(max_length= 200)

    flop = models.CharField(max_length=200, default='SOME STRING')
    river = models.CharField(max_length=200, default='SOME STRING')
    turn = models.CharField(max_length=200, default='SOME STRING')
    rake = models.CharField(max_length=200, default='SOME STRING')

    timeStamp = models.CharField(max_length=200)
    number_of_players = models.IntegerField(default=0)
    button = models.CharField(max_length=200)
    big_blind = models.CharField(max_length=200)
    small_blind =models.CharField(max_length=200)

#CREATE TABLE player(id int, name varchar(255), MRatio int, HandsPlayed varchar(255), AvrgProfit varchar(255),
                # AvgStake varchar(255), AvrgROI varchar(255), TotalStake varchar(255), PlayerSince varchar(255))

class Player(models.Model):
    m_ratio = models.IntegerField(default= 0)
    player_name = models.CharField(max_length= 200)
    hands_played = models.IntegerField(default= 0)
    avrg_profit = models.IntegerField(default= 0)
    avrg_stake = models.IntegerField(default= 0)
    avrg_ROI = models.IntegerField(default= 0)
    total_stake = models.IntegerField(default= 0)
    player_since = models.CharField(max_length= 200)

#CREATE TABLE move(handID int, playerID int, action varchar(255), amount varchar(255), phase varchar(255),
                    # position varchar(255), indicators varchar(255))

class Move(models.Model):
    hand_ID = models.CharField(max_length=200)
        #models.ForeignKey(Hand)
    player_ID = models.CharField(max_length=200)
        #models.ForeignKey(Player)
    action = models.CharField(max_length= 200)
    amount = models.CharField(max_length= 200)
    phase = models.CharField(max_length= 200)
    postion = models.CharField(max_length= 200)
    indicators = models.CharField(max_length= 200, default = 'Some string')

#CREATE TABLE playersInGame(playerID, HandID, StackSize varchar(255), Seat varchar(255))

class PlayersInGame(models.Model):
    player_ID = models.CharField(max_length=200)
        #models.ForeignKey(Player)
    hand_ID = models.CharField(max_length=200)
        #models.ForeignKey(Hand)
    stack_size = models.CharField(max_length= 200)
    seat = models.CharField(max_length= 50)

