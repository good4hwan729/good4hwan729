# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *
import math

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.

        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    if arm.getNumArmLinks() == 3:
        offseta = arm.getArmLimit()[0][0]
        offsetb = arm.getArmLimit()[1][0]
        offsetg = arm.getArmLimit()[2][0]
        gammarange = arm.getArmLimit()[2][1] - arm.getArmLimit()[2][0]
        betarange = arm.getArmLimit()[1][1] - arm.getArmLimit()[1][0]

    elif arm.getNumArmLinks() == 2:
        offseta = arm.getArmLimit()[0][0]
        offsetb = arm.getArmLimit()[1][0]
        offsetg = 0
        betarange = arm.getArmLimit()[1][1] - arm.getArmLimit()[1][0]

    else:
        offseta = arm.getArmLimit()[0][0]
        offsetb = 0
        offsetg = 0

    alpharange = arm.getArmLimit()[0][1] - arm.getArmLimit()[0][0]

    offset = [offseta, offsetb, offsetg] #the minimum angles will be the offsets

    rows = math.floor(alpharange / granularity) + 1 #calculate the number of rows and columns
    cols = 1
    zets = 1
    if arm.getNumArmLinks() >= 2:
        cols = math.floor(betarange / granularity) + 1
    if arm.getNumArmLinks() == 3:
        zets = math.floor(gammarange / granularity) + 1

    if arm.getNumArmLinks() == 1:
        initalpha = arm.getArmAngle()[0]
        initbeta = 0
        initgamma = 0
    elif arm.getNumArmLinks() == 2:
        initalpha = arm.getArmAngle()[0]
        initbeta = arm.getArmAngle()[1]
        initgamma = 0
    else:
        initalpha = arm.getArmAngle()[0]
        initbeta = arm.getArmAngle()[1]
        initgamma = arm.getArmAngle()[2]

    input_map =  [[[' ' for x in range(zets)] for y in range(cols)] for z in range(rows)]

    for row in range(rows): #alpha (y position)
        for col in range(cols): #beta (x position)
            for zet in range(zets): #gamma (z position)

                alpha = idxToAngle((row,col,zet), offset, granularity)[0] #convert indexes to angles
                beta = idxToAngle((row,col, zet), offset, granularity)[1]
                gamma = idxToAngle((row,col, zet), offset, granularity)[2]

                arm.setArmAngle((alpha, beta, gamma))

                if not isArmWithinWindow(arm.getArmPos(), window): #check if arm in current angles touches window
                    input_map[row][col][zet] = '%'
                    continue
                if doesArmTouchObjects(arm.getArmPosDist(), obstacles, False): #check if arm touches obstacles
                    input_map[row][col][zet] = '%'
                    continue
                if doesArmTouchObjects(arm.getArmPosDist(), goals, True) and not doesArmTipTouchGoals(arm.getEnd(), goals): #check if arm touches goals
                    input_map[row][col][zet] = '%'
                    continue
                if doesArmTipTouchGoals(arm.getEnd(), goals) or doesArmTouchObjects(arm.getArmPosDist(), goals, True): #we filter out when armtip touches the goals.
                    input_map[row][col][zet] = '.'


    starty, startx, startz = angleToIdx((initalpha, initbeta, initgamma), offset, granularity)
    input_map[starty][startx][startz] = 'P'

    #print(input_map)
    return Maze(input_map, offset, granularity) #return maze
