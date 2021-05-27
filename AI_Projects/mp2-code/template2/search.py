# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush
import queue

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None.
    """
    current = maze.getStart()
    goal = maze.getObjectives()
    rsvp = queue.Queue()
    visited = {}
    thisisgoal = None
    path = []
    rsvp.put(current)
    visited[current] = None
    allof = False
    while rsvp.empty() == False:
        current = rsvp.get()
        if current in goal:
            break
        choices = maze.getNeighbors(current[0], current[1], current[2])
        for c in choices:
            if c in goal:
                visited[c] = current
                thisisgoal = c
                rsvp.queue.clear()
                break
            elif c not in visited:
                visited[c] = current
                rsvp.put(c)

    for g in goal:
        if g in visited:
            allof =  True

    if allof == False:
        return None

    while thisisgoal != maze.getStart():
        path.append(thisisgoal)
        thisisgoal = visited[thisisgoal]

    path.append(maze.getStart())
    print(path)
    return path[::-1]
