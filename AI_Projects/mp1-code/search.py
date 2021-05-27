# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import queue
import heapq

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    current = maze.getStart()
    goal = maze.getObjectives()[0]
    rsvp = queue.Queue()
    visited = {}

    path = []
    rsvp.put(current)
    visited[current] = None

    while rsvp.empty() == False:
        current = rsvp.get()
        if current == goal:
            break
        choices = maze.getNeighbors(current[0], current[1])
        for c in choices:
            if c == goal:
                visited[c] = current
                rsvp.queue.clear()
                break
            elif c not in visited:
                visited[c] = current
                rsvp.put(c)

    while goal != maze.getStart():
        path.append(goal)
        goal = visited[goal]

    path.append(maze.getStart())
    return path[::-1]


def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    current = maze.getStart()
    goal = maze.getObjectives()[0]
    #print(astar_len(maze, current, goal))
    rsvp = []
    visited = {}
    curdis = 0
    manhat = abs(goal[0] - current[0]) + abs(goal[1] - current[1])
    totdis = curdis + manhat

    path = []
    heapq.heappush(rsvp, (totdis, curdis, manhat, current))
    visited[current] = None

    while rsvp:
        cur = heapq.heappop(rsvp)
        current = cur[3]
        curdis = cur[1]+1

        choices = maze.getNeighbors(current[0], current[1])
        for c in choices:
            manhat = abs(goal[0] - c[0]) + abs(goal[1] - c[1])
            totdis = curdis + manhat
            if c == goal:
                visited[c] = current
                rsvp.clear()
                break
            elif c not in visited:
                visited[c] = current
                heapq.heappush(rsvp, (totdis, curdis, manhat, c))

    while goal != maze.getStart():
        path.append(goal)
        goal = visited[goal]

    path.append(maze.getStart())
    return path[::-1]

def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    #print(tuple(sorted([(6,1)] + [(1,6)])))
    # TODO: Write your code here
    current = maze.getStart()
    goal = maze.getObjectives()
    rsvp = [] #the queue that we will push the nodes
    visited = {} #the dictionary to backtrace the path

    curdis = 0 #g(x)
    test = []
    curr_goal = []
    for g in goal:
        test.append(abs(g[0] - current[0]) + abs(g[1] - current[1]))
    manhat = min(test) #partial h(x)
    totdis = curdis + manhat + mst_corner(maze, goal) #Full f(x)

    gp = [] #goals traversed so far
    path = [] #path to print out final result
    value_dic = {} #dictionary to store values

    value_dic[(current, ())] = totdis #add start
    visited[(current,())] = (None, ()) #add start
    heapq.heappush(rsvp, (totdis, curdis, manhat, gp, current)) #push start

    while rsvp: #while the queue is not empty
        cur = heapq.heappop(rsvp) #pop out the node

        current = cur[4] #coordinate of the current node
        gp = cur[3] # goals that this node passed
        curdis = cur[1]+1 #g(x) + 1 since this is a new step.

        #initialize a list that will display goals remaining
        curr_goal.clear()
        for g in goal:
            if g not in gp:
                curr_goal.append(g)

        #if this node already found all 4 goals, we exit
        if len(cur[3]) == len(goal):
            break

        choices = maze.getNeighbors(current[0], current[1]) #gather neighbors
        for c in choices:

            for g in curr_goal: #calculate the manhattan dist to closest goal
                test.append(abs(g[0] - c[0]) + abs(g[1] - c[1]))
            if len(test) == 0:
                break
            manhat = min(test)
            test.clear() #reset

            totdis = curdis + manhat + mst_corner(maze, curr_goal) #f(x) for this neighbor

            if c in curr_goal and (c, tuple(sorted(gp + [c]))) not in visited: #if the node hasn't been visited and is a goal
                visited[(c, tuple(sorted(gp + [c])))] = (current, tuple(gp))
                value_dic[(c, tuple(sorted(gp+[c])))] = totdis
                heapq.heappush(rsvp, (totdis, curdis, manhat, sorted(gp + [c]), c))

            elif c not in curr_goal and (c,tuple(gp)) not in visited: #if the node hasn't been visited and isn't a goal
                visited[(c, tuple(gp))] = (current, tuple(gp))
                value_dic[(c, tuple(gp))] = totdis
                heapq.heappush(rsvp, (totdis, curdis, manhat, gp, c))

    #print(visited[(current,tuple(gp))])
    while current != maze.getStart() or len(gp) != 0: #BACKTRACKING PROCESS
        path.append(current)
        current, gp = visited[(current,tuple(gp))][0], visited[(current,tuple(gp))][1]


    path.append(maze.getStart())
    #print(path[::-1])
    return path[::-1]




def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    current = maze.getStart()
    goal = maze.getObjectives()

    rsvp = [] #the queue that we will push the nodes
    visited = {} #the dictionary to backtrace the path

    curdis = 0 #g(x)
    test = []
    curr_goal = []

    mst_dic = {}
    mst_dic[tuple(goal)] = mst_man(maze,goal)

    for g in goal:
        test.append(abs(g[0] - current[0]) + abs(g[1] - current[1]))
    manhat = min(test) #partial h(x)
    heuristic = manhat + mst_dic[tuple(goal)]
    totdis = curdis + heuristic #Full f(x)

    gp = [] #goals traversed so far
    path = [] #path to print out final result

    visited[(current,())] = (None, (), totdis) #add start
    heapq.heappush(rsvp, (totdis, curdis, heuristic, gp, current)) #push start

    while rsvp: #while the queue is not empty
        cur = heapq.heappop(rsvp) #pop out the node

        current = cur[4] #coordinate of the current node
        gp = cur[3] # goals that this node passed
        curdis = cur[1]+1 #g(x) + 1 since this is a new step.

        #initialize a list that will display goals remaining
        curr_goal.clear()
        for g in goal:
            if g not in gp:
                curr_goal.append(g)


        #if this node already found all 4 goals, we exit
        if len(cur[3]) == len(goal):
            break

        choices = maze.getNeighbors(current[0], current[1]) #gather neighbors
        for c in choices:

            for g in curr_goal: #calculate the manhattan dist to closest goal
                test.append(abs(g[0] - c[0]) + abs(g[1] - c[1]))
            if len(test) == 0:
                break
            manhat = min(test)
            test.clear() #reset

            if tuple(curr_goal) not in mst_dic.keys():
                mst_dic[tuple(curr_goal)] = mst_man(maze, curr_goal)

            heuristic = manhat + mst_dic[tuple(curr_goal)]
            totdis = curdis + heuristic #f(x) for this neighbor

            if c in curr_goal and (c, tuple(sorted(gp + [c]))) not in visited: #if the node hasn't been visited and is a goal
                visited[(c, tuple(sorted(gp + [c])))] = (current, tuple(gp), totdis)
                heapq.heappush(rsvp, (totdis, curdis, heuristic, sorted(gp + [c]), c))

            elif c not in curr_goal and (c,tuple(gp)) not in visited: #if the node hasn't been visited and isn't a goal
                visited[(c, tuple(gp))] = (current, tuple(gp), totdis)
                heapq.heappush(rsvp, (totdis, curdis, heuristic, gp, c))

    while current != maze.getStart() or len(gp) != 0: #BACKTRACKING PROCESS
        path.append(current)
        current, gp = visited[(current,tuple(gp))][0], visited[(current,tuple(gp))][1]

    path.append(maze.getStart())
    #print(path[::-1])
    return path[::-1]




def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    current = maze.getStart()
    goal = maze.getObjectives()

    test = {}
    curr_goal = []
    path = []
    mindis = 0
    path.append(current)
    for g in goal:
        curr_goal.append(g)

    while curr_goal:
        for g in range(len(curr_goal)):
            test[abs(current[0] - curr_goal[g][0]) + abs(current[1] - curr_goal[g][1])] = curr_goal[g]

        mindis = min(test.keys())
        mincoord = test[mindis]
        test.clear()

        for i in astar_single(maze, current, mincoord):
            if i in curr_goal:
                curr_goal.remove(i)
            path.append(i)
        current = mincoord

    return path

def mst(maze, goal):

    if goal == None: #minimal check
        return 0

    edges = {} #stores all edges of the graph
    mst_graph = [] #where we insert visited nodes
    mst_val = 0 #sum of the weights to be returned

    #start = (0, goal[0]) #choose an arbitrary node and set it's weight to 0

    rsvp = [] #priority queue

    heapq.heappush(rsvp, (0, goal[0]))

    for i in goal:
        for j in goal:
            if i != j:
                edges[(i,j)] = astar_len(maze, i, j) #create a dictionary for all edge values


    while len(mst_graph) < len(goal): #while visited nodes are not the total nodes
        lowest = heapq.heappop(rsvp)
        if lowest[1] in mst_graph: #check if there's no duplicates
            continue
        mst_val += lowest[0]
        mst_graph.append(lowest[1])
        for edge in edges.keys():
            if edge[0] == lowest[1] and edge[1] not in mst_graph:
                heapq.heappush(rsvp, (edges[edge], edge[1])) #add all neighbors

    return mst_val

def mst_man(maze, goal):

    if goal == None: #minimal check
        return 0

    edges = {} #stores all edges of the graph
    mst_graph = [] #where we insert visited nodes
    mst_val = 0 #sum of the weights to be returned

    #start = (0, goal[0]) #choose an arbitrary node and set it's weight to 0

    rsvp = [] #priority queue

    heapq.heappush(rsvp, (0, goal[0]))

    for i in goal:
        for j in goal:
            if i != j:
                edges[(i,j)] = abs(i[0] - j[0]) + abs(i[1] - j[1]) #create a dictionary for all edge values


    while len(mst_graph) < len(goal): #while visited nodes are not the total nodes
        lowest = heapq.heappop(rsvp)
        if lowest[1] in mst_graph: #check if there's no duplicates
            continue
        mst_val += lowest[0]
        mst_graph.append(lowest[1])
        for edge in edges.keys():
            if edge[0] == lowest[1] and edge[1] not in mst_graph:
                heapq.heappush(rsvp, (edges[edge], edge[1])) #add all neighbors

    return mst_val

def astar_len(maze, start, end):

    rsvp = []
    visited = {}
    current = start
    curdis = 0
    manhat = abs(end[0] - current[0]) + abs(end[1] - current[1])
    totdis = curdis + manhat

    path = []
    heapq.heappush(rsvp, (totdis, curdis, manhat, current))
    visited[start] = None

    while rsvp:
        cur = heapq.heappop(rsvp)
        current = cur[3]
        curdis = cur[1]+1

        choices = maze.getNeighbors(current[0], current[1])
        for c in choices:
            manhat = abs(end[0] - c[0]) + abs(end[1] - c[1])
            totdis = curdis + manhat
            if c == end:
                visited[c] = current
                rsvp.clear()
                break
            elif c not in visited:
                visited[c] = current
                heapq.heappush(rsvp, (totdis, curdis, manhat, c))

    while end != start:
        path.append(end)
        end = visited[end]

    path.append(start)

    return len(path) - 1

def astar_single(maze, start, end):

    rsvp = []
    visited = {}
    current = start
    curdis = 0
    manhat = abs(end[0] - current[0]) + abs(end[1] - current[1])
    totdis = curdis + manhat

    path = []
    heapq.heappush(rsvp, (totdis, curdis, manhat, current))
    visited[start] = None

    while rsvp:
        cur = heapq.heappop(rsvp)
        current = cur[3]
        curdis = cur[1]+1

        choices = maze.getNeighbors(current[0], current[1])
        for c in choices:
            manhat = abs(end[0] - c[0]) + abs(end[1] - c[1])
            totdis = curdis + manhat
            if c == end:
                visited[c] = current
                rsvp.clear()
                break
            elif c not in visited:
                visited[c] = current
                heapq.heappush(rsvp, (totdis, curdis, manhat, c))

    while end != start:
        path.append(end)
        end = visited[end]

    #path.append(start)

    return path[::-1]

def mst_corner(maze, goals):
    if len(goals) == 1:
        return 0
    elif len(goals) == 2:
        return (abs(goals[0][0] - goals[1][0]) + abs(goals[0][1] - goals[1][1]))
    elif len(goals) == 3:
        return maze.getDimensions()[0] + maze.getDimensions()[1] - 6
    else:
        if maze.getDimensions()[0] < maze.getDimensions()[1]:
            return 2 * (maze.getDimensions()[0] -3) + maze.getDimensions()[1] - 3
        else:
            return 2 * (maze.getDimensions()[1] - 3) + maze.getDimensions()[0] - 3
