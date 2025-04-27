# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

import time
import util

def depthFirstSearch(problem):
    """Search the deepest nodes in the search tree first."""
    start_time = time.perf_counter()

    stack = util.Stack()
    start_state = problem.getStartState()
    stack.push((start_state, []))
    visited = set()

    while not stack.isEmpty():
        state, actions = stack.pop()

        if problem.isGoalState(state):
            end_time = time.perf_counter()
            print(f"DFS completed in {end_time - start_time:.5f} seconds.")
            return actions

        if state not in visited:
            visited.add(state)

            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:
                    stack.push((successor, actions + [action]))

    end_time = time.perf_counter()
    print(f"DFS completed in {end_time - start_time:.5f} seconds.")
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    start_time = time.perf_counter()

    queue = util.Queue()
    start_state = problem.getStartState()
    queue.push((start_state, []))
    visited = set()

    while not queue.isEmpty():
        state, actions = queue.pop()

        if problem.isGoalState(state):
            end_time = time.perf_counter()
            print(f"BFS completed in {end_time - start_time:.5f} seconds.")
            return actions

        if state not in visited:
            visited.add(state)

            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited:
                    queue.push((successor, actions + [action]))

    end_time = time.perf_counter()
    print(f"BFS completed in {end_time - start_time:.5f} seconds.")
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    start_time = time.perf_counter()

    priority_queue = util.PriorityQueue()
    start_state = problem.getStartState()
    priority_queue.push((start_state, [], 0), 0)
    visited = dict()  # map of state -> cost

    while not priority_queue.isEmpty():
        state, actions, cost_so_far = priority_queue.pop()

        if problem.isGoalState(state):
            end_time = time.perf_counter()
            print(f"UCS completed in {end_time - start_time:.5f} seconds.")
            return actions

        if state not in visited or cost_so_far < visited[state]:
            visited[state] = cost_so_far

            for successor, action, step_cost in problem.getSuccessors(state):
                new_cost = cost_so_far + step_cost
                if successor not in visited or new_cost < visited.get(successor, float('inf')):
                    priority_queue.push((successor, actions + [action], new_cost), new_cost)

    end_time = time.perf_counter()
    print(f"UCS completed in {end_time - start_time:.5f} seconds.")
    return []

def nullHeuristic(state, problem=None):
    """A heuristic function estimates the cost from the current state to the nearest goal."""
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start_time = time.perf_counter()

    priority_queue = util.PriorityQueue()
    start_state = problem.getStartState()
    priority_queue.push((start_state, [], 0), heuristic(start_state, problem))
    visited = dict()  # map of state -> cost

    while not priority_queue.isEmpty():
        state, actions, cost_so_far = priority_queue.pop()

        if problem.isGoalState(state):
            end_time = time.perf_counter()
            print(f"A* Search completed in {end_time - start_time:.5f} seconds.")
            return actions

        if state not in visited or cost_so_far < visited[state]:
            visited[state] = cost_so_far

            for successor, action, step_cost in problem.getSuccessors(state):
                new_cost = cost_so_far + step_cost
                total_cost = new_cost + heuristic(successor, problem)
                if successor not in visited or new_cost < visited.get(successor, float('inf')):
                    priority_queue.push((successor, actions + [action], new_cost), total_cost)

    end_time = time.perf_counter()
    print(f"A* Search completed in {end_time - start_time:.5f} seconds.")
    return []



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
