import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue # This is a built-in priority queue which stores tuples (priority, item). It is recommended to use this to get the minimum cost node.
import math
import time

def dijkstra(graph: np.ndarray, start: list, end: list):
    """
    Dijkstra's algorithm for finding the shortest path between two nodes in a graph.
    :param graph: Occupancy Grid
    :param start: Start node.
    :param end: End node.
    :return: List of coords of the shortest path and the total cost.
    """

    start, end = tuple(start), tuple(end)
    frontier  = PriorityQueue()
    frontier.put((0, start))
    cost_so_far = {start: 0}
    came_from = {start: None}
    dir = [(1, 0), (0, 1),(-1, 0),(0, -1), (1, 1), (1, -1),(-1, 1), (-1, -1)]
    costs = [1, 1, 1, 1, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]
    
    while not frontier.empty():
        _, current = frontier.get()
        if current == end:
            break

        for i, d in enumerate(dir):
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < graph.shape[0] and 0 <= neighbor[1] < graph.shape[1] and graph[neighbor] == 0:
                new_cost = cost_so_far[current] + costs[i]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost
                    frontier.put((priority, neighbor))
                    came_from[neighbor] = current
    current = end
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    cost=cost_so_far.get(end,-1)
    path.reverse()
    
    return path,cost

def a_star(graph: np.ndarray, start: list, end: list):
    """
    A* algorithm for finding the shortest path between two nodes in a graph.
    :param graph: Occupancy Grid
    :param start: Start node.
    :param end: End node.
    :return: List of coords of the shortest path and the total cost.
    """

    start, end = tuple(start), tuple(end)
    frontier= PriorityQueue()
    frontier.put((0 + heuristic(start, end), start))
    cost_so_far = {start: 0}
    came_from = {start: None}
    dir = [(1, 0), (0, 1),(-1, 0),(0, -1), (1, 1), (1, -1),(-1, 1), (-1, -1)]
    costs = [1, 1, 1, 1, math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]
    
    while not frontier.empty():
        _, current = frontier.get()
        if current == end:
            break
        
        for i, d in enumerate(dir):
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < graph.shape[0] and 0 <= neighbor[1] < graph.shape[1] and graph[neighbor] == 0:
                new_cost = cost_so_far[current] + costs[i]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, end)
                    frontier.put((priority, neighbor))
                    came_from[neighbor] = current

    current = end
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    cost=cost_so_far.get(end, -1)
    path.reverse()

    return path,cost

def heuristic(node, end):

    return abs(node[0] - end[0]) + abs(node[1] - end[1])

if __name__ == '__main__':
    # Test your code here
    graph=np.load('/home/cse4568/catkin_ws/src/graph_planner/scripts/husky_playpen.npy')
    #print(data)
    #print(data.shape)
    #print(data.dtype)
    start=[32,14]
    end=[14,11]
    start_time = time.time()
    dijk_path, dijk_cost=dijkstra(graph,start,end)
    dijk_time = time.time() - start_time
    
    #print(dijk_time)
    #print(dijk_cost)

    start_time = time.time()
    as_path, as_cost =a_star(graph,start,end)
    as_time = time.time() - start_time

    #print(as_time)
    #print(as_cost)
