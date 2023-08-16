# Breadth First Search Project Description

In this project, I build a basic graph structure and extend this foundation to analyze connections between actors based on shared movie roles. The Breadth-First Search algorithm (BFS), a centerpiece of this analysis, facilitates our exploration into how closely connected certain actors are and casts light on the "small-world phenomenon" that is characteristic of the Hollywood industry.

**Key Points about BFS**:
- BFS provides a shortest path in terms of the number of edges between the starting and the target node in an unweighted graph.
- It uses a queue data structure to keep track of nodes yet to be explored.
- BFS is particularly useful in applications where the "shortest path" or "minimum number of steps" is of interest, such as in network routing algorithms or social network analysis.

## Mathematical Background and Overview of Functions

The project revolves around the fundamental concept of graphs. We represent these graphs using adjacency dictionaries, which efficiently store connections. Furthermore, using a combination of basic graph algorithms and the NetworkX library, we can execute advanced operations and visualize these connections.

1. **Graph Initialization**
    - `add_node(n)`: Incorporates a node `n` into the graph structure.
    - `add_edge(u, v)`: Establishes a connection or edge between nodes `u` and `v`. New nodes are created if they don't previously exist.
    - `remove_node(n)`: Eradicates node `n` and any associated edges.
    - `remove_edge(u, v)`: Disconnects the edge between nodes `u` and `v`.
    - `traverse(source)`: Implements a Breadth-First Search (BFS) from the designated source node and outputs an ordered list of visited nodes.
    - `shortest_path(source, target)`: Identifies the shortest path, using BFS, between the source and target nodes.

2. **MovieGraph Analysis**
    - `__init__(filename="movie_data.txt")`: Initializes the graph based on actor and movie data procured from the specified file.
    - `path_to_actor(source, target)`: Determines the shortest path between two actors, thus revealing the least number of movie connections that link them.
    - `average_number(target)`: Computes and visualizes the average number of movie connections for each actor in relation to the specified target.

## Project Flow

1. **Graph Initialization**:
    - Commence with the creation of a graph via the `Graph` class.
    - Utilize class methods to introduce or eradicate nodes and edges.
    - Traverse the graph structure using BFS to derive visitation sequence or ascertain the shortest route between two distinct nodes.

2. **MovieGraph Analysis**:
    - Boot up the `MovieGraph` class.
    - Extract actor and movie data from the given file. Subsequently, nodes for movies and actors are created, and links between actors and their movie roles are established.
    - Discover the minimal movie connections that bind two actors.
    - Calculate and illustrate the average connections for an actor compared to all other actors in the dataset.

## Applications

1. **General Graph Applications**:
    - Comprehending intricate network connections such as those in computer networks or social circles.
    - Tackling routing and pathfinding challenges in diverse sectors.

2. **MovieGraph Utility**:
    - Understanding intricate connections among actors, beneficial for casting decisions or in-depth movie analysis.
    - Serving as a foundation for recommendation systems, suggesting films based on actor overlaps.
    - Offering insights into the intricate network of connections, revealing the closeness of the Hollywood industry.

## Dependencies

- from collections import deque
- import networkx as nx
- from matplotlib import pyplot as plt

## How to Use

1. Ensure all dependencies are installed.
2. Ensure the movie data file resides in the right directory or specify the correct path when initializing the `MovieGraph` class.
3. Import the relevant classes and use the methods to analyze and visualize the data.

## Conclusion

Incorporating BFS, this project showcases the utility of basic graph operations by applying them to a real-world dataset. Beyond its academic intrigue, the analysis holds substantial implications in the realms of movie analytics and recommendation systems, emphasizing the role of BFS in identifying shortest paths and connections within complex networks.