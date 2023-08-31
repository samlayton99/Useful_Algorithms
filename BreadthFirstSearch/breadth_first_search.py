# breadth_first_search.py
"""Volume 2: Breadth-First Search.
<Sam Layton>
<Vol 2 Section 003>
<10/31/22>
"""

from collections import deque
import networkx as nx
from matplotlib import pyplot as plt


# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        # update the adjacency dictionary with an empty key-value pair
        self.d.update({n:set()})

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        # Check if u is in the graph, if not add it, then add v to u's set
        if u in self.d:
            self.d[u].add(v)
        else:
            self.d.update({u:set([v])})

        # Check if v is in the graph, if not add it, then add u to v's set
        if v in self.d:
            self.d[v].add(u)
        else:
            self.d.update({v:set([u])})
    
    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        # Raise a value error if n is not in the graph
        if n not in self.d:
            raise KeyError("Node not in graph")
        else:
            # Remove n from the graph 
            self.d.pop(n)
            
            # Remove n from all other nodes' sets
            for key in self.d.keys():

                # Check if n is in the set, if so remove it
                if n in self.d[key]:
                    self.d[key].remove(n)

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        # Raise a Key error if u or v are not in the graph
        if u not in self.d or v not in self.d:
            raise KeyError("Node not in graph")
        
        # Raise a Key error if there is no edge between u and v
        elif v not in self.d[u] or u not in self.d[v]:
            raise KeyError("No edge between u and v")

        # Remove the edge between u and v
        else:
            self.d[u].remove(v)
            self.d[v].remove(u)


    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        # Raise a Key error if source is not in the graph
        if source not in self.d:
            raise KeyError("Node not in graph")

        # Initialize visited list, queue, and marked dictionary
        queue = deque([source])
        visited = []
        marked = {source}

        # While the queue is not empty
        while len(queue) != 0:
            # Pop the first element in the queue and append it to visited
            node = queue.pop()
            visited.append(node)

            # Access the keys of the node's neighbors and loop through them
            neighbors = self.d[node]
            for item in neighbors:

                # If the neighbor has not been marked, mark it and add it to the queue
                if item not in marked:
                    marked.update(item)
                    queue.appendleft(item)

        # Return the list of visited nodes
        return visited


    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        # Raise a Key error if source or target are not in the graph
        if source not in self.d or target not in self.d:
            raise KeyError("Node not in graph")
        
        # Initialize visited list, queue, and marked dictionary
        queue = deque([source])
        path = []
        marked = {source:None}

        # While the queue is not empty
        while len(queue) != 0:
            # Pop the first element in the queue, append it to visited
            node = queue.pop()

            # Access the keys of the node's neighbors and loop through them
            neighbors = self.d[node]
            for item in neighbors:
                
                # If the neighbor has not been marked, mark it and add it to the queue
                if item not in marked:
                    marked.update({item:node})
                    queue.appendleft(item)
                
                # If the neighbor is the target, loop throught the marked dictionary until the source is found
                if item == target:
                    while item != None:
                        # Append the node to the path list
                        path.append(item)
                        item = marked[item]

                    # Reverse the path and return it
                    return path[::-1]


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="input_files/movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        # Initialize the sets and graph
        self.graph = nx.Graph()
        self.movies = set()
        self.actors = set()

        # Open the file and read each line
        with open(filename, 'r') as file:
            lines = file.readlines()
        # Loop through each line
        for lines in lines:
            # Split the line into a list of strings by the '/' character
            splitline = lines.strip().split('/')

            # Add the movie title to the set of movies
            self.movies.add(splitline[0])

            # Add the actor to the set of actors and add an edge between the movie and the actor
            for actor in splitline[1:]:
                self.actors.add(actor)
                self.graph.add_edge(splitline[0], actor)
    
        
    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        # find the shortest path between source and target and then return the path and length
        path = nx.shortest_path(self.graph, source, target)
        return path, (len(path) - 1) / 2


    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        # Make a dictionary of the shortest paths and then convert it to a list
        values = nx.shortest_path_length(self.graph, target).values()
        values = [float(i) / 2 for i in values]

        # Plot the distribution of path lengths
        plt.hist(values, bins=[i-.5 for i in range(8)])
        plt.xlabel("Degrees of Separation")
        plt.ylabel("Frequency")

        # Give a title and show the plot
        plt.title("Distribution of Path Lengths Among Actors")
        plt.tight_layout()
        plt.savefig('figures/Problem6.png')
        plt.show()

        # Return the average path length
        return sum(values) / len(values)
        
# run test cases that plot for problem 6
if __name__ == "__main__":
    # Problem 6
    mg = MovieGraph()
    mg.average_number("Kevin Bacon")
    