# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Name>
<Class>
<Date>
"""
import numpy as np
import pandas as pd
import networkx as nx



# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        # Make the labels if there are none
        if labels is None:
            labels = range(A.shape[0])

        # Check if A is square and the dimension of labels equals n
        if (A.shape[0] != A.shape[1]) or (A.shape[0] != len(labels)):
            raise ValueError("Label - Matrix Dimension Mismatch")

        # Get the sum of each column of A and then add 1 to each element, then normalize it
        sum = np.sum(A, axis=0)
        B = (A + np.where(sum == 0, 1, 0))
        self.Ahat = B / np.sum(B, axis=0)

        # Store the labels and length
        self.labels = labels
        self.n = len(labels)


    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # Make the 1 vector and the M matrix
        v = (1 - epsilon)/self.n * np.ones(self.n)
        M = np.eye(self.n) - epsilon * self.Ahat

        # Backsolve the system and return it with the dictionary
        p = np.linalg.solve(M, v)
        return {self.labels[i]: p[i] for i in range(self.n)}


    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # Calculate the Matrix B
        B = epsilon * self.Ahat + (1 - epsilon)/self.n * np.ones((self.n, self.n))

        # Calculate the eigenvalues and eigenvectors and get corresponding eigenvector
        w, v = np.linalg.eig(B)
        p = v[:, np.where(np.isclose(w, 1))[0][0]]

        # Normalize the eigenvector and return it with the dictionary
        p = p / np.sum(p)
        return {self.labels[i]: p[i] for i in range(self.n)}


    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # Make our initial vector of equal probabilities and loop through the iterations
        p0 = np.ones(self.n) / self.n
        for i in range(maxiter):

            # Calculate the next vector and check for convergence
            p1 = epsilon * self.Ahat @ p0 + (1 - epsilon)/self.n
            if np.linalg.norm(p1 - p0,1) < tol:
                p0 = p1
                break

            # Update the vector
            p0 = p1

        # Return the dictionary
        return {self.labels[i]: p0[i] for i in range(self.n)}


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    # Get the keys and values
    l = [(-val,key) for key, val in zip(d.keys(),d.values())]
    l.sort()

    # Return the sorted keys
    return [key for val, key in l]


# Problem 4
def rank_websites(filename="input_files/web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    # read in the file name while splitting the lines and then splitting at each /
    with open(filename, 'r') as f:
        data = f.read().strip()
    sites =  sorted(set(data.replace('\n', '/').split('/')))
    index = {site: i for i, site in enumerate(sites)}
    
    # make the matrix
    n = len(sites)
    A = np.zeros((n, n))

    # fill in the matrix
    for line in data.split('\n'):
        for j in line.split('/')[1:]:
            A[index[j], index[line.split('/')[0]]] = 1

    # make the graph and get the ranks
    graph = DiGraph(A, sites)
    ranks = graph.itersolve(epsilon)
    return get_ranks(ranks)


# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    # Read in the csv file and get the winners, losers, and teams
    df = pd.read_csv(filename)
    winners = df['Winner'].tolist()
    losers = df['Loser'].tolist()
    teams = sorted(set(winners + losers))

    # Make the matrix
    n = len(teams)
    A = np.zeros((n, n))

    # Loop through and fill the matrix
    for i in range(n):
        for j in range(len(losers)):
            if teams[i] == losers[j]:
                A[teams.index(winners[j]),i] += 1

    # Make the graph and get the ranks
    graph = DiGraph(A, teams)
    ranks = graph.itersolve(epsilon)
    return get_ranks(ranks)
            

# Problem 6
def rank_actors(filename="input_files/top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    # Open the filenname and read in the data
    with open(filename, 'r', encoding="utf-8") as f:
        data = f.read().strip()
        lines = data.split('\n')
 
    # Get the actors and make the graph
    DG = nx.DiGraph()
    for line in lines:
        movie = line.split('/')[1:]

        # Loop through the actors and add them to the graph
        for i in range(1,len(movie)):
            actorlist = movie[0:i][::-1]
            smallactor = actorlist[0]
            
            # Add the actors to the graph if they are not already there
            for actor in actorlist[1:]:
                if not DG.has_edge(smallactor, actor):
                    DG.add_edge(smallactor, actor, weight=1)

                # If the edge is already there, add 1 to the weight
                else:
                    DG[smallactor][actor]['weight'] += 1

    # Get the page rank
    rank = nx.pagerank(DG, alpha=epsilon)
    return get_ranks(rank)