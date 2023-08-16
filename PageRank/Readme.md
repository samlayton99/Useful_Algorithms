# Page Rank Project Description

Analyzing directed graphs and ranking nodes is fundamental to understanding the structure and importance of various interconnected systems. This project leverages the power of the PageRank algorithm, famously utilized by Google to rank web pages, to study the significance of nodes in diverse datasets, ranging from websites to actors. By integrating mathematical techniques like linear algebra, eigenvalues, and iterative methods, this project offers insights into the prominence of individual nodes in these systems.

## Mathematical Background and Overview of Concepts

PageRank algorithm's core is based on linear algebra, particularly in the domain of eigenvalues and eigenvectors. The recursive nature of this algorithm aims to rank nodes in a graph based on their connectivity, making it a robust method to study large systems.

1. **Adjacency Matrices**
   - Representations of finite graphs in the form of square matrices. The existence of an edge between vertices `j` and `i` is shown by the element at `(i,j)`.

2. **PageRank Algorithm**
   - A recursive method to determine the significance of nodes in a graph by examining their links.

3. **Eigenvectors and Eigenvalues**
   - Essential components of linear algebra which can be employed to calculate the PageRank of nodes.

4. **Linear Systems**
   - Techniques for solving systems of linear equations to deduce node significance.

## Classes and Functions

1. **DiGraph**
   - Represents directed graphs via adjacency matrices.
     - `__init__(self, A, labels=None)`: Initialize the graph and manage sinks.
     - `linsolve(self, epsilon=0.85)`: Calculate PageRank with the linear system approach.
     - `eigensolve(self, epsilon=0.85)`: Determine PageRank utilizing the eigenvalue method.
     - `itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12)`: Assess PageRank through the iterative process.

2. **Utilities**
   - Auxiliary functions for diverse applications.
     - `get_ranks(d)`: Extract a list of labels based on the computed PageRank vector.
     - `rank_websites(filename="web_stanford.txt", epsilon=0.85)`: Order websites considering hyperlinks.
     - `rank_ncaa_teams(filename, epsilon=0.85)`: Position NCAA teams using game outcomes.
     - `rank_actors(filename="top250movies.txt", epsilon=0.85)`: Evaluate actors based on their film collaborations with NetworkX.

## Project Flow

1. **Graph Initialization**: Define the graph utilizing adjacency matrices and optionally, labels. Remove any sinks (nodes lacking outgoing edges) to guarantee ergodicity.
2. **Compute PageRank**: Utilize three distinct methodologies:
   - Linear Systems: Frame the PageRank computation as a series of linear equations and resolve.
   - Eigenvalues: Identify the main eigenvector of the transition probability matrix.
   - Iterations: Continually revise the PageRank until a convergence point is reached.
3. **Ranking Nodes**: Categorize nodes (such as websites, sports teams, or actors) based on their determined PageRank values.
4. **Applications**:
   - **Web Page Ranking**: Prioritize web pages using their hyperlink data.
   - **NCAA Team Ranking**: Rank teams using their on-field performance.
   - **Actor Ranking**: Sort actors according to their film partnerships.

## Dependencies

- numpy
- pandas
- networkx

**Note**: Confirm the presence of data files such as `web_stanford.txt` and `top250movies.txt` in the working directory or input the appropriate path when invoking the pertinent functions.