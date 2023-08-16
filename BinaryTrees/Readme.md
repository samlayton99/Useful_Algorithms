# Binary Trees Project Description

A tree is link-based data structure where each node may refer to more than one other node. This structure makes trees more useful and efficient than regular linked lists in many applications. Many trees are constructed recursively, so I begin with an overview of recursion. I then implement a recursively structured doubly linked binary search tree (BST). Finally, I compare the standard linked list, our BST, and an AVL tree to illustrate the relative strengths and weaknesses of each data structure.

## Conceptual Framework and Mathematical Foundations

Trees, as defined in computational mathematics, are non-linear hierarchical structures defined by nodes and edges. Their distinct branching property facilitates efficient data management and retrieval. The essential mathematical concepts include:

1. **Tree**: Trees, in computational terms, represent a connected acyclic graph. Their hierarchical nature ensures a logarithmic time complexity for retrieval operations, providing an advantage over linear data structures like linked lists.
2. **Recursion**: A fundamental concept in both mathematics and computer science, recursion is the self-referential technique wherein a function invokes itself. This iterative strategy is especially useful in problems that exhibit fractal or self-replicating patterns, often seen in tree traversal and operations.
3. **Binary Search Tree (BST)**: This is a specialized tree that upholds the binary search property. Mathematically:
    - For each node, all elements in the left subtree are less than the node.
    - All elements in the right subtree are greater than the node.
4. **AVL Tree**: An AVL tree extends the properties of a BST. It is a balanced binary search tree where the difference between the heights of the left and right subtrees (the balance factor) is no more than one for all nodes. This balancing ensures logarithmic bounds for all operations, making it mathematically efficient.

## Detailed Structures and Functions

1. **SinglyLinkedListNode**:
    - `value`: A data storage segment derived from a set domain.
    - `next`: A reference pointer directing to the subsequent node.
2. **SinglyLinkedList**:
    - `head` & `tail`: Pointers that delineate the boundaries of the list.
    - `append(data)`: A method to append data, ensuring data integrity.
    - `iterative_find(data)`: A retrieval function with a predefined error threshold.
3. **BSTNode**:
    - `value`: A primary data storage element.
    - `prev`: An ancestral reference, often utilized in tree traversals.
    - `left` & `right`: Pointers distinguishing the hierarchical level of nodes.
4. **BST**:
    - `root`: Denotes the primary or foundational node, a cornerstone in tree traversal algorithms.
5. **AVL** (Derived from BST):
    - Inherits the mathematical properties and functions of BST while introducing balancing mechanisms.

## Structured Project Progression

1. **Recursive Methodologies**:
    - An examination into the mathematical underpinnings of recursion, discerning its iterative nature and base-case significance.
2. **Mastery over Singly Linked Lists**:
    - A rigorous exploration of the `SinglyLinkedList` structure, emphasizing its linear characteristics.
    - Application of recursive methodologies in data retrieval to showcase similarities with the `BST.find()` method.
3. **Binary Search Tree (BST) Architectural Understanding**:
    - Decoding the mathematical intricacies that define BSTs.
    - Evaluating node addition algorithms and the subsequent challenges in node removal algorithms, considering various complexities.
4. **Advanced AVL Tree Constructs**:
    - An introduction to AVL trees' auto-rebalancing algorithms.
    - Analyzing the AVL's rebalancing algorithms' mathematical foundations.
5. **Performance Analytics**:
    - A rigorous analytical comparison of `SinglyLinkedList`, `BST`, and `AVL` trees concerning construction and retrieval efficiencies, underpinned by real-world datasets and graph theories.

## Applicability in Advanced Domains

- Trees, especially BSTs and AVL trees, manifest as optimal solutions in data-heavy applications, ensuring logarithmic time complexities.
- Singly linked lists serve as a benchmark in linear data structures, showcasing constant-time dynamics.
- The inherent balancing mechanisms of AVL trees ensure consistent performance, even in edge-case scenarios.
- A thorough comparison of the structures provides a holistic understanding, allowing computational experts to make informed decisions based on specific problem domains.

## Dependencies

```python
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import random
import time