# binary_trees.py
"""Volume 2: Binary Trees.
<Sam Layton>
<Section 2>
<9/30/22>
"""

# These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import random
import time


class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        # Define a helper function to recursively search the list.
        def helper(node):
            # If the node is None, the data is not in the list.
            if node is None:
                raise ValueError(str(data) + " is not in the list")
            # If the node contains the data, return the node.
            if node.value == data:
                return node
            # Otherwise, search the rest of the list.
            return helper(node.next)

        # Call the helper function on the head of the list.
        return helper(self.head)

class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        # Create a new node to insert.
        newnode = BSTNode(data)

        # If the tree is empty, insert the new node as the root.
        if self.root is None:
            self.root = newnode
            return self.root

        # Otherwise, find the correct place to insert the new node.
        else:

            # Define a recursive function to traverse the tree.
            def _step(current, parent = None):
                """Recursively step through the tree until the correct node is inserted."""
                # Base case, when the current node is None, insert the new node.
                if current is None:
                    # Check if it is a left child
                    if data < parent.value:
                        parent.left = newnode
                        newnode.prev = parent
                    # If it is a right child, insert it as a right child
                    else:
                        parent.right = newnode
                        newnode.prev = parent
                    return newnode

                # If the data is already in the tree, raise a ValueError.
                else:
                    if current.value == data:
                        raise ValueError(str(data) + " is already in the tree.")
                
                # Recursively search left and right.
                if data < current.value:                # Recursively search left.
                    return _step(current.left, current)
                else:                                   # Recursively search right.
                    return _step(current.right, current)
            
            # Start the recursion on the root of the tree.
            _step(self.root)


        

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        # Find the node containing the data.
        node = self.find(data)

        # Case with no children (no left or right child)
        if node.left is None and node.right is None:
            # If only root node, set root to None
            if node.prev is None:
                self.root = None
            # Otherwise, remove the node from the tree and set the parent's child to None
            else:
                if node.prev.left is node:
                    node.prev.left = None
                else:
                    node.prev.right = None

        # Case where there is only one child
        elif node.left is None or node.right is None:
            # If the node is the root, replace the root with the child.
            if node.prev is None:
                if node.left is not None:
                    self.root = node.left
                    node.left.prev = None
                # If the child is on the right, replace the root with the right child
                else:
                    self.root = node.right
                    node.right.prev = None
            # Otherwise, replace the node with the child.
            else:
                if node.left is not None:
                    # If the node is the left child of its parent, replace it with its left child
                    if node.prev.left is node:
                        node.prev.left = node.left
                    # If the node is the right child of its parent, replace it with its left child
                    else:
                        node.prev.right = node.left
                    node.left.prev = node.prev
                # For when the child is on the right
                else:
                    if node.prev.left is node:
                        node.prev.left = node.right
                    # If the node is the right child of its parent, replace it with its the right child
                    else:
                        node.prev.right = node.right
                    node.right.prev = node.prev

        # Case where there are two children
        else:
            # Find the largest node in the left subtree.
            current = node.left
            while current.right is not None:
                current = current.right

            # Replace the node with the smallest node in the right subtree.
            node.value = current.value

            # Remove the largest node in the left subtree.
            if current.left is not None:
                current.left.prev = current.prev
                current.prev.left = current.left
            else:
                current.prev.right = None
            

    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """

    # import random sample of data of size n from "english.txt" file and store in a list
    with open("input_files/english.txt", "r") as f:
        data = f.readlines()
    data = [x.strip() for x in data]

    # Initialize buildt time lists to store times for each data structure
    bst_build_times = []
    avl_build_times = []
    sll_build_times = []

    # Initialize search time lists to store times for each data structure
    bst_search_times = []
    avl_search_times = []
    sll_search_times = []

    # create a list of random sample of size n^i from the data list
    maxpower = 10
    for i in range(3,maxpower + 1):
        n = 2**i
        sample = random.sample(data, n)
        elements = random.sample(sample, 5)

        # Initialized the SinglyLinkedList, BST, and AVL trees and put them in a list
        sll = SinglyLinkedList()
        bst = BST()
        avl = AVL()
        structures = [sll, bst, avl]
        
        
        # For each data structure, time the start and end times of appending/inserting from sample
        for structure in structures:
            #handle the case of SinglyLinkedList frst
            if structure == sll:
                start = time.time()
                # append each element in the sample to the SinglyLinkedList
                for item in sample:
                    structure.append(item)
                end = time.time()
                # append to build time list
                sll_build_times.append(end-start)

                # search for 5 random elements in the SinglyLinkedList
                start = time.time()
                # find each element in the singly linked list
                for item in elements:
                    structure.iterative_find(item)
                end = time.time()
                # append to build time list
                sll_search_times.append(end-start)


            # handle the case of BST and AVL trees
            else:
                # insert each element in the sample to the BST or AVL tree
                start = time.time()
                for item in sample:
                    structure.insert(item)
                end = time.time()

                # append to the correct list depending on the data structure
                if structure == bst:
                    bst_build_times.append(end-start)
                else:
                    avl_build_times.append(end-start)

                # search for 5 random elements in the BST or AVL tree
                start = time.time()
                # find each element in the BST or AVL tree
                for item in elements:
                    structure.find(item)
                end = time.time()

                # append to correct search list depending on the data structure
                if structure == bst:
                    bst_search_times.append(end-start)
                else:
                    avl_search_times.append(end-start)


    # plot the build times for each data structure
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)

    # Generate the x-axis values and plot the build times for each data structure
    plt.plot([2**i for i in range(3,maxpower + 1)], sll_build_times, label="SinglyLinkedList")
    plt.plot([2**i for i in range(3,maxpower + 1)], bst_build_times, label="BST")
    plt.plot([2**i for i in range(3,maxpower + 1)], avl_build_times, label="AVL")

    # Set the x and y labels and title
    plt.xlabel("Number of Items")
    plt.ylabel("Build Time")
    plt.title("Build Time vs. Number of Items")

    # Set the x and y scales to log and add a legend
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    # plot the search times for each data structure
    plt.subplot(1,2,2)
    plt.plot([2**i for i in range(3,maxpower + 1)], sll_search_times, label="SinglyLinkedList")
    plt.plot([2**i for i in range(3,maxpower + 1)], bst_search_times, label="BST")
    plt.plot([2**i for i in range(3,maxpower + 1)], avl_search_times, label="AVL")

    # Set the x and y labels and title
    plt.xlabel("Number of Items")
    plt.ylabel("Search Time")
    plt.title("Search Time vs. Number of Items")

    # Set the x and y scales to log and add a legend
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    # Show the plot and adjust spacing
    plt.tight_layout()
    plt.show()

