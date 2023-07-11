'''
import ast
import astunparse

class Node:
    def __init__(self, node, parent=None):
        self.node = node
        self.parent = parent
        self.children = []

    def add_child(self, node):
        self.children.append(node)


def traverse_tree_and_build(root_node, node, parent=None):
    root_node = Node(node, parent)
    for child in ast.iter_child_nodes(node):
        root_node.add_child(traverse_tree_and_build(root_node, child, root_node))
    return root_node


def get_leaves(node, leaves=None):
    if leaves is None:
        leaves = []
    for child in node.children:
        if len(child.children) == 0:
            leaves.append(child)
        else:
            get_leaves(child, leaves)
    return leaves


def get_path_from_to(n1, n2, path=None):
    if path is None:
        path = []
    if n1 == n2:
        return path + [n1]
    if n1 is None:
        return None
    return get_path_from_to(n1.parent, n2, path + [n1])


def node_to_str(node, source_code):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        return str(node.value)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return '+'
    elif isinstance(node, ast.Load):
        return 'Load'
    elif isinstance(node, ast.Store):
        return 'Store'
    else:
        return type(node).__name__


def get_leaf_to_leaf_paths(source_code):
    tree = ast.parse(source_code)
    root_node = traverse_tree_and_build(None, tree)
    leaves = get_leaves(root_node)
    paths = []
    for i in range(len(leaves)):
        for j in range(i+1, len(leaves)):
            path = get_path(leaves[i], leaves[j], source_code)
            if path:
                paths.append(path)
    return paths


def get_path(n1, n2, source_code):
    lca = get_lowest_common_ancestor(n1, n2)
    if lca is None:
        return
    path1 = get_path_from_to(lca, n1)
    path2 = get_path_from_to(lca, n2)
    path1 = [node_to_str(n.node, source_code) for n in reversed(path1)]
    path2 = [node_to_str(n.node, source_code) for n in path2[1:]]
    return path1 + path2


def get_lowest_common_ancestor(n1, n2):
    path1 = get_path_from_to(n1, n2)
    path2 = get_path_from_to(n2, n1)
    if path1 is None or path2 is None:
        return None
    for node in reversed(path1):
        if node in path2:
            return node
    return None


source_code = """
def function(a,b):


    result=a+b


    print('Result: ',  result)
"""

paths = get_leaf_to_leaf_paths(source_code)

for path in paths:
    print(path)






print("the whole tree",astunparse.dump(ast.parse(source_code)))
'''

import ast
from itertools import combinations

def add_parents(node, parent=None):
    node.parent = parent
    for child in ast.iter_child_nodes(node):
        add_parents(child, node)

def get_path_to_root(node):
    path = [node]
    while hasattr(node, 'parent') and node.parent is not None:
        node = node.parent
        path.append(node)
    return path

def get_common_ancestor(node1, node2):
    path1 = set(get_path_to_root(node1))
    path2 = set(get_path_to_root(node2))

    return path1 & path2

def leaf_to_leaf_paths(node):
    leaves = [leaf for leaf in ast.walk(node) if not list(ast.iter_child_nodes(leaf))]
    paths = []
    for leaf1, leaf2 in combinations(leaves, 2):
        common_ancestor = max(get_common_ancestor(leaf1, leaf2), key=lambda n: n.depth)
        path1 = get_path_to_root(leaf1)[:leaf1.depth - common_ancestor.depth + 1]
        path2 = get_path_to_root(leaf2)[:leaf2.depth - common_ancestor.depth][::-1]
        path = path1 + path2
        # Format the output string to reflect the direction of traversal
        formatted_path = ''
        for i in range(len(path) - 1):
            if i < len(path1) - 1:
                formatted_path += type(path[i]).__name__ + ' -> '
            else:
                formatted_path += type(path[i]).__name__ + ' <- '
        formatted_path += type(path[-1]).__name__
        paths.append(formatted_path)

    return paths

# Add depth and parents to each node
def add_depth_and_parents(node, depth=0, parent=None):
    node.depth = depth
    node.parent = parent
    for child in ast.iter_child_nodes(node):
        add_depth_and_parents(child, depth + 1, node)

code = """
def function(a,b):
    result = a + b
    print('Result: ', result)
"""

tree = ast.parse(code)
add_depth_and_parents(tree)

paths = leaf_to_leaf_paths(tree)

for path in paths:
    print(path)



