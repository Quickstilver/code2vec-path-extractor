
import random
import os

from typed_ast import ast27 as ast
from itertools import combinations
import hashlib


class RemoveLoadNode(ast.NodeTransformer):
    '''
    Removes all occurrences of the ast.Load node from the AST.
    '''
    def generic_visit(self, node):
        """
        Called if no explicit visitor function exists for a node.
        """
        for field, old_value in ast.iter_fields(node):
            # For singular nodes
            if isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    setattr(node, field, None)  # Set field to None if Load node is encountered and removed
                elif new_node != old_value:
                    setattr(node, field, new_node)
            # For lists of nodes
            elif isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is not None:  # Add to the list only if it's not a Load node
                            new_values.append(value)
                old_value[:] = new_values
        return node

    def visit_Load(self, node):
        return None  # Removes the Load node by returning None


# Determine parent for each node
def set_parents(node, parent=None):
    for child in ast.iter_child_nodes(node):
        child.parent = parent
        set_parents(child, child)

def get_leaf_nodes(node):
    if not list(ast.iter_child_nodes(node)):
        return [node]
    else:
        leaves = []
        for child in ast.iter_child_nodes(node):
            leaves.extend(get_leaf_nodes(child))
        return leaves

def get_path_upwards(node, target_ancestor):
    path = []
    while node.parent and node != target_ancestor:
        node = node.parent
        path.append(node)
    return path


def find_common_ancestor(node1, node2):
    ancestors_node1 = set()
    current = node1
    while hasattr(current, "parent"):
        ancestors_node1.add(current)
        current = current.parent

    current = node2
    while hasattr(current, "parent"):
        if current in ancestors_node1:
            return current
        current = current.parent
    return None

def get_code_snippet(node, code):
    lines = code.splitlines()

    # Handling for Compare nodes to get the entire comparison as a snippet
    if isinstance(node, ast.Compare):
        left_value = get_code_snippet(node.left, code)
        op = get_code_snippet(node.ops[0], code)  # For simplicity, just handle the first operator
        comparator = get_code_snippet(node.comparators[0], code)  # Similarly, handle the first comparator
        return f"{left_value} {op} {comparator}"

    if hasattr(node, 'lineno') and hasattr(node, 'col_offset') and hasattr(node, 'end_lineno') and hasattr(node, 'end_col_offset'):
        # Extract code snippet based on lineno and col_offset attributes
        start_line = lines[node.lineno - 1][node.col_offset:]
        end_line = lines[node.end_lineno - 1][:node.end_col_offset]
        middle_lines = lines[node.lineno:node.end_lineno - 1]
        return ' '.join([start_line] + middle_lines + [end_line]).strip()

    if isinstance(node, ast.Eq):
        return "=="	

    # Mapping based on provided list
    attribute_mapping = {
        ast.Attribute: "value",
        ast.Subscript: "value",
        ast.List: "elts",
        ast.Tuple: "elts",
        ast.Slice: "lower",
        ast.ExtSlice: "dims",
        ast.Index: "value",
        ast.ExceptHandler: "name",
        ast.TypeIgnore: "lineno",
        ast.Module: "body",
        ast.Interactive: "body",
        ast.Expression: "body",
        ast.FunctionType: ["argtypes", "returns"],
        ast.Suite: "body",
        ast.FunctionDef: "name",
        ast.ClassDef: "name",
        ast.Return: "value",
        ast.Delete: "targets",
        ast.Assign: "targets",
        ast.AugAssign: "target",
        ast.Print: "dest",
        ast.For: "target",
        ast.While: "test",
        ast.If: "test",
        ast.With: "context_expr",
        ast.Raise: "type",
        ast.TryExcept: "body",
        ast.TryFinally: "body",
        ast.Assert: "test",
        ast.Import: "names",
        ast.ImportFrom: "module",
        ast.Exec: "body",
        ast.Global: "names",
        ast.Expr: "value",
        ast.BoolOp: "op",
        ast.BinOp: "left",
        ast.UnaryOp: "op",
        ast.Lambda: "args",
        ast.IfExp: "test",
        ast.Dict: "keys",
        ast.Set: "elts",
        ast.ListComp: "elt",
        ast.SetComp: "elt",
        ast.DictComp: "key",
        ast.GeneratorExp: "elt",
        ast.Yield: "value",
        ast.Compare: "left",
        ast.Call: "func",
        ast.Repr: "value",
        ast.Num: "n",
        ast.Str: "s",
		ast.Name : "id"
    }

    attr_name = attribute_mapping.get(type(node))
    if attr_name:
        if isinstance(attr_name, list):
            # If there are multiple attributes, combine their values (this is a basic assumption, might need adjustment)
            return " ".join([str(getattr(node, name)) for name in attr_name if hasattr(node, name)])
        elif hasattr(node, attr_name):
            return str(getattr(node, attr_name))

    return str(type(node).__name__)  # Fallback to node type if we can't extract snippet





def extract_paths(module, leaf_nodes, code):
    comb = list(combinations(leaf_nodes, 2))
    all_paths = []

    for node1, node2 in comb:
        common_ancestor = find_common_ancestor(node1, node2)
        
        path1UP = [str(type(node).__name__) for node in get_path_upwards(node1, common_ancestor)]
        path2UP = [str(type(node).__name__) for node in get_path_upwards(node2, common_ancestor)]
        
        # For the leaf nodes, we extract the actual code snippet
        path1UP.insert(0, get_code_snippet(node1, code))
        path2UP.insert(0, get_code_snippet(node2, code))
        #print(path1UP[0])
        #print(path2UP[0])

        path2UP.reverse()
        all_paths.append((path1UP, path2UP))
    
    return all_paths


def concatenate_paths(paths):
    concatenated = []
    
    for path1, path2 in paths:
        new_path = []
        for node in path1[:-1]:
            new_path.append(node)
            new_path.append('->')
        for node in path2:
            new_path.append(node)
            new_path.append('<-')
        new_path = new_path[:-1]
        concatenated.append(new_path)
    
    return concatenated

def sanitize_string(s):
    return s.replace(" ", "").replace(",", "").replace("\n", "").replace("\t", "")

def hash_paths(concatenated_paths):
    method_ina_line = []

    for path in concatenated_paths:
        real_path_length = (len(path) - 2) // 2
        if real_path_length < 9:
            path_string = str(path[1:-1]).encode('utf-8')
            path_hash = str(hashlib.sha256(path_string).hexdigest())
            formatted_hash = f'b"{path_hash}"'

            first_node = sanitize_string(str(path[0]))
            last_node = sanitize_string(str(path[-1]))
            
            path_final = f"{first_node},{formatted_hash},{last_node} "
            method_ina_line.append(path_final)

    return method_ina_line





def create_ast_dataset(origin, destination):

    curr=os.getcwd()
    origin= os.path.join(curr, origin)
    destination_folder = os.path.join(curr, destination)
    destination_file = os.path.join(destination_folder, "datasetgrezzo.txt")
    os.makedirs(destination_folder, exist_ok=True)
    l=0

    with open(destination_file, 'w',encoding="utf-8") as dataset:

        for root, dirs, files in os.walk(origin, topdown = False):

            for name in files:

                file_int=""
                target_example=""
                strexample=""
                example=""
                target=os.path.split(os.path.split(root)[1])[1]
                print(target)
                doc= os.path.join(root, name)
                
                try:  # Start of try block
                    with open(doc, 'rb') as f:

                        contents = f.read().decode("utf-8")                    
                        
                        module = ast.parse(contents)
                        module = RemoveLoadNode().visit(module)
                        set_parents(module) 
                        leaf_nodes = get_leaf_nodes(module)
                        paths = extract_paths(module, leaf_nodes, contents)
                        concatenated_paths = concatenate_paths(paths)
                        example = hash_paths(concatenated_paths)
                        example=set(example)

                        if len(example)!=0:

                            if len(example)>200:   ###definisce il numero di path max per riga di esempio
                                l=l+1
                                example = random.sample(example, 200)

                            strexample = ''.join(example)

                        file_int = strexample + file_int
                    
                        if len(file_int)>0:
                            target_example= target + " " + file_int +'\n'
                            dataset.write(target_example)  #scrivere su txt

                except Exception as e:  # Handle the exception
                    print(f"An error occurred processing the file: {doc}")
                    print(f"Error: {e}")
                
    print("dataset created")
    print("numero example>200", l)



