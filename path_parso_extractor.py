
import random
import os

from itertools import combinations
import hashlib

import parso


def get_leaf_nodes(node):
    if not hasattr(node, 'children'):
        return [node]
    else:
        leaves = []
        for child in node.children:
            leaves.extend(get_leaf_nodes(child))
        return leaves

def get_path_upwards(node, target_ancestor):
    path = [node]
    while node != target_ancestor:
        node = node.parent
        path.append(node)
    return path

def find_common_ancestor(node1, node2):
    ancestors_node1 = set()
    current = node1
    while current:
        ancestors_node1.add(current)
        current = current.parent

    current = node2
    while current:
        if current in ancestors_node1:
            return current
        current = current.parent
    return None

def extract_paths(module, leaf_nodes):
    comb = list(combinations(leaf_nodes, 2))
    all_paths = []

    for node1, node2 in comb:
        common_ancestor = find_common_ancestor(node1, node2)
        
        path1UP = [str(node.type) for node in get_path_upwards(node1, common_ancestor)]
        path2UP = [str(node.type) for node in get_path_upwards(node2, common_ancestor)]
        
        # For the leaf nodes, instead of type, we'll use value
        path1UP[0] = str(node1.value)
        path2UP[0] = str(node2.value)

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
        # Remove the last added direction arrow (either '->' or '<-') since it's not needed
        new_path = new_path[:-1]
        concatenated.append(new_path)
    
    return concatenated

def sanitize_string(s):
    return s.replace(" ", "").replace(",", "").replace("\n", "").replace("\t", "")

def hash_paths(concatenated_paths):
    method_ina_line = []

    for path in concatenated_paths:
        real_path_length = (len(path) - 2) // 2

        # Adjust the check based on real_path_length without first and last nodes
        if real_path_length < 9: 
            path_string = str(path[1:-1]).encode('utf-8')
            path_hash = hashlib.sha256(path_string).hexdigest()
            formatted_hash = f'b"{path_hash}"'

            # Use the sanitize_string function directly on first and last nodes
            first_node = sanitize_string(path[0])
            last_node = sanitize_string(path[-1])
            
            path_final = f"{first_node},{formatted_hash},{last_node} "
            method_ina_line.append(path_final)

    return method_ina_line



def create_parso_dataset(origin, destination):

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
                        
                        module = parso.parse(contents)
                        leaf_nodes = get_leaf_nodes(module)
                        paths = extract_paths(module, leaf_nodes)
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
