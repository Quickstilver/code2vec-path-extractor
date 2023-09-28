import parso
from itertools import combinations
import hashlib



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

            # Use the sanitize_string function directly on first and last nodes
            first_node = sanitize_string(path[0])
            last_node = sanitize_string(path[-1])
            
            path_final = f"{first_node},{path_hash},{last_node} "
            method_ina_line.append(path_final)

    return method_ina_line

code = """
t = int(raw_input())

for case in range(t):
	print 'Case #'+str(case+1)+':'
	r, c, m = raw_input().split()
	r = int(r)
	c = int(c)
	m = int(m)

	if m==0:
		print 'c' + ('.'*(c-1))
		for i in range(r-1):
			print '.'*c
	elif r == 1:
		print 'c'+('.'*(c-m-1))+('*'*(m))
	elif c == 1:
		print 'c'
		for i in range(r-m-1):
			print '.'
		for i in range(m):
			print '*'
	elif r == 2:
		if m%2==0 and c>2 and m<r*c-2:
			print 'c'+('.'*(c-m/2-1))+('*'*(m/2))
			print ('.'*(c-m/2))+('*'*(m/2))
		elif m == r*c - 1:
			print 'c'+('*'*(c-1))
			print '*'*c
		else:
			print 'Impossible'
	elif c == 2:
		if m%2==0 and r>2 and m<r*c-2:
			print 'c.'
			for i in range(r-m/2-1):
				print '..'
			for i in range(m/2):
				print '**'
		elif m == r*c - 1:
			print 'c*'
			for i in range(r-1):
				print '**'
		else:
			print 'Impossible'
	elif c == 3 and r == 3:
		if m == 1:
			print 'c..'
			print '...'
			print '..*'
		elif m == 2:
			print 'Impossible'
		elif m == 3:
			print 'c..'
			print '...'
			print '***'
		elif m == 4:
			print 'Impossible'
		elif m == 5:
			print 'c.*'
			print '..*'
			print '***'
		elif m == 6:
			print 'Impossible'
		elif m == 7:
			print 'Impossible'
		elif m == 8:
			print 'c**'
			print '***'
			print '***'
	elif c == 3 and r == 4:
		if m == 1:
			print 'c..'
			print '...'
			print '...'
			print '..*'
		elif m == 2:
			print 'c..'
			print '...'
			print '..*'
			print '..*'
		elif m == 3:
			print 'c..'
			print '...'
			print '...'
			print '***'
		elif m == 4:
			print 'c.*'
			print '..*'
			print '..*'
			print '..*'
		elif m == 5:
			print 'Impossible'
		elif m == 6:
			print 'c..'
			print '...'
			print '***'
			print '***'
		elif m == 7:
			print 'Impossible'
		elif m == 8:
			print 'c.*'
			print '..*'
			print '***'
			print '***'
		elif m == 9:
			print 'Impossible'
		elif m == 10:
			print 'Impossible'
		elif m == 11:
			print 'c**'
			print '***'
			print '***'
			print '***'
	elif c == 3 and r == 5:
		if m == 1:
			print 'c..'
			print '...'
			print '...'
			print '...'
			print '..*'
		elif m == 2:
			print 'c..'
			print '...'
			print '...'
			print '..*'
			print '..*'
		elif m == 3:
			print 'c..'
			print '...'
			print '..*'
			print '..*'
			print '..*'
		elif m == 4:
			print 'c..'
			print '...'
			print '...'
			print '..*'
			print '***'
		elif m == 5:
			print 'c.*'
			print '..*'
			print '..*'
			print '..*'
			print '..*'
		elif m == 6:
			print 'c..'
			print '...'
			print '...'
			print '***'
			print '***'
		elif m == 7:
			print 'c..'
			print '...'
			print '..*'
			print '***'
			print '***'
		elif m == 8:
			print 'Impossible'
		elif m == 9:
			print 'c..'
			print '...'
			print '***'
			print '***'
			print '***'
		elif m == 10:
			print 'Impossible'
		elif m == 11:
			print 'c.*'
			print '..*'
			print '***'
			print '***'
			print '***'
		elif m == 12:
			print 'Impossible'
		elif m == 13:
			print 'Impossible'
		elif m == 14:
			print 'c**'
			print '***'
			print '***'
			print '***'
			print '***'
	elif c == 4 and r == 3:
		if m == 1:
			print 'c...'
			print '....'
			print '...*'
		elif m == 2:
			print 'c...'
			print '...*'
			print '...*'
		elif m == 3:
			print 'c..*'
			print '...*'
			print '...*'
		elif m == 4:
			print 'c...'
			print '....'
			print '****'
		elif m == 5:
			print 'Impossible'
		elif m == 6:
			print 'c.**'
			print '..**'
			print '..**'
		elif m == 7:
			print 'Impossible'
		elif m == 8:
			print 'c.**'
			print '..**'
			print '****'
		elif m == 9:
			print 'Impossible'
		elif m == 10:
			print 'Impossible'
		elif m == 11:
			print 'c***'
			print '****'
			print '****'
	elif c == 4 and r == 4:
		if m == 1:
			print 'c...'
			print '....'
			print '....'
			print '...*'
		elif m == 2:
			print 'c...'
			print '....'
			print '....'
			print '..**'
		elif m == 3:
			print 'c...'
			print '....'
			print '...*'
			print '..**'
		elif m == 4:
			print 'c...'
			print '....'
			print '....'
			print '****'
		elif m == 5:
			print 'c...'
			print '....'
			print '...*'
			print '****'
		elif m == 6:
			print 'c...'
			print '....'
			print '..**'
			print '****'
		elif m == 7:
			print 'c..*'
			print '...*'
			print '...*'
			print '****'
		elif m == 8:
			print 'c...'
			print '....'
			print '****'
			print '****'
		elif m == 9:
			print 'Impossible'
		elif m == 10:
			print 'c.**'
			print '..**'
			print '..**'
			print '****'
		elif m == 11:
			print 'Impossible'
		elif m == 12:
			print 'c.**'
			print '..**'
			print '****'
			print '****'
		elif m == 13:
			print 'Impossible'
		elif m == 14:
			print 'Impossible'
		elif m == 15:
			print 'c***'
			print '****'
			print '****'
			print '****'
	elif c == 4 and r == 5:
		if m == 1:
			print 'c...'
			print '....'
			print '....'
			print '....'
			print '...*'
		elif m == 2:
			print 'c...'
			print '....'
			print '....'
			print '....'
			print '..**'
		elif m == 3:
			print 'c...'
			print '....'
			print '...*'
			print '...*'
			print '...*'
		elif m == 4:
			print 'c...'
			print '....'
			print '....'
			print '....'
			print '****'
		elif m == 5:
			print 'c..*'
			print '...*'
			print '...*'
			print '...*'
			print '...*'
		elif m == 6:
			print 'c...'
			print '....'
			print '....'
			print '..**'
			print '****'
		elif m == 7:
			print 'c..*'
			print '...*'
			print '...*'
			print '..**'
			print '..**'
		elif m == 8:
			print 'c...'
			print '....'
			print '....'
			print '****'
			print '****'
		elif m == 9:
			print 'c..*'
			print '...*'
			print '...*'
			print '..**'
			print '****'
		elif m == 10:
			print 'c.**'
			print '..**'
			print '..**'
			print '..**'
			print '..**'
		elif m == 11:
			print 'c..*'
			print '...*'
			print '...*'
			print '****'
			print '****'
		elif m == 12:
			print 'c...'
			print '....'
			print '****'
			print '****'
			print '****'
		elif m == 13:
			print 'Impossible'
		elif m == 14:
			print 'c..*'
			print '...*'
			print '****'
			print '****'
			print '****'
		elif m == 15:
			print 'Impossible'
		elif m == 16:
			print 'c.**'
			print '..**'
			print '****'
			print '****'
			print '****'
		elif m == 17:
			print 'Impossible'
		elif m == 18:
			print 'Impossible'
		elif m == 19:
			print 'c***'
			print '****'
			print '****'
			print '****'
			print '****'
	elif c == 5 and r == 3:
		if m == 1:
			print 'c....'
			print '.....'
			print '....*'
		elif m == 2:
			print 'c....'
			print '.....'
			print '...**'
		elif m == 3:
			print 'c....'
			print '.....'
			print '..***'
		elif m == 4:
			print 'c...*'
			print '....*'
			print '...**'
		elif m == 5:
			print 'c....'
			print '.....'
			print '*****'
		elif m == 6:
			print 'c..**'
			print '...**'
			print '...**'
		elif m == 7:
			print 'c..**'
			print '...**'
			print '..***'
		elif m == 8:
			print 'Impossible'
		elif m == 9:
			print 'c.***'
			print '..***'
			print '..***'
		elif m == 10:
			print 'Impossible'
		elif m == 11:
			print 'c.***'
			print '..***'
			print '*****'
		elif m == 12:
			print 'Impossible'
		elif m == 13:
			print 'Impossible'
		elif m == 14:
			print 'c****'
			print '*****'
			print '*****'
	elif c == 5 and r == 4:
		if m == 1:
			print 'c....'
			print '.....'
			print '.....'
			print '....*'
		elif m == 2:
			print 'c....'
			print '.....'
			print '.....'
			print '...**'
		elif m == 3:
			print 'c....'
			print '.....'
			print '.....'
			print '..***'
		elif m == 4:
			print 'c...*'
			print '....*'
			print '....*'
			print '....*'
		elif m == 5:
			print 'c....'
			print '.....'
			print '.....'
			print '*****'
		elif m == 6:
			print 'c...*'
			print '....*'
			print '....*'
			print '..***'
		elif m == 7:
			print 'c....'
			print '.....'
			print '...**'
			print '*****'
		elif m == 8:
			print 'c..**'
			print '...**'
			print '...**'
			print '...**'
		elif m == 9:
			print 'c...*'
			print '....*'
			print '...**'
			print '*****'
		elif m == 10:
			print 'c....'
			print '.....'
			print '*****'
			print '*****'
		elif m == 11:
			print 'c..**'
			print '...**'
			print '...**'
			print '*****'
		elif m == 12:
			print 'c.***'
			print '..***'
			print '..***'
			print '..***'
		elif m == 13:
			print 'Impossible'
		elif m == 14:
			print 'c.***'
			print '..***'
			print '..***'
			print '*****'
		elif m == 15:
			print 'Impossible'
		elif m == 16:
			print 'c.***'
			print '..***'
			print '*****'
			print '*****'
		elif m == 17:
			print 'Impossible'
		elif m == 18:
			print 'Impossible'
		elif m == 19:
			print 'c****'
			print '*****'
			print '*****'
			print '*****'
	elif c == 5 and r == 5:
		if m == 1:
			print 'c....'
			print '.....'
			print '.....'
			print '.....'
			print '....*'
		elif m == 2:
			print 'c....'
			print '.....'
			print '.....'
			print '.....'
			print '...**'
		elif m == 3:
			print 'c....'
			print '.....'
			print '.....'
			print '.....'
			print '..***'
		elif m == 4:
			print 'c....'
			print '.....'
			print '.....'
			print '...**'
			print '...**'
		elif m == 5:
			print 'c....'
			print '.....'
			print '.....'
			print '.....'
			print '*****'
		elif m == 6:
			print 'c....'
			print '.....'
			print '.....'
			print '..***'
			print '..***'
		elif m == 7:
			print 'c....'
			print '.....'
			print '.....'
			print '...**'
			print '*****'
		elif m == 8:
			print 'c....'
			print '.....'
			print '.....'
			print '..***'
			print '*****'
		elif m == 9:
			print 'c...*'
			print '....*'
			print '....*'
			print '....*'
			print '*****'
		elif m == 10:
			print 'c....'
			print '.....'
			print '.....'
			print '*****'
			print '*****'
		elif m == 11:
			print 'c....'
			print '.....'
			print '....*'
			print '*****'
			print '*****'
		elif m == 12:
			print 'c....'
			print '.....'
			print '...**'
			print '*****'
			print '*****'
		elif m == 13:
			print 'c....'
			print '.....'
			print '..***'
			print '*****'
			print '*****'
		elif m == 14:
			print 'c..**'
			print '...**'
			print '...**'
			print '..***'
			print '*****'
		elif m == 15:
			print 'c....'
			print '.....'
			print '*****'
			print '*****'
			print '*****'
		elif m == 16:
			print 'c..**'
			print '...**'
			print '...**'
			print '*****'
			print '*****'
		elif m == 17:
			print 'c..**'
			print '...**'
			print '..***'
			print '*****'
			print '*****'
		elif m == 18:
			print 'Impossible'
		elif m == 19:
			print 'c..**'
			print '...**'
			print '*****'
			print '*****'
			print '*****'
		elif m == 20:
			print 'Impossible'
		elif m == 21:
			print 'c.***'
			print '..***'
			print '*****'
			print '*****'
			print '*****'
		elif m == 22:
			print 'Impossible'
		elif m == 23:
			print 'Impossible'
		elif m == 24:
			print 'c****'
			print '*****'
			print '*****'
			print '*****'
			print '*****'
	else:
		print 'Impossible'
"""

module = parso.parse(code)
print('parsato')
leaf_nodes = get_leaf_nodes(module)
print('foglie estratte')

paths = extract_paths(module, leaf_nodes)
print('path calcolati')
concatenated_paths = concatenate_paths(paths)
print('path concatenati')
hashed_results = hash_paths(concatenated_paths)
print('print hashed')

for line in hashed_results:
    print(line)
    print('----')


