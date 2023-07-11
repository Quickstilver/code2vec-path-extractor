from tree_sitter import Language, Parser
import itertools
import os
import random
import hashlib

def trailing_space(node,contents):

    start_row=node.start_point[0]
    end_row=node.end_point[0]
    start_column=node.start_point[1]
    end_columns=node.end_point[1]

    # the spaces before the node
    start_byte = node.start_byte
    while start_byte > 0 and contents[start_byte-1:start_byte] in (b' ', b'\n', b'\t'):
        start_byte -= 1
    spaces_before = contents[start_byte:node.start_byte].decode('utf-8')

    # the spaces after the node
    end_byte = node.end_byte
    while end_byte < len(contents) and contents[end_byte:end_byte+1] in (b' ', b'\n', b'\t'):
        end_byte += 1
    spaces_after = contents[node.end_byte:end_byte].decode('utf-8')

    return spaces_before,spaces_after


def traverse(tree,token,token_dict,contents):        

    def _traverse(node):
        for child in node.children:

            if child.children==[]:
                name= contents[child.start_byte:child.end_byte].decode('utf8')
                if child.type!='comment':                            #filtra fuori i commenti
                    token.append(name)
                    token_dict.append(child)
 
            _traverse(child)     

    _traverse(tree.root_node)    
    return(token,token_dict)


def leaf2leaf2(leaf_node, tree, content): #path in a form Terminal-HASH-Terminal, original code2vec format
    method_ina_line = []

    comb = list(itertools.combinations(leaf_node, 2))

    for pair in comb:
        start, end = pair

        start_bf, start_after = trailing_space(start, content)
        start_token = start_bf + content[start.start_byte:start.end_byte].decode('utf8') + start_after
        end_bf, end_after = trailing_space(end, content)
        end_token = end_bf + content[end.start_byte:end.end_byte].decode('utf8') + end_after

        start_token = start_token.replace(" ", "WS").replace(",", "CMA").replace("\n", "NL").replace("\t", "TAB")
        end_token = end_token.replace(" ", "WS").replace(",", "CMA").replace("\n", "NL").replace("\t", "TAB")

        pathUP, pathDOWN = [start_token], [end_token, "<-"]
        p1, p2 = start.parent, end.parent

        pathUP.append(p1.type)
        pathDOWN.extend([p2.type,"<-"])

        while p1 != p2:
            p1, p2 = p1.parent, p2.parent
            if p1:
                pathUP.append(p1.type)
            if p2:
                pathDOWN.extend([p2.type,"<-"])

        pathDOWN.reverse()
        dirty_path = pathUP + pathDOWN
        path2 = []
        go_down = False

        for n, item in enumerate(dirty_path):
            if item not in path2:
                path2.append(item)
                if item == "<-":
                    go_down = True
                if not go_down and dirty_path[n+1] != "<-":
                    path2.append("->")
                if item != "<-" and go_down and (n+1 < len(dirty_path)):
                    path2.append("<-")

        if (len(path2) + 1) // 2 < 9: ###lenght of path reale senza le tokenIniziali è len(path2)-2 DIV2 con DIV2 divisione intera, ora è settata a 7
            path_string = str(path2[1:-1]).encode('utf-8')
            path_hash = str(hashlib.sha256(path_string).hexdigest())
            path_final = f"{start_token},{path_hash},{end_token} "
            method_ina_line.append(path_final)

    return method_ina_line

def leaf2leaf3(leaf_node, tree, content): #path in a form Terminal- Path String - Terminal
    method_ina_line = []

    comb = list(itertools.combinations(leaf_node, 2))

    for pair in comb:
        start, end = pair

        start_bf, start_after = trailing_space(start, content)
        start_token = start_bf + content[start.start_byte:start.end_byte].decode('utf8') + start_after
        end_bf, end_after = trailing_space(end, content)
        end_token = end_bf + content[end.start_byte:end.end_byte].decode('utf8') + end_after

        start_token = start_token.replace(" ", "WS").replace(",", "CMA").replace("\n", "NL").replace("\t", "TAB")
        end_token = end_token.replace(" ", "WS").replace(",", "CMA").replace("\n", "NL").replace("\t", "TAB")

        pathUP, pathDOWN = [start_token], [end_token, "<-"]
        p1, p2 = start.parent, end.parent

        pathUP.append(p1.type)
        pathDOWN.extend([p2.type,"<-"])

        while p1 != p2:
            p1, p2 = p1.parent, p2.parent
            if p1:
                pathUP.append(p1.type)
            if p2:
                pathDOWN.extend([p2.type,"<-"])

        pathDOWN.reverse()
        dirty_path = pathUP + pathDOWN
        path2 = []
        go_down = False

        for n, item in enumerate(dirty_path):
            if item not in path2:
                path2.append(item)
                if item == "<-":
                    go_down = True
                if not go_down and dirty_path[n+1] != "<-":
                    path2.append("->")
                if item != "<-" and go_down and (n+1 < len(dirty_path)):
                    path2.append("<-")

        if (len(path2) + 1) // 2 < 9: ###lenght of path reale senza le tokenIniziali è len(path2)-2 DIV2 con DIV2 divisione intera, ora è settata a 7
            path_string = str(path2[1:-1]).encode('utf-8')
            path_hash = str(hashlib.sha256(path_string).hexdigest())
            path_final = f"{start_token},{path_string},{end_token} "
            method_ina_line.append(path_final)

    return method_ina_line


def leaf2leaf4(leaf_node, tree, content): #it use the content of the node instead of the type of the node as leaf2leaf3
    method_ina_line = []

    comb = list(itertools.combinations(leaf_node, 2))

    for pair in comb:
        start, end = pair

        start_token = content[start.start_byte:start.end_byte].decode('utf8')
        end_token = content[end.start_byte:end.end_byte].decode('utf8')

        start_token = start_token.replace(" ", "WS").replace(",", "CMA").replace("\n", "NL")
        end_token = end_token.replace(" ", "WS").replace(",", "CMA").replace("\n", "NL")

        pathUP, pathDOWN = [start_token], [end_token, "<-"]
        p1, p2 = start.parent, end.parent
        
        pathUP.append(content[p1.start_byte:p1.end_byte].decode('utf8'))
        pathDOWN.extend([content[p2.start_byte:p2.end_byte].decode('utf8'),"<-"])

        while p1 != p2:
            p1, p2 = p1.parent, p2.parent
            if p1:
                pathUP.append(content[p1.start_byte:p1.end_byte].decode('utf8'))
            if p2:
                pathDOWN.extend([content[p2.start_byte:p2.end_byte].decode('utf8'),"<-"])

        pathDOWN.reverse()
        dirty_path = pathUP + pathDOWN
        path2 = []
        go_down = False

        for n, item in enumerate(dirty_path):
            if item not in path2:
                path2.append(item)
                if item == "<-":
                    go_down = True
                if not go_down and dirty_path[n+1] != "<-":
                    path2.append("->")
                if item != "<-" and go_down and (n+1 < len(dirty_path)):
                    path2.append("<-")

        if (len(path2) + 1) // 2 < 9: ###lenght of path reale senza le tokenIniziali è len(path2)-2 DIV2 con DIV2 divisione intera, ora è settata a 7
            path_string = str(path2[1:-1]).encode('utf-8')
            path_hash = str(hashlib.sha256(path_string).hexdigest())
            path_final = f"{start_token},{path_hash},{end_token} "
            method_ina_line.append(path_final)

    return method_ina_line


def print_node(node, source_code, depth=0, indent=''):

    start_bf, start_after = trailing_space(node, source_code)
    node_code = start_bf + source_code[node.start_byte:node.end_byte].decode('utf8') + start_after
    node_code = node_code.replace(" ", "WS").replace(",", "CMA").replace("\n", "NL").replace("\t", "TAB")

    print(f"Node{depth} {indent}{node.type}: {node_code}")

    if node.children:
        for child in node.children:
            print_node(child, source_code, depth + 1, indent + '-')



def create_dataset(origin, destination):

    PY_LANGUAGE = Language('build/my-languages.so', 'python')   ##importa il linguaggio
    parser = Parser()  
    parser.set_language(PY_LANGUAGE) #crea il parser per python

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
                
                with open(doc, 'rb') as f:

                    token=[]
                    token_node=[] 
                    contents = f.read()
                    tree = parser.parse(contents)
                    
                    print_node(tree.root_node, contents)
                    leaf,leaf_node= traverse(tree,token,token_node,contents)
                    example=set(leaf2leaf3(leaf_node,tree,contents)) #in leaf2leaf viene definita lenght max path

                    if len(example)!=0:

                        if len(example)>200:   ###definisce il numero di path max per riga di esempio
                            l=l+1
                            example=random.sample(example, 200)

                        strexample = ''.join(example)

                    file_int = strexample + file_int
                
                    if len(file_int)>0:
                        target_example= target + " " + file_int +'\n'
                        dataset.write(target_example)  #scrivere su txt

                
    print("dataset created")
    print("numero example>200", l)

