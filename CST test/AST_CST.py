#AST CST CST_blanks

import ast
import astunparse
from tree_sitter import Language, Parser

src_code1='my_file.txt'

def create_ast(src):

    tree = ast.parse(src)

    print("the whole tree",astunparse.dump(tree))


def traverse_tree(tree):
    cursor = tree.walk()

    reached_root = False
    while reached_root == False:
        yield cursor.node
        

        if cursor.goto_first_child():
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                retracing = False



def create_cst(src):
    PY_LANGUAGE = Language('build/my-languages.so', 'python')   ##importa il linguaggio
    parser = Parser()  
    parser.set_language(PY_LANGUAGE) #crea il parser per python
    
    with open(src, 'rb') as srcbyte:
        contents = srcbyte.read()
    tree = parser.parse(contents)

    with open(src, 'r') as srcstr:
        file_array=srcstr.readlines()
    
    for node in traverse_tree(tree):
        start_row=node.start_point[0]
        end_row=node.end_point[0]
        start_column=node.start_point[1]
        end_columns=node.end_point[1]

        print(node)
        print('line',contents[node.start_byte:node.end_byte])

        # Print the spaces before the node
        start_byte = node.start_byte
        while start_byte > 0 and contents[start_byte-1:start_byte] in (b' ', b'\n', b'\t'):
            start_byte -= 1
        spaces_before = contents[start_byte:node.start_byte].decode('utf-8')
        print('spaces before:', repr(spaces_before))

        # Print the spaces after the node
        end_byte = node.end_byte
        while end_byte < len(contents) and contents[end_byte:end_byte+1] in (b' ', b'\n', b'\t'):
            end_byte += 1
        spaces_after = contents[node.end_byte:end_byte].decode('utf-8')
        print('spaces after:', repr(spaces_after))



        # Count the spaces before the node
        start_byte = node.start_byte
        while start_byte > 0 and contents[start_byte-1:start_byte] == b' ':
            start_byte -= 1
        num_spaces_before = node.start_byte - start_byte

        # Count the spaces after the node
        end_byte = node.end_byte
        while end_byte < len(contents) and contents[end_byte:end_byte+1] == b' ':
            end_byte += 1
        num_spaces_after = end_byte - node.end_byte

        print('number of spaces before:', num_spaces_before)
        print('number of spaces after:', num_spaces_after)


       
        # Count the new line before the node
        start_byte = node.start_byte
        while start_byte > 0 and contents[start_byte-1:start_byte] in (b'\n'):
            start_byte -= 1
        num_nw_before = node.start_byte - start_byte

        # Count the new line after the node
        end_byte = node.end_byte
        while end_byte < len(contents) and contents[end_byte:end_byte+1] in (b'\n'):
            end_byte += 1
        num_nw_after = end_byte - node.end_byte

        print('number of new line before:', num_nw_before)
        print('number of new line after:', num_nw_after)



        # Count the tab before the node
        start_byte = node.start_byte
        while start_byte > 0 and contents[start_byte-1:start_byte] in (b'\t'):
            start_byte -= 1
        num_tab_before = node.start_byte - start_byte

        # Count the tab after the node
        end_byte = node.end_byte
        while end_byte < len(contents) and contents[end_byte:end_byte+1] in (b'\t'):
            end_byte += 1
        num_tab_after = end_byte - node.end_byte

        print('number of tab before:', num_tab_before)
        print('number of tab after:', num_tab_after)





        ''''
        i=0
        while str(file_array[start_row][end_columns]) == " ":
            end_columns=end_columns+1
            i=i+1
        print("number white space:" ,i)
        '''

   


create_cst(src_code1)