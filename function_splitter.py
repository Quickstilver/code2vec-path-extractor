# !/usr/bin/python3
import os
from regex import D
from tree_sitter import Language, Parser


def snippler(origin,destination):
  
  PY_LANGUAGE = Language('build/my-languages.so', 'python')
  parser = Parser()
  parser.set_language(PY_LANGUAGE)
  curr=os.getcwd()
  destination= os.path.join(curr, destination)
  origin= os.path.join(curr, origin)
  os.chdir(origin) 

  for root, dirs, files in os.walk(".", topdown = False):

    root= str(root[2:])

    for name in files:
        doc= os.path.join(root, name)

        with open(doc, 'rb') as f:
            contents = f.read()
            tree = parser.parse(contents)
            rootTree= tree.root_node
            names = name.replace('.py', '')    

            for i, child in enumerate(rootTree.children):
                new_file = f"{names}_{i}.py"
                new_path = os.path.join(destination,root,names,new_file)
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                to_write = contents[child.start_byte:child.end_byte].decode('utf8')

                with open(new_path, "w+", encoding="utf-8") as f:
                    f.write(to_write)

  os.chdir(os.getcwd())
  print(os.getcwd())
  print("dataset splitted!")



#snippler("gcjpyMINICanc", "gcjpyredMINISPLITcance")


