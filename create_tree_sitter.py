from tree_sitter import Language, Parser

def build_tree_sitter():
  Language.build_library(
    # Store the library in the `build` directory
    'build/my-languages.so',

    # Include one or more languages
    [
      'tree-sitter-python'
    ]
  )
  return 
