import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

#data = pd.read_csv('dataredPreproMINI1\\train1.txt', header = None,delimiter = "\t", lineterminator='\n')
splitted_data="datasets\\raw_dataset\\gcjpyredMINI"


def average_file_length(folder_path):
    total_char_length = 0
    total_line_length = 0
    total_files = 0

    # Walk through all the files and subdirectories in the given folder
    for dirpath, dirnames, files in os.walk(folder_path,topdown = False):
        for name in files:

            filepath = os.path.join(dirpath, name)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                total_char_length += len(content)  
                total_line_length += len(content.splitlines())
            total_files += 1



    average_char_length = total_char_length / total_files
    average_line_length = total_line_length / total_files

    print(f"Average length of Python files: {average_char_length:.2f} characters")
    print(f"Average length of Python files: {average_line_length:.2f} lines")


# Test

average_file_length(splitted_data)

