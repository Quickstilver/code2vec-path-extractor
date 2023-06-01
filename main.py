import os
import random
import shutil
import subprocess
from create_tree_sitter import build_tree_sitter
from function_splitter import snippler
from path_cst_extractor import create_dataset
from Line_Of_Dataset import create_dict,split_train_test_val

if __name__ == '__main__':
   
    ###SET OVERSAMPLING
    preprocessing= True #set oversampling On or Off    

    #dataredPreproBLACKED:354 numero example>200 1976
    #dataredPreproMINI: 354 numero example>200 1674
    #dataredPrepro:354 numero example>200 1976
    
    #dataredPreproMINI n_examples: 1888
    #dataredPreproBLACKED n_examples: 1593
    #dataredPrepro n_examples: 1593


    build_tree_sitter()

    folders=['gcjpyred', 'gcjpyredMINI', 'gcjpyredBLACKED'] 
    

    # get a list of all subdirectories in the first folder
    first_folder_path = os.path.join('datasets','raw_dataset', folders[0])
    all_subdirs = [d for d in os.listdir(first_folder_path) if os.path.isdir(os.path.join(first_folder_path, d))]

    # randomly select 10 of these subdirectories
    random.seed(1)  # use a seed for reproducibility
    selected_subdirs = random.sample(all_subdirs, 10)

    for folder in folders:
        new_folder = os.path.join('datasets', 'random_subdirs', folder)
        os.makedirs(new_folder, exist_ok=True)

        # copy the selected subdirectories into the new folder
        for subdir in selected_subdirs:
            src_path = os.path.join('datasets','raw_dataset', folder, subdir)
            dest_path = os.path.join(new_folder, subdir)
            if not os.path.exists(dest_path):
                shutil.copytree(src_path, dest_path)

    #creation pipeline
    for folder in folders:

        print(f"Processing {folder}...")
        origin = os.path.join('datasets','random_subdirs', folder)
        destination = os.path.join('datasets','processed_dataset', folder + "10Prepro")

        create_dataset(origin, destination) 
        split_train_test_val(preprocessing, destination)
        create_dict(destination)
        print(f"Completed processing {folder}\n")