import os
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

    #folders=['gcjpyred', 'gcjpyredMINI', 'gcjpyredBLACKED'] 
    

    for folder in folders:
        
        print(f"Processing {folder}...")
        origin = os.path.join('datasets','raw_dataset', folder)
        destination = os.path.join('datasets','processed_dataset', folder + "10")

        create_dataset(origin, destination) #crea il dataset come file di testo, qui c'è la lengh limitation
        split_train_test_val(preprocessing, destination)
        create_dict(destination)
        print(f"Completed processing {folder}\n")




    



    
   

    
    
   
