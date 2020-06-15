import pandas as pd
from glob import glob as globlin ## The 7bb globlin
import wfdb

def get_all_paths(main_path, sub_path):
    file_paths = globlin(main_path + '/*.' + sub_path)
    return file_paths



if __name__ == '__main__':
    mit_db_paths = get_all_paths('../Data_thesis/MIT', 'dat')
    cudb_paths = get_all_paths('../Data_thesis/CUDB', 'dat')
    vfdb_paths = get_all_paths('../Data_thesis/vfdb', 'dat')

    

