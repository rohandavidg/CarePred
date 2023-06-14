import random
import os
import numpy as np
import pandas as pd
import torch



def save_flags(FLAGS):
    with open(FLAGS.outroot + "/results/" + FLAGS.folder + '/flags.cfg','w') as f:
        for arg in vars(FLAGS):
            f.write('--%s=%s\n'%(arg, getattr(FLAGS, arg)))


def mkdir(directory):
    directories = directory.split("/")   

    folder = ""
    for d in directories:
        folder += d + '/'
        if not os.path.exists(folder):
            print('creating folder: %s'%folder)
            os.mkdir(folder)

def remove_files_in_dir(directory):
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            
def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def cat_files(output_file, folder_path):
    remove_file(output_file)
    with open(output_file, 'a') as output:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    shutil.copyfileobj(file, output)
    return output_file


