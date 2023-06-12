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
