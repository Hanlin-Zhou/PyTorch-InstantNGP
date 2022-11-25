import torch
from torch import cuda
import logging
import os
from datetime import datetime
import shutil


def get_device():
    if cuda.is_available():
        print("Training on CUDA")
        return torch.device('cuda')
    else:
        print("Training on CPU")
        return torch.device('cpu')


def logging_setup(path):
    logging.basicConfig(level=logging.INFO, filename=path + '/trainer.log',
                        filemode='w', format='%(asctime)s|%(levelname)s| %(message)s')


def get_time_str():
    now = datetime.now()
    return now.strftime("%m_%d_%Y__%H_%M_%S")


def setup_result_output(args):
    img_name = args.source_directory.split("/")[-1].split(".")[0]
    new_result_dir_name = img_name + '_' + get_time_str()
    new_result_dir = args.result_directory + new_result_dir_name
    os.mkdir(new_result_dir)
    args.result_directory = new_result_dir + '/'
    shutil.copyfile(args.config_path, args.result_directory + "result_config.yaml")
