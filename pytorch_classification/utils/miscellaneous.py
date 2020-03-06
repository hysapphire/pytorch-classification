import errno
import os

from .comm import is_main_process


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_config(cfg, path):
    if is_main_process():
        with open(path, "w") as f:
            f.write(cfg.dump())


def save_dict_data(dict_data, path):
    if is_main_process():
        with open(path, "w") as f:
            for k in sorted(dict_data.keys()):
                print(str(k) + ": " + str(dict_data[k]), file=f)


def print_dict_data(dict_data):
    if is_main_process():
        for k in sorted(dict_data.keys()):
            print(str(k) + ": " + str(dict_data[k]))
