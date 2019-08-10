import gc
import torch

import numpy as np
import torch.nn as nn

from torch.autograd import Variable
from optparse import OptionParser

import os
import shutil
import torch
from optparse import OptionParser


def mkdir(dir_path, dir_name, forced_remove=False):
    """
    Make new directory
    Parameters
    ----------
    dir_path : str
        path to new directory
    dir_name : str
        new directory name
    forced_remove : boolean, optional
        if forced_remove is true and there is a directory in dir_path with dir_name then delete it.
    """
    new_dir = '{}/{}'.format(dir_path, dir_name)
    if forced_remove and os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)


def touch(file_path, file_name, forced_remove=False):
    """
    Make a new empty file
    Parameters
    ----------
    file_path : str
        new file path
    file_name : str
        new file name
    forced_remove : boolean, optional
        if forced_remove is true and there is a file in file_path with file_name then delete it.
    """
    new_file = '{}/{}'.format(file_path, file_name)
    assert os.path.isdir(
        file_path), ' \"{}\" does not exist.'.format(file_path)
    if forced_remove and os.path.isfile(new_file):
        os.remove(new_file)
    if not os.path.isfile(new_file):
        open(new_file, 'a').close()


def write_file(file_path, file_name, content, new_line=True, forced_remove_prev=False):
    """
    write a line to the file
    Parameters
    ----------
    file_path : str
        path to the file that we want to write to it
    file_name : str
        name of file that we want to write to it
    content : str
        content of new line
    new_line : boolean, optional
        adding a new line character to the end of new line or not 
    forced_remove_prev : boolean, optional
        if forced_remove_prev is false then append the new line to the file
    """
    touch(file_path, file_name, forced_remove=forced_remove_prev)
    with open('{}/{}'.format(file_path, file_name), 'a') as f:
        f.write('{}'.format(content))
        if new_line:
            f.write('\n')
        f.close()


def copy_file(src_path, src_file_name, dst_path, dst_file_name):
    """
    copying file
    Parameters
    ----------
    src_path : str
        source file path
    src_file_name : str
        source file name
    dst_path : str
        destination path
    dst_file_name : str
        name of copied file
    """
    shutil.copyfile('{}/{}'.format(src_path, src_file_name),
                    '{}/{}'.format(dst_path, dst_file_name))


def ls(dir_path):
    """
    listing directory contents
    Parameters
    ----------
    dir_path : str
        directory that we want to list its content

    Returns
    -------
    : list of strings
        contents of directory
    """
    return os.listdir(dir_path)


def save_np_file(file_name, np_arr):
    """
    saving a numpy array
    Parameters
    ----------
    file_name : str
        file path and file name that we want to saving a numpy array to it
    np_arr : numpy.array
        numpy array that we want to saving it
    """
    np.save(file_name, np_arr)


def get_args():
    parser = OptionParser()
    parser.add_option('-a', '--analysis', dest='analysis', default=1, type='int',
                      help='analysis number')
    parser.add_option('-v', '--validate', action='store_true', dest='val',
                      default=False, help='validate model')

    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
    parser.add_option('-r', '--learning-rate', dest='lr', default=1e-3,
                      type='float', help='learning rate')
    parser.add_option('-m', '--momentum', dest='momentum', default=0.9,
                      type='float', help='learning rate')
    # parser.add_option('-b', '--batchnorm', action='store_true', dest='batch_norm',
    #                   default=False, help='use batch normalization')

    parser.add_option('-o', '--optimization', dest='optimization', default='adam',
                      type='string', help='optimization method: { adam, sgd }')

    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-w', '--worker', dest='num_workers', default=1,
                      type='int', help='use cuda')

    parser.add_option('-j', '--start-epoch', dest='start_epoch', default=0,
                      type='int', help='starting epoch number')

    parser.add_option('-l', '--load', action='store_true', dest='load',
                      default=False, help='load(read) file model')
    parser.add_option('-s', '--save', dest='save', action='store_true',
                      default=False, help='save file model')

    (options, args) = parser.parse_args()
    return options


def del_tensors(tensors):
    for tensor in tensors:
        del tensor
    gc.collect()
    torch.cuda.empty_cache()
