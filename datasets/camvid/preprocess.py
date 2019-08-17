import os
import shutil
import torch

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np

from PIL import Image


def mkdir(dir_path, dir_name, forced_remove=False):
    new_dir = '{}/{}'.format(dir_path, dir_name)
    if forced_remove and os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)


def touch(file_path, file_name, forced_remove=False):
    new_file = '{}/{}'.format(file_path, file_name)
    assert os.path.isdir(
        file_path), ' \"{}\" does not exist.'.format(file_path)
    if forced_remove and os.path.isfile(new_file):
        os.remove(new_file)
    if not os.path.isfile(new_file):
        open(new_file, 'a').close()


def write_file(file_path, file_name, content, new_line=True, forced_remove_prev=False):
    touch(file_path, file_name, forced_remove=forced_remove_prev)
    with open('{}/{}'.format(file_path, file_name), 'a') as f:
        f.write('{}'.format(content))
        if new_line:
            f.write('\n')
        f.close()


def copy_file(src_path, src_file_name, dst_path, dst_file_name):
    shutil.copyfile('{}/{}'.format(src_path, src_file_name),
                    '{}/{}'.format(dst_path, dst_file_name))


def ls(dir_path):
    return os.listdir(dir_path)


def generate_ann_mask(mask_path, labels):
    mask = np.array(Image.open(mask_path))
    ann = np.zeros((len(labels), mask.shape[0], mask.shape[1]))
    for ix, (label) in enumerate(labels):
        r, g, b, cat = label
        r, g, b = int(r), int(g), int(b)
        label_mask = (mask[:, :, 0] == r) & (
            mask[:, :, 1] == g) & (mask[:, :, 2] == b)
        ann[ix, :, :] = label_mask
    ann = np.argmax(ann, axis=0)
    return torch.tensor(ann)


def preprocess(mode, items_name, camvid_path, annotation_version):
    assert mode in [
        'train', 'val', 'test'], 'preprocess mode must be "train" or "val" or "test".'
    assert annotation_version in [1, 2], 'annotation_version must be 1 or 2.'

    mkdir('.', mode, forced_remove=False)
    mkdir('./{}'.format(mode), 'images', forced_remove=False)
    mkdir('./{}'.format(mode),
          'annotations_v{}'.format(annotation_version), forced_remove=False)

    labels = [line.split() for line in open(
        './labels_v{}.txt'.format(annotation_version))]
    for label in labels:
        (_, _, _, cat_name) = label
        write_file('{}'.format(mode),
                   'labels_v{}'.format(annotation_version),
                   '{}'.format(cat_name),
                   new_line=True,
                   forced_remove_prev=False)

    for ix, (item_name) in enumerate(items_name):
        copy_file('{}/images'.format(camvid_path), item_name,
                  './{}/images'.format(mode), item_name)

        ann = generate_ann_mask(
            '{}/masks/{}'.format(camvid_path, item_name), labels)
        torch.save(ann, './{}/annotations_v{}/{}'.format(mode,
                                                         annotation_version,
                                                         os.path.splitext(item_name)[0]))

        write_file(mode, 'file_names', item_name,
                   new_line=True, forced_remove_prev=False)
        print('%s(version %d): %d/%d( %.2f%%)' % (mode,
                                                  annotation_version,
                                                  ix,
                                                  len(items_name),
                                                  (ix/len(items_name))*100
                                                  ))


def _main():
    camvid_path = '.'
    annotation_versions = [1, 2]

    images_name = ls('{}/images'.format(camvid_path))

    percent = dict()
    percent['train'], percent['val'], percent['test'] = 0.80, 0.05, 0.15
    assert percent['train']+percent['val'] + \
        percent['test'] == 1, '"train percent"+ "val percent"+ "test percent" must be 1.'

    items_name = dict()
    items_name['train'] = images_name[0: int(
        len(images_name) * percent['train'])]
    items_name['val'] = images_name[int(len(images_name) * percent['train']): int(
        len(images_name) * (percent['train'] + percent['val']))]
    items_name['test'] = images_name[int(
        len(images_name) * (percent['train'] + percent['val'])): len(images_name)]

    for annotation_version in annotation_versions:
        for mode in ['train', 'val', 'test']:
            preprocess(mode, items_name[mode], camvid_path, annotation_version)


if __name__ == "__main__":
    _main()
