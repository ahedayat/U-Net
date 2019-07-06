import os
import glob
import shutil
import random 
import gc

import numpy as np
import torchvision.transforms as transforms

from shutil import copyfile
from PIL import Image

data_partition_percentage = {
    'train' : 0.85,
    'val' : 0.05
    # ,'test' : 0.1
}

def get_classes(label_file_path):
    with open(label_file_path) as f:
        labels = f.readlines()
    labels = [ label.split() for label in labels ]
    colors = list()
    color2id = dict()
    for ix,label in enumerate(labels):
        r, g, b, class_name = label
        labels[ix] = class_name
        colors.append( ( int(r), int(g), int(b) ) )
        color2id[(int(r),int(g),int(b))] = ix
    return labels, colors, color2id

def get_file_names(file_path):
    file_names = glob.glob( file_path + '/*.png') 
    file_names = [ os.path.split(file_name)[1] for file_name in file_names ]
    random.shuffle( file_names )
    random.shuffle( file_names )
    return file_names

def generate_data_file_path(path, version, delete_if_exist=True):
    if delete_if_exist and os.path.isdir( path ):
        shutil.rmtree(path)
        os.mkdir(path)
        os.mkdir(path+'/images')
        os.mkdir(path+'/processed_images')
        os.mkdir(path+'/mirrored_images')
        os.mkdir(path+'/annotations')
    os.mkdir(path+'/annotations_v{}'.format(version))

def get_ann_file( void_index, color2id, colored_ann ):
    row, column, channels = colored_ann.shape
    ann = np.zeros((row,column)) 

    all_colors = color2id.keys() 

    for r in range(row):
        for c in range(column):
            rr,gg,bb = int(colored_ann[r,c,0]), int(colored_ann[r,c,1]), int(colored_ann[r,c,2])
            if (rr,gg,bb) in all_colors:
                ann[r,c] = int(color2id[(rr,gg,bb)])
    return ann

def input_filled_mirroring(img, e = 92):      # fill missing data by mirroring the input image
    '''input size 388 --> output size 572'''
    # w, h = x.shape
    x = np.array(img)
    w, h, c = np.shape(x)[0], np.shape(x)[1], np.shape(x)[2]
    y = np.zeros((h + e * 2, w + e * 2,c), dtype=np.uint8)
    #e = 62  # extra width on 1 edge
    for channel in range(c):
        y[e:h + e, e:w + e,channel] = x[:,:,channel]
        y[e:e + h, 0:e,channel] = np.flip(y[e:e + h, e:2 * e,channel], 1)  # flip vertically
        y[e:e + h, e + w:2 * e + w,channel] = np.flip(y[e:e + h, w:e + w,channel], 1)  # flip vertically
        y[0:e, 0:2 * e + w,channel] = np.flip(y[e:2 * e, 0:2 * e + w,channel], 0)  # flip horizontally
        y[e + h:2 * e + h, 0:2 * e + w,channel] = np.flip(y[h:e + h, 0:2 * e + w,channel], 0)  # flip horizontally
    y=Image.fromarray(np.uint8(y))
    return y

def generate_data_partition(void_index, color2id, data_path, ann_path, partition_path, file_names, start_index, end_index, version, copy_image=True, copy_ann=False, input_image_size=(572,572),output_image_size=(388,388)):
    generate_data_file_path(partition_path, version, delete_if_exist=copy_image)
    partition_file_names = list()
    counter = 0
    length = end_index - start_index
    in_w, in_h = input_image_size
    out_w, out_h = output_image_size

    img_scale = transforms.Resize(input_image_size)
    img_scale_mirrored = transforms.Resize(output_image_size)

    ann_scale = transforms.Resize(output_image_size)
    for ix in range(start_index, end_index):
        partition_file_names.append( file_names[ix] )
        file_name, file_extension = os.path.splitext( file_names[ix] )
        #copy image
        if copy_image:
            copyfile( data_path+'/'+file_name+file_extension, partition_path+'/images/'+file_name+file_extension )
            image = Image.open( data_path+'/'+file_name+file_extension )
            image_scaled = img_scale(image)
            image_scaled.save( partition_path+'/processed_images/'+file_name+file_extension )
            
            
            image_scaled_mirrored = img_scale_mirrored(image)
            image_scaled_mirrored = input_filled_mirroring(image_scaled_mirrored, e = (in_w - out_w)//2)
            image_scaled_mirrored.save( partition_path+'/mirrored_images/'+file_name+file_extension )

            del image, image_scaled, image_scaled_mirrored
        
        #copy annotation
        if not copy_ann:
            colored_ann = np.array( Image.open( ann_path+'/'+file_name+'_L'+file_extension ) )
            ann = get_ann_file( void_index, color2id, colored_ann )
            ann_img = Image.fromarray(ann.astype(np.uint8))
            ann_img = ann_scale(ann_img)
            ann_img.save(partition_path+'/annotations_v{}/'.format(version)+file_name+file_extension)
            del colored_ann, ann, ann_img
        else:
            copyfile( ann_path+'/'+file_name+'_L'+file_extension, partition_path+'/annotations/'+file_name+file_extension )
            ann_img = Image.open( ann_path+'/'+file_name+'_L'+file_extension )
            ann_img = ann_scale( ann_img )
            ann_img.save(partition_path+'/annotations_v{}/'.format(version)+file_name+file_extension)

            del ann_img
        counter += 1
        print('%s : %d/%d ( %.2f %% )' % (partition_path, counter, length, (counter/length)*100), end='\r')
            
        del file_name, file_extension
        gc.collect()

    with open(partition_path+'/file_names.txt', 'w') as f:
        for file_name in partition_file_names:
            f.write("{}\n".format(file_name))
    print('')

def _main():
    data_path = './git/701_StillsRaw_full'
    ann_path = './git/LabeledApproved_full'
    partition_path = '.'
    labels_file_path_v1 = './labels_v1.txt'
    labels_file_path_v2 = './labels_v2.txt'

    file_names = get_file_names(data_path)
    labels_v1, colors_v1, color2id_v1 = get_classes(labels_file_path_v1)
    labels_v2, colors_v2, color2id_v2 = get_classes(labels_file_path_v2)
    num_data = len(file_names)

    data_partition = {
    'train' : int( num_data * data_partition_percentage['train'] ),
    'val' : int( num_data * data_partition_percentage['val'] ),
    # 'test' : num_data - int( num_data * data_partition_percentage['train'] ) - int( num_data * data_partition_percentage['val'] )
    }

    start_indexes = [0,data_partition['train'], data_partition['train']+data_partition['val']]
    void_index_v1 = labels_v1.index('Void')
    void_index_v2 = labels_v2.index('Void')

    print('annotations (v0) :')
    generate_data_partition( None, None, data_path, ann_path, './test', file_names, start_indexes[2], num_data, 0, copy_image = True, copy_ann=True)
    generate_data_partition( None, None, data_path, ann_path, './train', file_names, start_indexes[0], start_indexes[1], 0, copy_image = True, copy_ann=True)
    generate_data_partition( None, None, data_path, ann_path, './val', file_names, start_indexes[1], start_indexes[2], 0, copy_image = True, copy_ann=True)

    print('annotations (v1) :')    
    generate_data_partition( void_index_v1, color2id_v1, data_path, ann_path, './test', file_names, start_indexes[2], num_data, 1, copy_image = False, copy_ann=False)
    generate_data_partition( void_index_v1, color2id_v1, data_path, ann_path, './train', file_names, start_indexes[0], start_indexes[1], 1, copy_image = False, copy_ann=False)
    generate_data_partition( void_index_v1, color2id_v1, data_path, ann_path, './val', file_names, start_indexes[1], start_indexes[2], 1, copy_image = False, copy_ann=False)

    print('annotations (v2) :')
    generate_data_partition( void_index_v2, color2id_v2, data_path, ann_path, './test', file_names, start_indexes[2], num_data, 2, copy_image = False, copy_ann=False)
    generate_data_partition( void_index_v2, color2id_v2, data_path, ann_path, './train', file_names, start_indexes[0], start_indexes[1], 2, copy_image = False, copy_ann=False)
    generate_data_partition( void_index_v2, color2id_v2, data_path, ann_path, './val', file_names, start_indexes[1], start_indexes[2], 2, copy_image = False, copy_ann=False)



if __name__ == '__main__':
    _main()
    
    # print(labels)
