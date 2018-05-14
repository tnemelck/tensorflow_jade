#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 23:12:34 2018

@author: elvex
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from functools import partial
from glob import glob
from scipy.io import loadmat

#%%
dic = {"Bois" : 0,
       "PVC" : 1,
       "Joint" : 2,
       "Verre" : 3,
       "PA" : 4,
       "PE" : 4,
       "PC" : 4,
       "Autres" : 5,
       "Cailloux" : 5}


#%%

def init_var():
    with open(".config_variables", 'r') as f:
        R = f.read().splitlines() 
        R = list(map(lambda x : x.split(': '), R))
        train_dir = R[0][-1]
        image_label_adr = R[1][-1]
        image_train_adr = R[2][-1]
        image_validation_adr = R[3][-1]
        image_test_adr = R[4][-1]
        label_train_adr = R[5][-1]
        label_validation_adr = R[6][-1]
        label_test_adr = R[7][-1]
        IMG_W = int(R[8][-1])
        IMG_H = int(R[9][-1])
        NB_CLASSES = int(R[10][-1])
        LR = float(R[11][-1])
        log_dir = R[12][-1]
        nb_img_max = int(R[13][-1])
        batch_size = int(R[14][-1])
        step_iter = int(R[15][-1])
        
        return (train_dir, image_label_adr, 
                image_train_adr, image_validation_adr, image_test_adr,
                label_train_adr, label_validation_adr, label_test_adr,
                IMG_W, IMG_H, NB_CLASSES, LR, log_dir, nb_img_max,
                batch_size, step_iter)
        
        

#%%

# you need to change this to your data directory
        
(train_dir, image_label_adr, 
                image_train_adr, image_validation_adr, image_test_adr,
                label_train_adr, label_validation_adr, label_test_adr,
                IMG_W, IMG_H, NB_CLASSES, LR, log_dir, nb_img_max,
                batch_size, step_iter) = init_var()
        
#%%


def is_file_img_lbl(overide):
    return (not (os.path.exists(image_label_adr))) or (overide)

def is_files_sample(overide):
    result = (overide or not 
        (os.path.exists(image_test_adr))
        or (os.path.exists(image_train_adr))
        or (os.path.exists(image_validation_adr))
        or (os.path.exists(label_test_adr))
        or (os.path.exists(label_train_adr))
        or (os.path.exists(label_validation_adr)))
    return result


def get_files(overide = False):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    if is_file_img_lbl(overide):
        
        lst_adr = glob("/utilisateurs/SHARED_DATA/Img_datasets/VEKA/Imagette_v3/*.mat")
        lst_lbl = list(map (lambda x : dic.get(x.split("_")[0], 5), lst_adr))
        
        print("bois : {}\nPVC : {}\nJoint : {}\nVerre : {}\nPA/PE/PC : {}\nAutres : {}".format(lst_lbl.count(0),
              lst_lbl.count(1), lst_lbl.count(2), lst_lbl.count(3), lst_lbl.count(4)))
        
        ary = np.vstack((lst_adr, lst_lbl))
        
        np.save(image_label_adr, ary)
        
        #np.random.shuffle(temp)
        #image_list = temp[:, 0]
        #label_list = temp[:, 1].astype("int32")
        
        #image_list.save(image_adr)
        #label_list.save(label_adr)
    
    
    return None

#%%
   
def get_sample(overide = True):
    get_files(False)
    
    if is_files_sample(overide):
        temp = np.load(image_label_adr)
        
        np.random.shuffle(temp)
        image_list = temp[:, 0]
        label_list = temp[:, 1].astype("int32")
        
        L = label_list.shape[0]
        
        e0 = 0
        e1 = int(L * 0.6)
        e2 = e1 + int(L * 0.2)
        e3 = L
        
        img_train, lbl_train = image_list[e0:e1][:nb_img_max], label_list[e0:e1][:nb_img_max]
        img_vldtn, lbl_vldtn = image_list[e1:e2][:nb_img_max], label_list[e1:e2][:nb_img_max]
        img_test, lbl_test = image_list[e2:e3][:nb_img_max], label_list[e2:e3][:nb_img_max]
        
        np.save(image_train_adr, img_train)
        np.save(image_validation_adr, img_vldtn)
        np.save(image_test_adr, img_test)
        
        np.save(label_train_adr, lbl_train)
        np.save(label_validation_adr, lbl_vldtn)
        np.save(label_test_adr, lbl_test)
        
    return None
    
#%%

def _parse_function(filename, label, crop = False):
    name = (filename.split(".")[0]).split("/")[-1]
    mtx = loadmat(filename)[name]
    image = tf.convert_to_tensor(mtx)
    if crop: image = tf.image.resize_image_with_crop_or_pad(image, IMG_W, IMG_H)
    image = tf.image.resize_images(image, [IMG_W, IMG_H])
    image = tf.cast(image, tf.float16)
    return image, label

#%%

def get_tensor_train(crop = True, overide = False):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    
    get_sample(overide)
    img = (np.load(image_train_adr))
    lbl = (np.load(label_train_adr))
    image_adr = tf.constant(img)
    label = tf.constant(lbl)
    dataset = tf.data.Dataset.from_tensor_slices((image_adr, label))
    fonction = partial(_parse_function, crop = crop)
    dataset = dataset.map(fonction)
    batch = dataset.batch(len(img))
    iterator = batch.make_one_shot_iterator()
    image_final, label_final = iterator.get_next()
    
    return image_final, label_final


#%%

def get_tensor_validate(crop = True, overide = False):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    
    get_sample(overide)
    img = (np.load(image_validation_adr))
    lbl = (np.load(label_validation_adr))
    image_adr = tf.constant(img)
    label = tf.constant(lbl)
    dataset = tf.data.Dataset.from_tensor_slices((image_adr, label))
    fonction = partial(_parse_function, crop = crop)
    dataset = dataset.map(fonction)
    batch = dataset.batch(len(img))
    iterator = batch.make_one_shot_iterator()
    image_final, label_final = iterator.get_next()
    
    return image_final, label_final


#%%

def get_tensor_test(crop = True, overide = False):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    
    get_sample(overide)
    img = (np.load(image_test_adr))
    lbl = (np.load(label_test_adr))
    image_adr = tf.constant(img)
    label = tf.constant(lbl)
    dataset = tf.data.Dataset.from_tensor_slices((image_adr, label))
    fonction = partial(_parse_function, crop = crop)
    dataset = dataset.map(fonction)
    batch = dataset.batch(len(img))
    iterator = batch.make_one_shot_iterator()
    image_final, label_final = iterator.get_next()
    
    return image_final, label_final


 
#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes



def test():
    
    BATCH_SIZE = 5
    crop = True
    
    global train_dir
    
    image_batch, label_batch = get_tensor_train(overide = True, crop = crop)
    img_int = tf.cast(image_batch, tf.uint8)
    
    with tf.Session() as sess:
        
        img, label = sess.run([img_int, label_batch])
        # just test one batch
        for j in np.arange(BATCH_SIZE):
            print('label: %d' %label[j])
            plt.imshow(img[j,:,:,:])
            plt.show()
                
