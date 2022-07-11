#-*- coding: UTF-8 -*- 
###########################################
# Written by Haimiao Zhang
# Coauthors: Bin Dong, Baodong Liu
# Copyright preserved
# Contact: hmzhang@bistu.edu.cn
###########################################

import tensorflow as tf
import numpy as np
import odl.contrib.tensorflow
from utils import conv, net_block, para_lam, prelu, radon_op, gen_mask, disk, load_data

sess = tf.InteractiveSession()
RT, space, radon, iradon, fbp, op_norm = radon_op()

with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, shape=[None, 512, 512, 1], name="x_true")
    x_fbp = tf.placeholder(tf.float32, shape=[None, 512, 512, 1], name="x_fbp")
    y_rt = tf.placeholder(tf.float32, shape=[None, RT.range.shape[0], RT.range.shape[1], 1], name="y_rt")
    y_mask = tf.placeholder(tf.float32, shape=[None, RT.range.shape[0], RT.range.shape[1], 1], name="y_mask")
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):
        primal = tf.concat([x_fbp] , axis=-1)
        dual = tf.concat([radon(x_fbp)/op_norm] , axis=-1)
        primal_values =tf.concat([tf.zeros_like(x_true),x_fbp] , axis=-1)
        dual_values = tf.concat([tf.zeros_like(y_rt),radon(x_fbp)/op_norm] , axis=-1)
    
    with tf.name_scope('initial_b1'):
        b1=0.0
        b2=0.0

    GaCY=y_mask*y_rt
    PtGaCY = iradon(GaCY)
    for i in range(7):
        with tf.variable_scope('d1_iterate_{}'.format(i)):
            W1f=conv(dual)
            update = prelu(conv(W1f+b1), name='prelu_1')
            d1=conv(update)

        with tf.variable_scope('d2_iterate_{}'.format(i)):
            W2u=conv(primal)
            update = prelu(conv(W2u+b2), name='prelu_1')
            d2=conv(update)
            b1=b1+W1f-d1
            b2=b2+W2u-d2

        with tf.variable_scope('f_iterate_{}'.format(i)):
            WtF_db=conv(d1-b1)
            evalop = radon(primal)
            update_f = tf.concat([(1.0-y_mask)*evalop/op_norm, GaCY/op_norm, WtF_db], axis=-1)
            update = net_block(update_f,name='update_f')
            update_f = para_lam(dual_values, name='LM_para_f_{}'.format(i))
            dual = update_f+update

        with tf.variable_scope('u_iterate_{}'.format(i)):
            WtU_db=conv(d2-b2)
            PtGaF = iradon(dual*(1.0-y_mask))
            update_u = tf.concat([PtGaF/op_norm, PtGaCY/op_norm, WtU_db], axis=-1)
            update = net_block(update_u, name='update_u')
            update_u = para_lam(primal_values, name='LM_para_u_{}'.format(i))
            primal = update_u+update

        primal_values=tf.concat([primal_values,primal], axis=-1)
        dual_values=tf.concat([dual_values,dual], axis=-1)

    x_result = primal

saver = tf.train.Saver()
saver.restore(sess,'weights/ckp')

print('load test data')
vl_sino, vl_phan, vl_fbpu=load_data(RT, space)
sino_mask = gen_mask(RT)

print('\n --------Test Phase----------------')
x_rec=np.zeros_like(vl_phan)
for ijk in range(len(vl_sino)):
    x_rec_tmp=sess.run(x_result,
              feed_dict={x_true: vl_phan[ijk:(ijk+1),...,0:1],
                         y_rt:   vl_sino[ijk:(ijk+1),...,0:1],
                         x_fbp:  vl_fbpu[ijk:(ijk+1),...,0:1],
                         y_mask: sino_mask,
                         is_training: False})
    x_rec[ijk:(ijk+1),...,0:1]=x_rec_tmp

def nrmse_hm(img1,img2):
    nim, nx, ny = img1.shape
    diffsquared = (img1-img2)**2
    num_pix = float(nx*ny)
    meanrmse  = np.sqrt( diffsquared.sum(axis=1).sum(axis=1)/num_pix).mean()
    return meanrmse

disk_mask_tmp=disk()
disk_mask=np.empty((1, space.shape[0], space.shape[1]), dtype='float32')
disk_mask[0,...]=disk_mask_tmp

recu_arr= np.rot90(x_rec[...,0],k=1,axes=(1,2))*disk_mask
phantom_arr= np.rot90(vl_phan[...,0],k=1,axes=(1,2))*disk_mask
recu_arr=np.clip(recu_arr,a_min=0.0,a_max=1.0)
NRMSE=nrmse_hm(recu_arr, phantom_arr)
print("\nTest result, RNMSE={:.8f}".format(NRMSE))

# np.save(report_path+'/predictions.npy', recu_arr)
print('*** Done!!!')
