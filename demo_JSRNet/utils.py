###########################################
# Written by Haimiao Zhang
# Coauthors: Bin Dong, Baodong Liu
# Copyright preserved
# Contact: hmzhang@bistu.edu.cn
###########################################

import tensorflow as tf
import odl
import numpy as np
from glob import glob

def prelu(_x, init=0.0, name='prelu', trainable=True):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alphas', shape=[int(_x.get_shape()[-1])], initializer=tf.constant_initializer(init), dtype=tf.float32, trainable=True)
        pos = tf.nn.relu(_x)
        neg = -alphas * tf.nn.relu(-_x)
    return pos + neg

def conv(x, filters=9):
    conv2 = tf.keras.layers.Conv2D(filters=filters,kernel_size=3,strides=(1,1),padding='same',kernel_initializer=tf.contrib.layers.xavier_initializer())
    return conv2(x)

def net_block(image,name):
    l=image
    with tf.variable_scope(name+'block1') as scope:
        for i in range(5):
            l = add_layer('dense_layer.{}'.format(i), l, map_num=12)

    with tf.variable_scope(name+'output') as scope:
        output = conv(l,filters=1)
    return output

def add_layer(name, img, map_num=36):
    with tf.variable_scope(name) as scope:
        output = prelu(conv(img,filters=map_num), name='prelu_'+name)
        img = tf.concat([output, img], 3)
    return img

def para_lam(batch_input,name):
    with tf.variable_scope(name) as scope:
        in_channels = batch_input[...,-1:].get_shape()[3]
        filter = tf.get_variable("filter", [1, 1, in_channels, 1], dtype=tf.float32, initializer=tf.random_normal_initializer(-0.1, 0.0))
        conv1 = tf.nn.conv2d(batch_input[...,-2:-1], filter, [1, 1, 1, 1], padding="VALID")
        conv2 = tf.nn.conv2d(batch_input[...,-1:], 1.0-filter, [1, 1, 1, 1], padding="VALID")
        return conv1+conv2

def radon_op(img_size=[512,512], sino_size=[256, 1024]):
    xx=256.5
    space = odl.uniform_discr([-xx, -xx], [xx, xx], [img_size[0], img_size[0]],dtype='float32')
    angles=np.array(sino_size[0]).astype(int)
    angle_partition = odl.uniform_partition(0, 2 * np.pi, angles)
    detectors=np.array(sino_size[1]).astype(int)
    detector_partition = odl.uniform_partition(-521.0624,521.0624, detectors)
    src_radius=234.0262/np.sin(10.1*np.pi/180.0)
    geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,src_radius=src_radius, det_radius=src_radius)
    RT = odl.tomo.RayTransform(space, geometry,impl='astra_cuda')
    op_norm=odl.operator.power_method_opnorm(RT)
    op_norm=np.array(np.sqrt(2.0)*op_norm*2*np.pi**2)
    fbp = odl.tomo.fbp_op(RT,filter_type='Ram-Lak',frequency_scaling=0.9)
    radon_layer = odl.contrib.tensorflow.as_tensorflow_layer(RT, 'RayTransform')
    iradon_layer = odl.contrib.tensorflow.as_tensorflow_layer(RT.adjoint,'RayTransformAdjoint')
    return RT, space, radon_layer, iradon_layer, fbp, op_norm

def gen_mask(RT):
    mask=np.ones((1024, 256))
    for i in range(128):
        mask[:,i*2+1]=0.0

    sino_mask_np=mask.transpose()
    sino_mask = np.empty((1, RT.range.shape[0], RT.range.shape[1], 1), dtype='float32')
    sino_mask[0,...,0] = sino_mask_np
    return sino_mask

def disk():
    imx, imy=512, 512
    mask=np.zeros([imx,imy])
    for i in range(imx):
        for j in range(imy):
            if (i-imx//2)**2+(j-imy//2)**2<=230**2:
                mask[i,j]=1.0
    return mask

def load_data(RT, space):
    # Load data
    vl_sino_arr = np.empty((10, RT.range.shape[0], RT.range.shape[1], 1), dtype='float32')
    vl_phan_arr = np.empty((10, space.shape[0], space.shape[1], 1), dtype='float32')
    vl_fbpu_arr = np.empty((10, space.shape[0], space.shape[1], 1), dtype='float32')

    base_dir='./'
    vl_sino_dir = base_dir+'sino_data.npy' # sinogram
    vl_phan_dir = base_dir+'phan_data.npy' # phantom
    vl_fbpu_dir = base_dir+'fbpu_data.npy' # FBPu

    vl_sino_dir = sorted(glob(vl_sino_dir))
    vl_phan_dir = sorted(glob(vl_phan_dir))
    vl_fbpu_dir = sorted(glob(vl_fbpu_dir))

    vl_sino_tmp=np.empty((10, RT.range.shape[0], RT.range.shape[1], 1), dtype='float32')
    vl_sino_tmp[0:10,::2,:,0] = np.load(vl_sino_dir[0])
    for i in range(10):
        vl_sino_arr[i,...,0]= RT.range.element(vl_sino_tmp[i,...,0]*np.array(28.46))

    vl_phan_tmp=np.empty((10, space.shape[0], space.shape[1], 1), dtype='float32')
    vl_phan_tmp[...,0] = np.load(vl_phan_dir[0])
    for i in range(10):
        vl_phan_arr[i,...,0]= space.element(np.rot90(vl_phan_tmp[i,...,0],k=-1,axes=(0,1)))

    vl_fbpu_tmp=np.empty((10, space.shape[0], space.shape[1], 1), dtype='float32')
    vl_fbpu_tmp[...,0] = np.load(vl_fbpu_dir[0])
    for i in range(10):
        vl_fbpu_arr[i,...,0]= space.element(np.rot90(vl_fbpu_tmp[i,...,0],k=-1,axes=(0,1)))

    return vl_sino_arr, vl_phan_arr, vl_fbpu_arr
