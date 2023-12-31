# -*- coding: utf-8 -*-

# USAGE
# python Keras2TFy -w weights_file.hdf52

import os
import sys
import shutil

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json, load_model

from config import fcn_config as cfg
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
import tensorflow as tf
from qkeras.utils import _add_supported_quantized_objects

co = {}
_add_supported_quantized_objects(co)
co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
##################################################################################

import argparse #DB
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m",  "--model", help="CNN Models: fcn8, fcn8ups, unet1, unet2, unet3")

args = vars(ap.parse_args())

model_name = args["model"]

##############################################
# Set up directories
##############################################

KERAS_MODEL_DIR = cfg.KERAS_MODEL_DIR #DB

WEIGHTS_DIR = KERAS_MODEL_DIR

CHKPT_MODEL_DIR = cfg.CHKPT_MODEL_DIR


# set learning phase for no training: This line must be executed before loading Keras model
K.set_learning_phase(0)

# load weights & architecture into new model
if model_name=="fcn8ups" :
        weights= "fcn8ups/qmodel_final.h5"
elif model_name=="fcn8" :
        weights= "fcn8/ep200_trained_fcn8_224x224.hdf5"
elif model_name=="unet1" :
        weights= "unet/ep200_trained_unet_model1_224x224.hdf5"
elif model_name=="unet2" :
        weights= "unet/ep200_trained_unet_model2_224x224.hdf5"
else: # elif model_name=="unet3" :
        weights= "unet/ep200_trained_unet_model3_224x224.hdf5"

print("model name = ", model_name)
#weights="fcn8ups/ep2_trained_fcn8ups_224x224.hdf5"
filename = os.path.join(WEIGHTS_DIR,weights)

assert os.path.isdir(WEIGHTS_DIR)
assert os.path.isfile(filename)
model = tf.keras.models.load_model(filename, custom_objects=co)

##print the CNN structure
#model.summary()

# make list of output node names
output_names=[out.op.name for out in model.outputs]

# set up tensorflow saver object
saver = tf.compat.v1.train.Saver()

# fetch the tensorflow session using the Keras backend
sess = tf.compat.v1.keras.backend.get_session()

# Check the input and output name
print ("\n TF input node name:")
print(model.inputs)
print ("\n TF output node name:")
print(model.outputs)

# write out tensorflow checkpoint & meta graph
if model_name=="fcn8ups" :
	save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "fcn8ups/float_model.ckpt"))
elif model_name=="fcn8" :
	save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "fcn8/float_model.ckpt"))
elif model_name=="unet1" :
	save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "unet1/float_model.ckpt"))
elif model_name=="unet2" :
	save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "unet2/float_model.ckpt"))
else: # elif model_name=="unet3" :
	save_path = saver.save(sess, os.path.join(CHKPT_MODEL_DIR, "unet3/float_model.ckpt"))

print ("\nFINISHED CREATING TF FILES\n")
