# Change dataloader multiprocess start method to anything not fork
import open3d as o3d
import numpy as np
import os
import sys
import json
import random
import logging
from easydict import EasyDict as edict

# Torch packages
import torch
from torch.serialization import default_restore_location
# Train deps
from config import get_config

from lib.test_mean import test
from lib.utils import load_state_with_same_shape, get_torch_device, count_parameters
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset

from models import load_model, load_wrapper


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

config = get_config()

if config.is_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found")
# device = get_torch_device(config.is_cuda)

device = config.device_id

logging.info('===> Configurations')
dconfig = vars(config)
#for k in dconfig:
#    logging.info('    {}: {}'.format(k, dconfig[k]))

DatasetClass = load_dataset(config.dataset)
testDatasetClass = load_dataset(config.testdataset)
if config.test_original_pointcloud:
    if not DatasetClass.IS_FULL_POINTCLOUD_EVAL:
        raise ValueError(
            'This dataset does not support full pointcloud evaluation.')

if config.evaluate_original_pointcloud:
    if not config.return_transformation:
        raise ValueError(
            'Pointcloud evaluation requires config.return_transformation=true.')

if (config.return_transformation ^ config.evaluate_original_pointcloud):
    raise ValueError('Rotation evaluation requires config.evaluate_original_pointcloud=true and '
                        'config.return_transformation=true.')

logging.info('===> Initializing dataloader')
if config.is_train:
    val_data_loader = initialize_data_loader(
        testDatasetClass,
        config,
        num_workers=config.num_val_workers,
        phase=config.val_phase,
        augment_data=False,
        shuffle=True,
        repeat=False,
        batch_size=config.val_batch_size,
        limit_numpoints=False)
    if val_data_loader.dataset.NUM_IN_CHANNEL is not None:
        num_in_channel = val_data_loader.dataset.NUM_IN_CHANNEL
    else:
        num_in_channel = 3  # RGB color

    num_labels = val_data_loader.dataset.NUM_LABELS


logging.info('===> Building model')
NetClass = load_model(config.model)
if config.wrapper_type == 'None':
    model1 = NetClass(num_in_channel, num_labels, config)
    model2 = NetClass(num_in_channel, num_labels, config)
    model3 = NetClass(num_in_channel, num_labels, config)
    logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__,
                                                                          count_parameters(model1)))
    logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__,
                                                                          count_parameters(model2)))
    logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__,
                                                                          count_parameters(model3)))
else:
    wrapper = load_wrapper(config.wrapper_type)
    model = wrapper(NetClass, num_in_channel, num_labels, config)
    logging.info('===> Number of trainable parameters: {}: {}'.format(
        wrapper.__name__ + NetClass.__name__, count_parameters(model)))

#logging.info(model1)
#logging.info(model2)
    # model = model.to(device)

if config.weights == 'modelzoo':  # Load modelzoo weights if possible.
    logging.info('===> Loading modelzoo weights')
    model1.preload_modelzoo()

# Load weights if specified by the parameter.
elif config.weights1.lower() != 'none':
    logging.info('===> Loading weights: ' + config.weights1)
    logging.info('===> Loading weights: ' + config.weights2)
    logging.info('===> Loading weights: ' + config.weights3)
    # state = torch.load(config.weights)
    state1 = torch.load(config.weights1, map_location=lambda s,
                        l: default_restore_location(s, 'cpu'))

    state2 = torch.load(config.weights2, map_location=lambda s,
                        l: default_restore_location(s, 'cpu'))

    state3 = torch.load(config.weights3, map_location=lambda s,
                        l: default_restore_location(s, 'cpu'))

    if 'state_dict' in state1.keys():
        state_key_name1 = 'state_dict'
    elif 'model_state' in state1.keys():
        state_key_name1 = 'model_state'
    else:
        raise NotImplementedError

    if 'state_dict' in state2.keys():
        state_key_name2 = 'state_dict'
    elif 'model_state' in state2.keys():
        state_key_name2 = 'model_state'
    else:
        raise NotImplementedError

    if 'state_dict' in state3.keys():
        state_key_name3 = 'state_dict'
    elif 'model_state' in state3.keys():
        state_key_name3 = 'model_state'
    else:
        raise NotImplementedError

    if config.weights_for_inner_model:
        model1.model.load_state_dict(state1['state_dict'])
        model2.model.load_state_dict(state2['state_dict'])
        model3.model.load_state_dict(state3['state_dict'])
    else:
        if config.lenient_weight_loading:
            matched_weights1 = load_state_with_same_shape(
                model1, state1['state_dict'])
            model_dict1 = model1.state_dict()
            model_dict1.update(matched_weights1)
            model1.load_state_dict(model_dict1)
            matched_weights2 = load_state_with_same_shape(
                model2, state2['state_dict'])
            model_dict2 = model2.state_dict()
            model_dict2.update(matched_weights2)
            model2.load_state_dict(model_dict2)
            matched_weights3 = load_state_with_same_shape(
                model3, state3['state_dict'])
            model_dict3 = model3.state_dict()
            model_dict3.update(matched_weights3)
            model3.load_state_dict(model_dict3)
        else:
            model1.load_state_dict(state1['state_dict'])
            model2.load_state_dict(state2['state_dict'])
            model3.load_state_dict(state3['state_dict'])
    # model = model.to(device)
model1 = model1.to(device)
model2 = model2.to(device)
model3 = model3.to(device)

test(model1, model2, model3, val_data_loader, config)


