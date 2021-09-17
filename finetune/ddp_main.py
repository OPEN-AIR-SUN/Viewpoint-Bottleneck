# Change dataloader multiprocess start method to anything not fork
import numpy as np
import open3d as o3d
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn')  # Reuse process created
except RuntimeError:
    pass

import json
import logging
import os
import random
import sys

# Torch packages
import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from torch.serialization import default_restore_location

# Train deps
from config import get_config
from lib import distributed_utils
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from lib.test import test
from lib.train import train
from lib.utils import count_parameters, load_state_with_same_shape, timestamp
from models import load_model, load_wrapper


def setup_logging(config):
    handlers = [logging.StreamHandler(sys.stdout)]

    # FIXME update config 'run_name', 'log_dir', 'logging'
    # if config.logging:
    #     handlers.append(
    #         logging.StreamHandler(
    #             os.path.join(config.log_dir, f'{config.run_name} - {timestamp()}.log')))

    # TODO add file handler
    if config.distributed_world_size > 1 and config.distributed_rank > 0:
        logging.getLogger().setLevel(logging.WARN)
    else:
        logging.getLogger().setLevel(logging.INFO)

    logging.basicConfig(format=f'{os.uname()[1]} %(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)


def main():
    config = get_config()
    num_devices = torch.cuda.device_count()

    print(f"Run with {num_devices} GPUS. Total batch size is {num_devices * config.batch_size}")

    if torch.cuda.device_count() > 1:
        port = random.randint(10000, 20000)
        config.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        config.distributed_rank = None  # set based on device id
        mp.spawn(
            fn=distributed_main,
            args=(config,),
            nprocs=config.distributed_world_size,
        )
    else:
        main_worker(config)


def distributed_main(i, config, start_rank=0):
    config.device_id = i
    if config.distributed_rank is None:  # torch.multiprocessing.spawn
        config.distributed_rank = start_rank + i
    main_worker(config, init_distributed=True)


def main_worker(config, init_distributed=False):
    if config.resume:
        # TODO repair config.resume
        json_config = json.load(open(config.resume + '/config.json', 'r'))
        json_config['resume'] = config.resume
        config = edict(json_config)

    if config.is_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")

    # setup initial seed
    # TODO check if python and numpy seed need to be set
    torch.cuda.set_device(config.device_id) # important
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    device = config.device_id
    distributed = config.distributed_world_size > 1

    if init_distributed:
        config.distributed_rank = distributed_utils.distributed_init(config)

    setup_logging(config)
    logging.info('====> Configurations <====')
    dconfig = vars(config)
    for k, v in dconfig.items():
        logging.info(f'----> {k:>30}: {v}')

    train_dataset_class = load_dataset(config.dataset)
    test_dataset_class = load_dataset(config.testdataset)
    if config.test_original_pointcloud and not train_dataset_class.IS_FULL_POINTCLOUD_EVAL:
        raise ValueError('This dataset does not support full pointcloud evaluation.')

    # TODO check necessity of eva..pointcloud and return_transformation
    if config.evaluate_original_pointcloud and not config.return_transformation:
        raise ValueError('Pointcloud evaluation requires config.return_transformation=True.')

    if (config.return_transformation ^ config.evaluate_original_pointcloud):
        raise ValueError(
            'Rotation evaluation requires config.evaluate_original_pointcloud=true and '
            'config.return_transformation=true.')

    logging.info('====> Initializing dataloader <====')
    if config.is_train:  # TODO change is_train to train
        # TODO simplify the params
        train_data_loader = initialize_data_loader(train_dataset_class,
                                                   config,
                                                   phase=config.train_phase,
                                                   num_workers=config.num_workers,
                                                   augment_data=True,
                                                   shuffle=True,
                                                   repeat=True,
                                                   batch_size=config.batch_size,
                                                   limit_numpoints=config.train_limit_numpoints)

        val_data_loader = initialize_data_loader(test_dataset_class,
                                                 config,
                                                 num_workers=config.num_val_workers,
                                                 phase=config.val_phase,
                                                 augment_data=False,
                                                 shuffle=True,
                                                 repeat=False,
                                                 batch_size=config.val_batch_size,
                                                 limit_numpoints=False)

        num_in_channel = 3 if train_data_loader.dataset.NUM_IN_CHANNEL is None \
                            else train_data_loader.dataset.NUM_IN_CHANNEL

        num_labels = train_data_loader.dataset.NUM_LABELS
    else:
        test_data_loader = initialize_data_loader(test_dataset_class,
                                                  config,
                                                  num_workers=config.num_workers,
                                                  phase=config.test_phase,
                                                  augment_data=False,
                                                  shuffle=False,
                                                  repeat=False,
                                                  batch_size=config.test_batch_size,
                                                  limit_numpoints=False)

        num_in_channel = 3 if test_data_loader.dataset.NUM_IN_CHANNEL is None \
                                else test_data_loader.dataset.NUM_IN_CHANNEL

        num_labels = test_data_loader.dataset.NUM_LABELS

    # TODO to be revised
    logging.info('====> Building model <====')
    NetClass = load_model(config.model)
    if config.wrapper_type == 'None':
        model = NetClass(num_in_channel, num_labels, config)
        logging.info('----> Number of trainable parameters: {}: {}'.format(
            NetClass.__name__, count_parameters(model)))
    else:
        wrapper = load_wrapper(config.wrapper_type)
        model = wrapper(NetClass, num_in_channel, num_labels, config)
        logging.info('----> Number of trainable parameters: {}: {}'.format(
            wrapper.__name__ + NetClass.__name__, count_parameters(model)))

    logging.info(f'Model structure: \n{model}')

    if config.weights == 'modelzoo':  # Load modelzoo weights if possible.
        logging.info('====> Loading modelzoo weights <====')
        model.preload_modelzoo()

    # Load weights if specified by the parameter.
    elif config.weights.lower() != 'none':
        logging.info('====> Loading weights: <====' + config.weights)
        state = torch.load(config.weights,
                           map_location=lambda s, l: default_restore_location(s, 'cpu'))

        assert 'state_dict' in state.keys() or 'model_state' in state.keys()

        if config.weights_for_inner_model:
            model.model.load_state_dict(state['state_dict'])
        else:
            if config.lenient_weight_loading:
                matched_weights = load_state_with_same_shape(model, state['state_dict'])
                model_dict = model.state_dict()
                model_dict.update(matched_weights)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(state['state_dict'])
    model = model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[device],
                                                          output_device=device,
                                                          broadcast_buffers=False,
                                                          bucket_cap_mb=config.bucket_cap_mb)

    if config.is_train:
        train(model, train_data_loader, val_data_loader, config)
    else:
        test(model, test_data_loader, config)


if __name__ == '__main__':
    main()
