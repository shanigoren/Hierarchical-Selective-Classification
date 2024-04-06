# import sys
import argparse


def get_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--seed', type=int, default=0, help='Give random seed')
    # parser.add_argument('-nw', '--num_workers', type=int, default=0, help='Number of workers to fetch samples from memory.')
    parser.add_argument('-d', '--dataset', default='iNat21-mini',
                        choices=['food101', 'flowers102', 'dtd', 'fgvcaircraft', 'imagenet', 'stanfordcars', 'sun397', 'iNat21-mini'],
                        help='Give the dataset name from the choices.')
    parser.add_argument('-m', '--model_name', default='resnet50.tv_in1k', \
                        help='Decide the trained model architecture from the choices')
    parser.add_argument('-sc', '--scheduler', default='WarmupCosine', \
                        help='Decide the trained model architecture from the choices')
    parser.add_argument('-e', '--max_epochs', type=int, default=1, help='Give number of epochs for training')
    parser.add_argument('-bs', '--batch_size', type=int, default=512,
                        help='Batch size.\n\
                            Note that actual batch size is equal to batch_size*accumulating_grads_factor')
    parser.add_argument('-ag', '--accumulating_grads_factor', type=int, default=1,
                        help='Accumulating grads factor.\n\
                            How many steps must be accumulated to perform an update step')
    # parser.add_argument('-p', '--percentage', type=int, help='Percentage of dataset to be used for training', default=None)
    # parser.add_argument('-g', '--gpus', type=int , nargs="+", help='which GPUs (or CPU) to be used for training', default=[0])
    parser.add_argument('-g', '--gpu_id', type=str, help='string represents which GPUs to be used for training.\n\
                        For example 0 to train only on single gpu. 0,1 for two gpus.', default="0")

    # parser.add_argument('-a', '--api_key', type=str, help='Comet ML API key', default=None)
    # parser.add_argument('-w', '--workspace', type=str, help='Comet ML workspace name or ID', default=None)
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='learning rate for training')
    parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float, help='weight decay for training')
    parser.add_argument('-f', '--finetune', action='store_true', help='Whether to finetune or not.' \
                        , default=True)
    parser.add_argument('-sub', '--subset', action='store_true', help='Only train on a subset of the dataset.' \
                        , default=False)
    parser.add_argument('-dss', '--dataset_subset_size', type=int, default=1000,
                        help='Train and validation subset size (for debugging)')

    # parser.add_argument('-m', '--model', choices = ['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26'], help = 'Give the model name from the choices')
    # parser.add_argument('-g', '--gpu', choices=[0, 1, 'cpu'], help='which GPU (or CPU) to be used for training', default=0)
    # parser.add_argument('-p', '--percentage', type=int, help='Percentage of dataset to be used for training', default=None)
    # parser.add_argument('-e', '--experiment', choices=['stagewise-kd', 'traditional-kd', 'simultaneous-kd', 'no-teacher', 'fsp-kd', 'attention-kd'])

    args = parser.parse_args()
    return args