import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='tsp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=1280000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=12800,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')

    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=3,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")


    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=2000, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default=None,
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts






















































# import os
# import time
# import argparse
# import torch
#
#
# def get_options(args=None):
#     parser = argparse.ArgumentParser(description="Meta learning for MOTSP")
#     parser.add_argument('--problem', default='tsp', help="The problem to solve, default 'tsp'")
#     parser.add_argument('--meta_algorithm', default='reptile', help="The Meta-learning algorithm")
#     parser.add_argument('--no_first_order', action='store_true', default=False, help='whether to disable the first-order approximation')
#     parser.add_argument('--is_load_multi', action='store_true', default=False)
#     parser.add_argument('--get_training_data', action='store_true', default=False)
#     # lr
#     parser.add_argument('--meta_lr', type=float, default=1e-4, help="Set the meta learning rate for the meta policy")
#     parser.add_argument('--sub_train_lr', type=float, default=1e-4,
#                         help="Set the learning rate for training the sub-policy")
#     parser.add_argument('--finetunning_lr', type=float, default=1e-4,
#                         help="Set the learning rate for the finetunning phase")
#     parser.add_argument('--meta_lr_decay', type=float, default=0.99, help='Meta learning rate decay per epoch')
#
#     # step
#     parser.add_argument('--update_step', type=int, default=100, help='Adaptation steps when training meta policy')
#     parser.add_argument('--update_step_test', type=int, default=100, help='Finetuning steps')
#
#     # train
#     parser.add_argument('--is_train', action='store_true', default=False, help='whether to train')
#     parser.add_argument('--is_test', action='store_true', default=False, help='whether to test')
#     parser.add_argument('--task_num', type=int, default=5, help='The number of subtasks that need to be trained each time the parameters of the meta-policy are updated')
#     parser.add_argument('--training_epochs', type=int, default=100, help='The number of epochs to train the meta policy')
#     parser.add_argument('--start_epoch', type=int, default=0, help='The start of the epoch number to train the meta policy')
#     parser.add_argument('--meta_iterations', type=int, default=10000,
#                         help='The number of iterations to train the meta policy')
#     parser.add_argument('--start_meta_iteration', type=int, default=0,
#                         help='The start of the iteration number to train the meta policy')
#
#     parser.add_argument('--is_transfer', action='store_true', default=False, help='whether to transfer learning')
#     # size
#     parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
#     parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
#     parser.add_argument('--epoch_size', type=int, default=51200, help='Number of instances per epoch during training')
#     parser.add_argument('--val_size', type=int, default=10000,
#                         help='Number of instances used for reporting validation performance')
#     parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')
#     parser.add_argument('--eval_batch_size', type=int, default=512,
#                         help="Batch size to use during (baseline) evaluation")
#
#     # cuda
#     parser.add_argument('--CUDA_VISIBLE_ID', default="0")
#     parser.add_argument('--no_cuda', action='store_true', default = False, help='whether to disable cuda')
#
#     # load
#     parser.add_argument('--is_load', action='store_true', default=False, help='whether to load model')
#     # parser.add_argument('--load_dir', default='outputs/tsp_20/meta_MORL_clipgradnorms5_tsp20_critic_20200913T224718', help='Directory to load models')
#     # parser.add_argument('--load_name', default='meta-model-epoch-70.pt', help='Name to load model parameters and optimizer state from')
#     parser.add_argument('--load_path', default='None.pt', help = 'Path to load model parameters and optimizer state from')
#     parser.add_argument('--eval_hv_dir', default='None')
#     # Model
#     parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
#     parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
#     parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
#     parser.add_argument('--n_encode_layers', type=int, default=3,
#                         help='Number of layers in the encoder/critic network')
#     parser.add_argument('--tanh_clipping', type=float, default=10.,
#                         help='Clip the parameters to within +- this value using tanh. '
#                              'Set to 0 to not perform any clipping.')
#     parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")
#
#     # Training
#     parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
#     parser.add_argument('--max_grad_norm', type=float, default=1.0,
#                         help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
#     parser.add_argument('--baseline', default=None,
#                         help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
#     parser.add_argument('--checkpoint_encoder', action='store_true',
#                         help='Set to decrease memory usage by checkpointing encoder')
#     parser.add_argument('--shrink_size', type=int, default=None,
#                         help='Shrink the batch size if at least this many instances in the batch are finished'
#                              ' to save memory (default None means no shrinking)')
#     parser.add_argument('--data_distribution', type=str, default=None,
#                         help='Data distribution to use during training, defaults and options depend on problem.')
#
#     # Misc
#     parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
#     parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
#     parser.add_argument('--run_name', default='run', help='Name to identify the run')
#     parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
#     parser.add_argument('--resume', help='Resume from previous checkpoint file')
#     parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
#     parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
#
#     parser.add_argument('--cal_hv', action='store_true', default=False, help='whether to calculate HV')
#
#     # test
#     parser.add_argument('--resume_i', type=int, default=0, help='The resume id to test')
#     mt_opts = parser.parse_args(args)
#
#     # mt_opts.load_path = os.path.join(
#     #     mt_opts.load_dir,
#     #     mt_opts.load_name
#     # )
#     if mt_opts.meta_algorithm == "reptile" and mt_opts.meta_lr == 1e-4:
#         mt_opts.meta_lr = 1.
#
#     mt_opts.use_cuda = False if mt_opts.no_cuda else True
#     mt_opts.run_name = "{}_{}".format(mt_opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
#     mt_opts.save_dir = os.path.join(
#         mt_opts.output_dir,
#         "{}_{}".format(mt_opts.problem, mt_opts.graph_size),
#         mt_opts.run_name
#     )
#     mt_opts.output_dir = "{}_{}".format(mt_opts.meta_algorithm, mt_opts.output_dir)
#     if mt_opts.no_first_order:
#         mt_opts.output_dir += "_no_first_order"
#     assert mt_opts.meta_algorithm == 'reptile', "Only support for reptile"
#     assert mt_opts.epoch_size % mt_opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
#     return mt_opts
