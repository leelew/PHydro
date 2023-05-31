import argparse


def parse_args():
    """Hyperparameters"""
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--inputs_path', type=str, default="/data/lilu/PHydro_era/input/")
    parser.add_argument('--outputs_path', type=str, default="/data/lilu/PHydro_era/output/")
    parser.add_argument('--work_path', type=str, default="/data/lilu/PHydro_era")

    # data
    parser.add_argument('--reuse_input', type=bool, default=False)
    parser.add_argument('--use_ancillary', type=bool, default=True)
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--interval', type=int, default=365)
    parser.add_argument('--window_size', type=int, default=0)
    parser.add_argument('--num_out', type=int, default=5)
    parser.add_argument('--num_feat', type=int, default=11, choices=[8, 11])
    parser.add_argument('--resid_idx', type=int, default=0)
    parser.add_argument('--lam', type=int, default=1)
    parser.add_argument('--spatial_cv', type=int, default=-1)
    parser.add_argument('--temporal_cv', type=int, default=-1)
    parser.add_argument('--ngrid', type=int, default=200)

    # model
    parser.add_argument('--random_ratio', type=float, default=1)
    parser.add_argument('--model_name', type=str, default="hard_multi_tasks_v1")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--scaling_factor', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--main_idx', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--split_ratio', type=float, default=0.75)
    parser.add_argument('--num_repeat', type=int, default=1)
    return vars(parser.parse_args())
