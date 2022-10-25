import argparse


def parse_args():
    """Hyperparameters"""
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--inputs_path', type=str, default="/data/lilu/PHydro/input/")
    parser.add_argument('--outputs_path', type=str, default="/data/lilu/PHydro/output/")
    parser.add_argument('--work_path', type=str, default="/data/lilu/PHydro")

    # data
    parser.add_argument('--use_ancillary', type=bool, default=False)
    parser.add_argument('--seq_len', type=int, default=365)
    parser.add_argument('--interval', type=int, default=365)
    parser.add_argument('--window_size', type=int, default=0)
    parser.add_argument('--num_out', type=int, default=6)
    parser.add_argument('--num_feat', type=int, default=9)

    # model
    parser.add_argument('--model_name', type=str, default="single_task")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--n_filter_factors', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.5) 
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    return vars(parser.parse_args())
