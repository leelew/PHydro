import argparse


def parse_args():
    """Hyperparameters"""
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--inputs_path', type=str)

    # data
    parser.add_argument('--use_ancillary', type=bool, default=True)
    parser.add_argument('--seq_len', type=int, default=365)
    parser.add_argument('--interval', type=int, default=7)
    parser.add_argument('--window_size', type=int, default=0)

    # model
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    return parser.parse_args()
