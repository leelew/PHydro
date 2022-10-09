import argparse


def parse_args():
    """Hyperparameters"""
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--inputs_path', type=str)
    parser.add_argument('--outputs_path', type=str)
    
    # data
    parser.add_argument('--use_ancillary', type=bool, default=True)
    parser.add_argument('--seq_len', type=int, default=365)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--window_size', type=int, default=0)
    parser.add_argument('--num_out', type=int, default=6)

    # model
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--niter', type=int, default=10000)
    parser.add_argument('--n_filter_factors', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.5) 
    parser.add_argument('--batch_size', type=int, default=32)

    return vars(parser.parse_args())
