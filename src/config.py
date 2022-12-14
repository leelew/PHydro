import argparse


def parse_args():
    """Hyperparameters"""
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--inputs_path', type=str, default="/data/lilu/PHydro/input/")
    parser.add_argument('--outputs_path', type=str, default="/data/lilu/PHydro/output/")
    parser.add_argument('--work_path', type=str, default="/data/lilu/PHydro")

    # data
    parser.add_argument('--reuse_input', type=bool, default=False)
    parser.add_argument('--use_ancillary', type=bool, default=True)
    parser.add_argument('--seq_len', type=int, default=365)
    parser.add_argument('--interval', type=int, default=365)
    parser.add_argument('--window_size', type=int, default=0)
    parser.add_argument('--num_out', type=int, default=6)
    parser.add_argument('--num_feat', type=int, default=24, choices=[9, 24])
    parser.add_argument('--resid_idx', type=int, default=0)
    parser.add_argument('--lam', type=int, default=0.01)

    # model
    parser.add_argument('--model_name', type=str, default="hard_multi_tasks_v1")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--niter', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--num_repeat', type=int, default=5)
    return vars(parser.parse_args())
