import argparse
from train import train


def parse_training_args():

    parser = argparse.ArgumentParser(description="Character Level Language Identification Model")
    parser.add_argument("--learning_rate", default=0.01, type=float,
                        help="The initial learning rate."
                        )
    parser.add_argument("--batch_size", default=128, type=int,
                        help="kwarg passed to dataloader."
                        )
    parser.add_argument("--num_epochs", default=10, type=int,
                        help="Number of training epochs."
                        )
    parser.add_argument("--train_data", default="data/wili-2018/x_train_new.txt", type=str,
                        help="Path to training data."
                        )
    parser.add_argument("--train_labels", default="data/wili-2018/y_train_new.txt", type=str,
                        help="Path to training labels."
                        )
    parser.add_argument("--val_data", default="data/wili-2018/x_val.txt", type=str,
                        help="Path to validation data."
                        )
    parser.add_argument("--val_labels", default="data/wili-2018/y_val.txt", type=str,
                        help="Path to validation labels."
                        )
    parser.add_argument("--test_data", default="data/wili-2018/x_test.txt", type=str,
                        help="Path to testing data."
                        )
    parser.add_argument("--test_labels", default="data/wili-2018/y_test.txt", type=str,
                        help="Path to testing labels."
                        )
    parser.add_argument("--split_paragraphs", default=False, type=bool,
                        help="Split paragraphs to sentences while prepocessing."
                        )
    parser.add_argument("--max_chars", default=250, type=int,
                        help="Maximum input sequence length. Shorter sequences will be padded and longer truncated."
                        )
    parser.add_argument("--fix_lengths", default=True, type=bool,
                        help="Fix the length of the input sequence to maximum length allowed."
                        )
    parser.add_argument("--use_cuda", default=True, type=bool,
                        help="Whether to use GPU."
                        )
    parser.add_argument("--dropout", default=0., type=float,
                        help="Dropout probability at fc layers of network."
                        )
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Path to logging directory."
                        )
    parser.add_argument("--level", default="char", type=str,
                        help="Character level model. Here in case we extend the implementation later."
                        )
    parser.add_argument("--emb_dim", default=80, type=int,
                        help="Embedding dimension for the network."
                        )
    parser.add_argument("--min_frequency", default=10, type=int,
                        help="Minimum word frequency for inclusion in the vocab."
                        )

    return parser


if __name__ == '__main__':

    parser = parse_training_args()
    args = vars(parser.parse_args())
    train(**args)