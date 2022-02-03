import pandas as pd
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='RecPlay')
parser.add_argument('--model_init_seed', type=int, default=1)
# BERT #
parser.add_argument('--bert_max_len', type=int, default=50, help='Length of sequence for bert')
parser.add_argument('--bert_num_items', type=int, default=None, help='Number of total items')
parser.add_argument('--bert_hidden_units', type=int, default=256, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=16, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=0.2, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_mask_prob', type=float, default=0.2, help='Probability for masking items in the training sequence')
args = parser.parse_args()
