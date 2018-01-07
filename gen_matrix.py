#!/usr/bin/env python3

"""Generate Random Matrix

Writes out JSON serialized matrix, one row per line,
each row serialized as dictionary with `id` and `features` keys.
"""

import json
import random

_NUM_FEATURES = 16000
_NUM_ELEMENTS = 10000


def gen_element(num_high, num_low):
  """Return a vector with `num_high` 1.0 and `num_low` 0.0
     distributed randomly."""
  elm = [0.0] * num_low + [1.0] * num_high
  random.shuffle(elm)
  return elm


def gen_matrix(num_elements, first_id, num_high, num_low):
  """Generate matrix with `num_elements` columns"""
  for elm_id in range(num_elements):
      yield {'id': f'{elm_id+first_id:08x}', 'features': gen_element(num_high, num_low)}


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Generate a random matrix.')
  parser.add_argument('--num-features', default=16000, type=int,
                      help='Number of features per matrix row')
  parser.add_argument('--num-elements', default=1000, type=int,
                      help='Number of matrix rows to generate')
  parser.add_argument('--first-id', default=0, type=int,
                      help='ID to start from for generated matrix rows')
  args = parser.parse_args()
  num_high = args.num_features // 2
  num_low = args.num_features - num_high
  for col in gen_matrix(args.num_elements, args.first_id, num_high, num_low):
    print(json.dumps(col))
