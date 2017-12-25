#!/usr/bin/env python3

"""Generate Random Matrix

Writes out JSON serialized matrix, one column per line,
each column serialized as dictionary with `id` and `features` keys.
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


def gen_matrix(num_elements, num_high, num_low):
  """Generate matrix with `num_elements` columns"""
  for elm_id in range(_NUM_ELEMENTS):
      yield {'id': f'{elm_id:08x}', 'features': gen_element(num_high, num_low)}


if __name__ == '__main__':
  num_high = _NUM_FEATURES // 2
  num_low = _NUM_FEATURES - num_high
  for col in gen_matrix(_NUM_ELEMENTS, num_high, num_low):
    print(json.dumps(col))
