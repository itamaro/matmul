#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2017 Itamar Ostricher

"""In-memory matrix multiplication + keep top-K results per column with numpy

Usage:
matmul.py [input_file]
"""

from __future__ import absolute_import

import json
import logging

import numpy as np


def gen_cols(mat_path):
  with open(mat_path, 'r') as mat_f:
    for ser_col in mat_f:
      yield json.loads(ser_col)


def read_matrix(mat_path):
  col_ids = []
  matrix = []
  for col in gen_cols(mat_path):
    col_ids.append(col['id'])
    matrix.append(col['features'])
  return col_ids, matrix


def gen_matches(col_ids, matrix, top_k=1000):
  mult = np.dot(matrix, np.transpose(matrix))
  for i, scores in enumerate(mult):
    top = [{'id': col_ids[l], 'score': scores[l]}
           for l in reversed(np.argsort(scores)[-top_k:])]
    yield {'id': col_ids[i], 'matches': top}


def run(mat_file):
  col_ids, matrix = read_matrix(mat_file)
  for match in gen_matches(col_ids, matrix):
    print(json.dumps(match))


if __name__ == '__main__':
  import sys
  mat_file = sys.argv[1] if len(sys.argv) > 1 else '../input/mat.400.16k.json'
  run(mat_file)
