#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2017 Itamar Ostricher

"""In-memory sharded matrix multiplication & top-K results per row with numpy

Usage:
sharded_matmul.py [-m <input_file>] [-m <input_file> ...]

Treats every individual input file as a single shard.
"""

from __future__ import absolute_import

from functools import partial, reduce
from itertools import combinations_with_replacement
import json
import os
import shutil
import tempfile

import numpy as np


def gen_rows(mat_path):
  with open(mat_path, 'r') as mat_f:
    for ser_row in mat_f:
      yield json.loads(ser_row)


def read_matrix(mat_path):
  row_ids, matrix = [], []
  for row in gen_rows(mat_path):
    row_ids.append(row['id'])
    matrix.append(row['features'])
  return np.array(row_ids), np.matrix(matrix)


def gen_scores(row_ids1, matrix1, row_ids2, matrix2, top_k):
  """Generate a block (minor) of scores for two shards of input."""
  mult = np.dot(matrix1, np.transpose(matrix2))
  for i, scores in enumerate(mult):
    argresults = list(reversed(np.argsort(scores, axis=1).tolist()[0][-top_k:]))
    top = {'id': [row_ids2[j] for j in argresults],
           'score': [mult[i, j] for j in argresults]}
    yield {'id': row_ids1[i], 'matches': top}


def write_scores_block(tmpdir, block_id, ids1, mat1, ids2, mat2, top_k):
  """Write a scores block (minor) to disk."""
  with open(os.path.join(tmpdir,
                         '.'.join(map(str, block_id))), 'w') as block_f:
    scores = list(gen_scores(ids1, mat1, ids2, mat2, top_k))
    json.dump(scores, block_f)


def write_partial_scores(mat_files, top_k, tmpdir):
  """Go over all pairs of input shards and write their
     corresponding score blocks to disk."""
  for m1, m2 in combinations_with_replacement(enumerate(mat_files), r=2):
    block_id = (m1[0], m2[0])
    row_ids1, matrix1 = read_matrix(m1[1])
    if m1 == m2:
      write_scores_block(
          tmpdir, block_id, row_ids1, matrix1, row_ids1, matrix1, top_k)
    else:
      row_ids2, matrix2 = read_matrix(m2[1])
      write_scores_block(
          tmpdir, block_id, row_ids1, matrix1, row_ids2, matrix2, top_k)
      write_scores_block(
          tmpdir, block_id[::-1], row_ids2, matrix2, row_ids1, matrix1, top_k)


def reduce_score_blocks(block1, block2, top_k):
  """Join two matching score blocks to a single unified block.

  Assumption: block1 & block2 are blocks for the same rows in the final matrix.
  """
  for row1, row2 in zip(block1, block2):
    ids = row1['matches']['id'] + row2['matches']['id']
    scores = row1['matches']['score'] + row2['matches']['score']
    argresults = list(reversed(np.argsort(scores)[-top_k:]))
    row1['matches'] = {
        'id': [ids[i] for i in argresults],
        'score': [scores[i] for i in argresults],
    }
  return block1


def reduce_partial_scores(num_batches, top_k, tmpdir):
  """Go over all row batches and reduce the score blocks to a final block."""
  for i in range(num_batches):

    def load_block(j):
      with open(os.path.join(tmpdir, f'{i}.{j}'), 'r') as block_f:
        return json.load(block_f)

    for row in (reduce(partial(reduce_score_blocks, top_k=top_k),
                       map(load_block, range(num_batches)))):
      print(json.dumps(row, sort_keys=True))


def run(mat_files, top_k):
  tmpdir = tempfile.mkdtemp()
  try:
    write_partial_scores(mat_files, top_k, tmpdir)
    reduce_partial_scores(len(mat_files), top_k, tmpdir)
  finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(
      description='Local matrix multiplication with NumPy.')
  parser.add_argument('-m', action='append', help='Matrix input file(s)')
  parser.add_argument('-k', type=int, default=1000,
                      help='Number of top results to keep')
  args = parser.parse_args()
  run(args.m or ['../input/mat.400.16k.json'], args.k)
