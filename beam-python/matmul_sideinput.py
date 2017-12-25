#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2017 by Itamar Ostricher

"""MatMul Beam pipeline - entire matrix as side input.

Run locally:
python matmul_sideinput.py \
    --input ../smallMat.json --output ../out/beam-py-scores

Run using Cloud Dataflow:
python matmul_sideinput.py \
    --runner DataflowRunner --project PROJECT_ID \
    --input gs://beam-matmul/mat/1000X16k/mat.001.json \
    --output gs://beam-matmul/output/py-beam/1000X16k-side
"""

from __future__ import absolute_import

import argparse
import json
import logging

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import numpy as np


def calc_scores(col, mat, num_results):
  col_ids = []
  features = []
  for other_col in mat:
    col_ids.append(other_col['id'])
    features.append(other_col['features'])
  scores = np.dot(features, col['features'])
  return {
      'id': col['id'],
      'matches': [{'id': col_ids[j], 'score': scores[j]}
                  for j in reversed(np.argsort(scores)[-num_results:])]
  }


def run(argv=None):
  """Main entry point; defines and runs the wordcount pipeline."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', dest='input', help='Input file to process.',
                      default='../input/smallMat.json')
  parser.add_argument('--output', dest='output', required=True,
                      help='Output file to write results to.')
  parser.add_argument('--runner', dest='runner', default='DirectRunner',
                      help='Runner class to use (DataflowRunner for GCP).')
  parser.add_argument('--project', dest='project',
                      help='GCP project for Cloud Dataflow')
  parser.add_argument('--num-results', dest='num_results',
                      default=1000, type=int,
                      help='Number of results top results to keep per column.')
  known_args, pipeline_args = parser.parse_known_args(argv)
  pipeline_args.extend([
      '--runner={}'.format(known_args.runner),
      '--project={}'.format(known_args.project),
      '--staging_location=gs://beam-matmul/staging',
      '--temp_location=gs://beam-matmul/temp',
      '--job_name=matmul-side-input',
  ])

  # We use the save_main_session option because one or more DoFn's in this
  # workflow rely on global context (e.g., a module imported at module level).
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True
  with beam.Pipeline(options=pipeline_options) as p:
    files = p | 'Read ls List' >> ReadFromText(known_args.input)
    mat = files | 'Parse Columns' >> beam.Map(json.loads)
    (mat | 'Calc Scores' >> beam.Map(calc_scores, beam.pvalue.AsIter(mat),
                                     known_args.num_results)
         | 'Serialize as JSON' >> beam.Map(json.dumps)
         | 'Write Scores' >> WriteToText(known_args.output)
    )


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
