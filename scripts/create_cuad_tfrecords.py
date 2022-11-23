# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A utility to submit a Vertex Training T5X job."""

import os

from absl import flags
from absl import app
from absl import logging
from datetime import datetime

from typing import List
from typing import Dict
from typing import Any
from typing import Union
from typing import Optional


from datasets import load_dataset_builder
from datasets import load_dataset


flags.DEFINE_string('project_id', None, 'GCP Project')
flags.DEFINE_string('region', None, 'Vertex Pipelines region')
flags.DEFINE_string('staging_bucket', None, 'Staging bucket')
flags.DEFINE_string('training_sa', None, 'Training SA')
flags.DEFINE_string('tfrecords_name', 'cuad.tfrecord', 'The name of tfrecords file')
flags.DEFINE_string('job_name_prefix', 't5x_job', 'Job name prefix')
flags.DEFINE_list('gin_files', None, 'Gin configuration files')
flags.DEFINE_list('gin_overwrites', None, 'Gin overwrites')
flags.DEFINE_list('gin_search_paths', None, 'Gin search paths')
flags.DEFINE_string('accelerator_type', 'TPU_V2', 'Accelerator type')
flags.DEFINE_integer('accelerator_count', 8, 'Number of cores')
flags.DEFINE_string('run_mode', 'train', 'Run mode')
flags.DEFINE_string('tfds_data_dir', None, 'TFDS data dir')
flags.DEFINE_bool('sync', True, 'Execute synchronously')
#flags.mark_flag_as_required('project_id')
#flags.mark_flag_as_required('region')
#flags.mark_flag_as_required('staging_bucket')
#flags.mark_flag_as_required('gin_files')
#flags.mark_flag_as_required('image_uri')
FLAGS = flags.FLAGS



def _main(argv):
    logging.info('Starging conversion')
    
    ds_builder = load_dataset_builder('cuad')

    logging.info(ds_builder.info.description)

    dataset = load_dataset('cuad', split='train')
    dataset.set_format(type='numpy')

    logging.info(dataset)

    dataset.export(FLAGS.tfrecords_name, format='tfrecord')





if __name__ == "__main__":
    app.run(_main)

