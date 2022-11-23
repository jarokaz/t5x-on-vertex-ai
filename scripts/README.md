# WMT En to De on T5 5.1 base
python run.py \
--project_id=jk-mlops-dev \
--region=us-central1 \
--image_uri=gcr.io/jk-mlops-dev/t5x-base \
--staging_bucket=gs://jk-t5x-staging \
--gin_files=../configs/finetune_t511_base_wmt.gin \
--gin_overwrites=USE_CACHED_TASKS=False \
--accelerator_type=TPU_V2 \
--accelerator_count=8 \
--run_mode=train \
--tfds_data_dir=gs://jk-t5x-staging/datasets 

# CNN Daily Mail on longT5 base
python run.py \
--project_id=jk-mlops-dev \
--region=us-central1 \
--image_uri=gcr.io/jk-mlops-dev/t5x-base \
--staging_bucket=gs://jk-t5x-staging \
--gin_files=../configs/finetune_longt5_large.gin \
--gin_search_paths=/flaxformer \
--gin_overwrites=USE_CACHED_TASKS=False \
--accelerator_type=TPU_V2 \
--accelerator_count=128 \
--run_mode=train \
--tfds_data_dir=gs://jk-t5x-staging/datasets 

# CNN Daily Mail on longT5 base
python run.py \
--project_id=renatoleite-dev \
--region=europe-west4 \
--image_uri=gcr.io/renatoleite-dev/t5x-base \
--staging_bucket=gs://rl-t5x-europe-west4 \
--gin_files=../configs/finetune_longt5_base.gin \
--gin_search_paths=/flaxformer \
--gin_overwrites=USE_CACHED_TASKS=False \
--accelerator_type=TPU_V3 \
--accelerator_count=32 \
--run_mode=train \
--tfds_data_dir=gs://rl-t5x-europe-west4


# CNN Daily Mail longT5 XL

python run.py \
--project_id=jk-mlops-dev \
--region=us-central1 \
--image_uri=gcr.io/jk-mlops-dev/t5x-base \
--staging_bucket=gs://jk-t5x-staging \
--gin_files=../configs/finetune_longt5_transient_xl.gin \
--gin_search_paths=/flaxformer \
--gin_overwrites=USE_CACHED_TASKS=False \
--accelerator_type=TPU_V2 \
--accelerator_count=128 \
--run_mode=train \
--tfds_data_dir=gs://jk-t5x-staging/datasets


# XSUM UL2
python run.py \
--project_id=renatoleite-dev \
--region=europe-west4 \
--image_uri=gcr.io/renatoleite-dev/jk-t5x-base \
--staging_bucket=gs://rl-t5x-europe-west4 \
--gin_files=../configs/finetune_ul2_xsum.gin \
--gin_search_paths=/flaxformer \
--gin_overwrites=USE_CACHED_TASKS=False \
--accelerator_type=TPU_V3 \
--accelerator_count=128 \
--run_mode=train \
--tfds_data_dir=gs://rl-t5x-europe-west4/datasets




# XSUM longT5 XL

python run.py \
--project_id=jk-mlops-dev \
--region=europe-west4 \
--image_uri=gcr.io/jk-mlops-dev/t5x-base \
--staging_bucket=gs://jk-staging-europe-west4 \
--gin_files=../configs/finetune_longt5_transient_xl_xsum.gin \
--gin_search_paths=/flaxformer \
--gin_overwrites=USE_CACHED_TASKS=False \
--accelerator_type=TPU_V3 \
--accelerator_count=32 \
--run_mode=train \
--tfds_data_dir=gs://jk-staging-europe-west4/datasets


# XSUM UL2
python run.py \
--project_id=jk-mlops-dev \
--region=europe-west4 \
--image_uri=gcr.io/jk-mlops-dev/t5x-base \
--staging_bucket=gs://jk-staging-europe-west4 \
--gin_files=../configs/finetune_ul2_xsum.gin,../configs/ul220b_public.gin \
--gin_search_paths=/flaxformer \
--gin_overwrites=USE_CACHED_TASKS=False,TRAIN_STEPS=2_700_000,INITIAL_CHECKPOINT_PATH=\"gs://scenic-bucket/ul2/ul220b/checkpoint_2650000\" \
--accelerator_type=TPU_V3 \
--accelerator_count=128 \
--run_mode=train \
--tfds_data_dir=gs://jk-staging-europe-west4/datasets


# CUAD Long T5 Large
python run.py \
--project_id=jk-mlops-dev \
--region=europe-west4 \
--image_uri=gcr.io/jk-mlops-dev/t5x-base \
--staging_bucket=gs://jk-staging-europe-west4 \
--gin_files=../configs/finetune_longT5_xl_cuad.gin \
--gin_search_paths=/flaxformer \
--gin_overwrites=USE_CACHED_TASKS=False, \
--accelerator_type=TPU_V3 \
--accelerator_count=128 \
--run_mode=train \
--tfds_data_dir=gs://jk-staging-europe-west4/datasets



# Tensorboard


export TENSORBOARD_NAME=projects/895222332033/locations/us-central1/tensorboards/2937103421045473280
export REGION=us-central1
export EXPERIMENT_NAME=ul2-xsum-10
export LOG_DIR=gs://jk-staging-europe-west4/t5x_jobs/t5x_job_20221112014722

tb-gcp-uploader --tensorboard_resource_name $TENSORBOARD_NAME \
--logdir $LOG_DIR \
--experiment_name $EXPERIMENT_NAMEexport PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"