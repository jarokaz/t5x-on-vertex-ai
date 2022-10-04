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


## CNN Daily Mail eval longT5 XL
python run.py \
--project_id=jk-mlops-dev \
--region=us-central1 \
--image_uri=gcr.io/jk-mlops-dev/t5x-base \
--staging_bucket=gs://jk-t5x-staging \
--gin_files=../configs/eval_longt5_transient_xl.gin \
--gin_search_paths=/flaxformer \
--gin_overwrites=USE_CACHED_TASKS=False,CHECKPOINT_PATH=\"gs://jk-t5x-staging/t5x_jobs/t5x_job_20221002170444/checkpoint_1009000\" \
--accelerator_type=TPU_V2 \
--accelerator_count=32 \
--run_mode=eval \
--tfds_data_dir=gs://jk-t5x-staging/datasets



# Tensorboard


export TENSORBOARD_NAME=projects/895222332033/locations/us-central1/tensorboards/2937103421045473280
export REGION=us-central1
export EXPERIMENT_NAME=longt5-xl-1
export LOG_DIR=gs://jk-t5x-staging/t5x_jobs/t5x_job_20221002170444

tb-gcp-uploader --tensorboard_resource_name $TENSORBOARD_NAME \
--logdir $LOG_DIR \
--experiment_name $EXPERIMENT_NAME