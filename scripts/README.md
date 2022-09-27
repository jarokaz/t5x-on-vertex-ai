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
--gin_files=../configs/finetune_longt5_base.gin \
--gin_search_paths=/flaxformer \
--gin_overwrites=USE_CACHED_TASKS=False \
--accelerator_type=TPU_V2 \
--accelerator_count=32 \
--run_mode=train \
--tfds_data_dir=gs://jk-t5x-staging/datasets 

