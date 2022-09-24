python run.py \
--project_id=jk-mlops-dev \
--region=us-central1 \
--image_uri=gcr.io/jk-mlops-dev/t5x-base \
--staging_bucket=jk-t5x-staging \
--gin_files=../configs/finetune_t511_base_wmt.gin \
--gin_overwrites=USE_CACHED_TASKS=false \
--accelerator_type=TPU_V2 \
--accelerator_count=8 \
--run_mode=train \
--tfds_data_dir=gs://jk-t5x-staging/datasets 

