sudo mkdir /mnt/vlm-pd;sudo mount -o ro,noload /dev/sdb /mnt/vlm-pd"

export WANDB_API_KEY=97c3b77fabd233d22d7b9a71319fce93f7400469
        source /mnt/vlm-pd/miniconda3/bin/activate;
        conda activate /mnt/vlm-pd/miniconda3/envs/vlm;
        export TFDS_DATA_DIR=gs://us-central2-storage/tensorflow_datasets;
        cd ~/big_vision;
        python -m big_vision.trainers.proj.image_text.siglip \
            --config big_vision/configs/proj/image_text/siglip_replication.py:batch_size=1024 \
            --workdir gs://us-central2-storage/tensorflow_datasets/siglip_replication_pod_`date '+%m-%d_%H%M'`
    "


cd ~/big_vision;
python -m big_vision.trainers.proj.image_text.siglip \
    --config big_vision/configs/proj/image_text/siglip_replication.py:batch_size=512 \
    --workdir gs://us-central2-storage/tensorflow_datasets/siglip_replication_pod_`date '+%m-%d_%H%M'`