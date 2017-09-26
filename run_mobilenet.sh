CHECKPOINT_FILE=/data/sls/u/meng/skanda/home/mobilenet/checkpoints/mobilenet/1_224/mobilenet_v1_1.0_224.ckpt
DATASET_DIR=/data/sls/scratch/skoppula/imagenet/tfrecords/val/
python eval_image_classifier.py --alsologtostderr --checkpoint_path=${CHECKPOINT_FILE} --dataset_name=imagenet --dataset_split_name=validation --model_name=mobilenet_v1_050 --dataset_dir=${DATASET_DIR} --preprocessing_name=inception_v3
