config=$1
checkpoint=$2
submission_dir=$3

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 bash tools/dist_test.sh $config $checkpoint 4 --format-only --eval-options submission_dir=$3