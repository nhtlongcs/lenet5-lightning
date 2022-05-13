CUDA_VISIBLE_DEVICES=0  python src/train.py \
                        -c configs/lenet-caltech256.yml \
                        -o global.save_dir=./runs 