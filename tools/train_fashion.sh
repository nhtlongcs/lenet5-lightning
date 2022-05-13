CUDA_VISIBLE_DEVICES=0  python src/train.py \
                        -c configs/lenet-fashion.yml \
                        -o global.save_dir=./runs 