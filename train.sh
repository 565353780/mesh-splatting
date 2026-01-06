DATA_FOLDER=$HOME/chLi/Dataset/GS/haizei_1

python train.py \
  -s ${DATA_FOLDER}/gs/ \
  -m ${DATA_FOLDER}/meshsplatting/ \
  --eval
