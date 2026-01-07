DATA_FOLDER=$HOME/chLi/Dataset/MM/Match/gold_dragon
ITERATIONS=1

LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH \
  CUDA_VISIBLE_DEVICES=3 \
  python train_with_mesh.py \
  -s ${DATA_FOLDER}/colmap/ \
  -m ${DATA_FOLDER}/meshsplatting/ \
  --mesh_file ${DATA_FOLDER}/gold_dragon.glb \
  --indoor \
  --eval \
  --iterations ${ITERATIONS} \
  --fix_topology

python create_ply.py \
  ${DATA_FOLDER}/meshsplatting/point_cloud/iteration_${ITERATIONS}/ \
  --out ${DATA_FOLDER}/meshsplatting/point_cloud/iteration_${ITERATIONS}/mesh.ply
