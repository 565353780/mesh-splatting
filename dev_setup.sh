cd ..
git clone --depth 1 https://github.com/rahul-goel/fused-ssim.git

pip install ninja pybind11

pip3 install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

pip install lpips plyfile mediapy opencv-python matplotlib \
  tqdm trimesh mmengine mmcv timm

cd fused-ssim
python setup.py install

cd ../mesh-splatting/submodules/diff-triangle-mesh-rasterization
python setup.py install

cd ../simple-knn
python setup.py install

cd ../effrdel
pip install -e .
