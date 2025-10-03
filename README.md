# tdflow

# ⬇️ Installation
```
conda create -n tdflow python=3.11
conda activate tdflow
pip install -U "jax[cuda12]"
```

# Run experiments
```
python train.py \
    --dataset_name visual-scene-play-v0 \
    --save_dir {SAVE DIR}
```