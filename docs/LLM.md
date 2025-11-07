# Requirements:

- Install Miniforge then:

```
conda create -n ellie-ml python=3.10 -y
conda activate ellie-ml
```

```
pip install --upgrade pip setuptools wheel
pip install "numpy>=1.22,<1.24" tensorflow-macos==2.12.0 tensorflow-metal==1.0.0 keras==2.12 tensorboard==2.12
pip uninstall -y jax jaxlib
```

To test:

```
python - << 'PY'
import tensorflow as tf
print("TF:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
PY
```

Should show:

```
TF: 2.12.0
GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```