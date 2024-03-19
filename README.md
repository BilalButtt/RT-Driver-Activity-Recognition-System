# RT-Driver-Activity-Recognition-System
FYP_Group_27

## 4. ASONE Installation
```shell
python3 -m venv .env
source .env/bin/activate

pip install numpy Cython
pip install cython-bbox asone onnxruntime-gpu==1.12.1
pip install typing_extensions==4.7.1
pip install super-gradients==3.1.3
# for CPU
pip install torch torchvision
# for GPU
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```
