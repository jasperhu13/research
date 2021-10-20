#!/bin/bash
#run pretrained MVIT on imagenet
pip install -U torch torchvision cython
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install pyyaml==5.1
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
git clone https://github.com/facebookresearch/slowfast
export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH
sed -i '/"PIL"/d' slowfast/setup.py
python slowfast/setup.py build develop
sed -i 's|import simplejson|import json as simplejson|' slowfast/slowfast/utils/logging.py
cd slowfast
mv ../mvit_model.py .
mv ../validate.py .
python3 validate.py
