import sys
from pip._internal import main as pipmain

# Install dependencies for Python 3.11.3
pipmain(['install', 'pandas'])
pipmain(['install', 'numpy'])
pipmain(['install', 'torch==2.0.0'])
pipmain(['install', 'torchvision==0.15.1'])
pipmain(['install', 'protobuf==3.20.3'])
pipmain(['install', 'pytorch-lightning==2.0.2'])