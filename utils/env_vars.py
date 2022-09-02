import os

def get_imagenet_root():
    return os.environ.get('IMAGENET_ROOT', '/d1/dataset/ILSVRC2012/')