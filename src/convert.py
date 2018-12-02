#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=

'''
File: convert.py
Author: Collin Farquhar
Description: Converts a saved PyTorch model to onnx 
then converts from onnx to TensorFlow

Referencing the onnx tutorial at:
https://github.com/onnx/tutorials/blob/master/tutorials/PytorchTensorflowMnist.ipynb
'''

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

'''
will want to change this import to an argument, possible
The PyTorch class is required, but a solution without hardcoding would be
prefered
'''
from torchNet import Net

import onnx
from onnx_tf.backend import prepare

import numpy as np
from IPython.display import display
from PIL import Image

class OptionParser():
    def __init__(self):
        "User based option parser"
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--ptorch", action="store",
            dest="ptorch", default="", help="Input PyTorch file")
        self.parser.add_argument("--onnx", action="store",
            dest="onnx", default="", help="Output ONNX file")
        self.parser.add_argument("--fout", action="store",
            dest="fout", default="", help="Output TF file")
        self.parser.add_argument("--imgs", action="store",
            dest="imgs", default="", help="Comma separated list of images (for testing purposes)")


def run(ptorch, fonnx, tout, imgs=[]):
    # Load the trained model from file
    trained_model = Net()
    trained_model.load_state_dict(torch.load(ptorch))

    # Export the trained model to ONNX
    dummy_input = Variable(torch.randn(1, 1, 28, 28)) # one black and white 28 x 28 picture will be the input to the model
    torch.onnx.export(trained_model, dummy_input, fonnx)

    # Load the ONNX file
    model = onnx.load(fonnx)

    # Import the ONNX model to Tensorflow
    tf_rep = prepare(model)


    # Input nodes to the model
    print('inputs:', tf_rep.inputs)

    # Output nodes from the model
    print('outputs:', tf_rep.outputs)

    # All nodes in the model
    print('tensor_dict:')
    print(tf_rep.tensor_dict)

    
    # Run the model in TensorFlow
    for img in imgs:
        if not os.path.isfile(img):
            continue
        print('Image 1:')
        img = Image.open(img).resize((28, 28)).convert('L')
        display(img)
        output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
        print('The digit is classified as ', np.argmax(output))
    
    if tfout:
        tf_rep.export_graph(tout)

def main():
    "Main function"
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    imgs = opts.imgs.split(',')
    run(opts.ptorch, opts.onnx, opts.fout, imgs)

if __name__ == '__main__':
    main()
