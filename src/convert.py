'''
File: convert.py
Author: Collin Farquhar
Description: Converts a saved PyTorch model to onnx 
then converts from onnx to TensorFlow

Referencing the onnx tutorial at:
https://github.com/onnx/tutorials/blob/master/tutorials/PytorchTensorflowMnist.ipynb
'''

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchNet import Net

import onnx
from onnx_tf.backend import prepare

import numpy as np
from IPython.display import display
from PIL import Image


# Load the trained model from file
trained_model = Net()
trained_model.load_state_dict(torch.load('output_mnist.pth'))

# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 1, 28, 28)) # one black and white 28 x 28 picture will be the input to the model
torch.onnx.export(trained_model, dummy_input, "output_mnist.onnx")

def main():
    
    '''
    parser = argparse.ArgumentParser(description='model converter from PyTorch
            to TensorFlow using onnx ')
    parser.add_argument("net_class", help"")
    '''


    # Load the ONNX file
    model = onnx.load('output_mnist.onnx')

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
    print('Image 1:')
    img = Image.open('assert/two.png').resize((28, 28)).convert('L')
    display(img)
    output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
    print('The digit is classified as ', np.argmax(output))

    print('Image 2:')
    img = Image.open('assets/three.png').resize((28, 28)).convert('L')
    display(img)
    output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
    print('The digit is classified as ', np.argmax(output))

    
    tf_rep.export_graph('output_mnist.pb')

if __name__ == '__main__':
    main()
