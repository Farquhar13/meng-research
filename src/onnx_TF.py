import argparse
import onnx 
from onnx_tf.backend import prepare

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np

def main():
    """Main function"""
    
    # Option parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", action="store", help="input onnx model")
    parser.add_argument("--imgs", action="store", 
                        help="Comma separated list of images")
    args = parser.parse_args()

    # Convert onnx to TF
    model = onnx.load(args.onnx)

    print("preparing model")
    tf_rep = prepare(model)

    # load mnist using PyTorch
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True)
   
    # run the model
    output = tf_rep.run(np.asarray(test_loader, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
    
if __name__ == '__main__':
    main()
