#!/usr/bin/env python
#-*- coding: utf-8 -*-
#pylint: disable=

'''
File: readable.py
Author: Collin Farquhar
Description: Creates a human text readable model representation
of an onnx model

Referencing the PyTorch tutorial at:
https://pytorch.org/docs/stable/onnx.html
'''

import argparse
import onnx

class OptionParser():
    def __init__(self):
        "User based option parser"
        self.parser = argparse.ArgumentParser(prog="PROG")
        self.parser.add_argument("--onnx", action="store",
            dest="onnx", default="", help="onnx model to convert to text")
        self.parser.add_argument("--txtout", action="store",
            dest="txtout", default="model.txt", 
            help="output of text representation of the model")

def run(onnx_model, txtout):
    """Creates readable text file from onnx model"""
    # Load the ONNX model
    model = onnx.load(onnx_model)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    with open(txtout, "w") as f:
        f.write(onnx.helper.printable_graph(model.graph))

def main():
    "Main function"
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    run(opts.onnx, opts.txtout)

if __name__ == '__main__':
    main()
