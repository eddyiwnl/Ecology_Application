#!/usr/bin/env python3

import json
import argparse
import numpy as np
import pandas as pd
import Model.modelLogic as modelLogic
import ImageDataset
from torch.utils.data import DataLoader
import torch

parser = argparse.ArgumentParser(
    prog='modelExecutable.py',
    description='Use this to run your model',
    epilog='')

parser.add_argument('modelCheckpoint', type=str,
                    help='A checkpoint file for the model described by modelLogic.py')
parser.add_argument('images', metavar='M', type=str, nargs='+',
                    help='A list of images to be processed by the model')
parser.add_argument('-B', dest='BATCHSIZE', type=int, default=4,
                    help='Specify a batch size to be used for image processing and prediction (default is 4)')
parser.add_argument('-o', dest='outFile', type=str, default='model_output.json',
                    help="Specify the output filename for prediction results (default is 'model_output.json')")
args = parser.parse_args()

checkpoint_path = args.modelCheckpoint
image_files = args.images
BATCHSIZE = args.BATCHSIZE
output_filename = args.outFile

# Set the random seeds for reproducibility
torch.manual_seed(9)
np.random.seed(9)

model = modelLogic.Model.load_from_checkpoint(
    checkpoint_path, map_location=modelLogic.device)
model.eval()

weights = modelLogic.weights
transformations = ImageDataset.Transformations(weights)
image_ds = ImageDataset.ImageDataset(image_files, transformations)

# This handles the batching and loading of images via the ImageDataset class
image_dl = DataLoader(dataset=image_ds,
                      batch_size=BATCHSIZE,
                      shuffle=False,
                      collate_fn=ImageDataset.collate_double)

# model_predictions: list of dicts where all boxes, labels, and scores
image_names, model_predictions = model(image_dl)

# format the model predictions for and output to the file
with open(output_filename, 'w') as file:
    output = {}
    for i in range(len(image_names)):
        image_name = image_names[i]
        pred = model_predictions[i]
        output[image_name] = {'predictions': {
            'pred_boxes': pred["boxes"].tolist(),
            'pred_labels': pred["labels"].tolist(),
            'pred_scores': pred["scores"].tolist()
        }}
    json.dump(output, file, indent=2)

print(f'\nModel predictions successfull: {output_filename}\n')
exit(0)
