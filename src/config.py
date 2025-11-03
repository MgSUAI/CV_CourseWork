import os

dataset_folder = os.path.realpath('db/MSRC_ObjCategImageDatabase_v2')
dataset_images_subfolder = 'Images'
output_folder = os.path.realpath('output/descriptors')
output_subfolder = 'globalRGBhisto'

## global histogram config
num_bins = 4