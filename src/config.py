import os

dataset_folder = os.path.realpath('db/MSRC_ObjCategImageDatabase_v2')
dataset_images_subfolder = 'Images'
output_folder = os.path.realpath('output/descriptors')
output_subfolder = 'globalRGBhisto'

# global color histogram config
num_bins = 4

# grid color histogram config
grid_rows = 4
grid_cols = 4

# EOH config
eoh_bins = 4

# Weights for EOH and Color Histogram combination
w_eoh = 0.8
w_color = 0.2

# results print
num_result_images_to_print = 10