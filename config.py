import os

dataset_folder = os.path.realpath('db/MSRC_ObjCategImageDatabase_v2')
dataset_images_subfolder = 'Images'
output_folder = os.path.realpath('output')
output_subfolder = 'descriptors/globalRGBhisto'
output_plots_subfolder = 'plots'
output_metrics_subfolder = 'metrics'

num_result_images_to_print = 10

top_n_results = 20  # for metrics calculation

# Choose whether to display plots or not. All plots are saved in file.
display_plots = False

