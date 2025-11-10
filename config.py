import os

dataset_folder = os.path.realpath('db/MSRC_ObjCategImageDatabase_v2')
dataset_images_subfolder = 'Images'
output_folder = os.path.realpath('output')
output_subfolder = 'descriptors/globalRGBhisto'
output_plots_subfolder = 'plots'
output_metrics_subfolder = 'metrics'

num_result_images_to_print = 10

# Choose whether to display plots or not. All plots are saved in file.
display_plots = False

distance_metric = 'euclidean'  # Options: 'manhattan', 'euclidean', 'mahalanobis', 'chi_squared'

################################################
# global color histogram config
# descriptor_type = 'global_color_histogram'
# num_bins = 8
################################################

# #################################################
# # grid color histogram config
# descriptor_type = 'grid_color_histogram'
# num_bins = 8
# grid_rows = 4
# grid_cols = 4
# #################################################

# # #################################################
# # EOH config
# descriptor_type = 'eoh'
# eoh_bins = 12
# grid_rows = 8
# grid_cols = 8
# # #################################################

# ################################################
# # EOH + Grid Color Histogram config
# descriptor_type = 'eoh_plus_ch'
# eoh_bins = 8
# gch_bins = 8
# ch_bins = 4
# grid_rows = 4
# grid_cols = 4
# # Weights for EOH and Color Histogram combination
# weight_eoh = 0.5  
# weight_color = 0.5
# #################################################


#################################################
# use_pca = True
# pca_components = 10
#################################################
