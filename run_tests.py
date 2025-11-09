import main
from types import SimpleNamespace

## globarl_color_histogram config
run_configs = {
    'descriptor_type': 'eoh', #'global_color_histogram',
    'num_bins': 16,
    'distance_metric': 'euclidean'
}

## grid color histogram config
run_configs = {
    'descriptor_type': 'grid_color_histogram',
    'num_bins': 4,
    'grid_rows': 8,
    'grid_cols': 4,
    'distance_metric': 'euclidean'
}

run_configs = SimpleNamespace(**run_configs)

print(run_configs.descriptor_type)
print(run_configs.num_bins)
print(run_configs.distance_metric)

main.main(run_configs)