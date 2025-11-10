import cvpr_execute_searches
from types import SimpleNamespace


if __name__ == "__main__":

    ### globarl_color_histogram config
    run_configs_ch = {
        'descriptor_type': 'global_color_histogram', 
        'descriptor_name': 'Global Color Histogram',
        'num_bins': 4,
        'use_pca': False,
        'pca_components': 1
    }

    ### grid color histogram config
    run_configs_sgch = {
        'descriptor_type': 'grid_color_histogram',
        'descriptor_name': 'Spatial Grid Color Histogram',
        'num_bins': 4,
        'grid_rows': 4,
        'grid_cols': 4,
        'use_pca': True,
        'pca_components': 20
    }

    run_configs_eoh = {
        'descriptor_type': 'eoh',
        'descriptor_name': 'Edge Orientation Histogram (EOH)',
        'eoh_bins': 12,
        'grid_rows': 4,
        'grid_cols': 4,
        'use_pca': False,
        'pca_components': 30
    }

    run_configs_eoh_plus_ch = {
        'descriptor_type': 'eoh_plus_ch',
        'descriptor_name': 'EOH + Grid Color Histogram',
        'eoh_bins': 8,
        'gch_bins': 8,
        'grid_rows': 4,
        'grid_cols': 4,
        # Weights for EOH and Color Histogram combination
        'weight_eoh': 0.5,
        'weight_color': 0.5,
        'use_pca': True,
        'pca_components': 25
    }



    # for bin in range(2, 21, 2):
    #     # run_configs_ch['num_bins'] = bins
    #     run_configs = run_configs_ch
    #     run_configs['num_bins'] = bin
    #     run_configs = SimpleNamespace(**run_configs)
    #     cvpr_execute_searches.main(run_configs)
        

    run_configs = run_configs_ch
    run_configs = SimpleNamespace(**run_configs)
    cvpr_execute_searches.main(run_configs)

