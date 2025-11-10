import cvpr_execute_searches
from types import SimpleNamespace
import traceback


def run_gch_tests(use_pca=False):
    ### globarl_color_histogram config
    run_configs_gch = {
        'descriptor_type': 'global_color_histogram', 
        'descriptor_name': 'Global Color Histogram',
        'num_bins': 2,
        'use_pca': False,
        'pca_components': 1
    }

    run_configs_gch['use_pca'] = use_pca
    run_configs_dict = run_configs_gch
    for bin in range(2, 9, 2):
        run_configs_dict['num_bins'] = bin
        run_configs = SimpleNamespace(**run_configs_dict)
        cvpr_execute_searches.main(run_configs)


def run_spatial_gch_tests(use_pca=False):
    ### grid color histogram config
    run_configs_sgch = {
        'descriptor_type': 'grid_color_histogram',
        'descriptor_name': 'Spatial Grid Color Histogram',
        'num_bins': 4,
        'grid_rows': 4,
        'grid_cols': 4,
        'use_pca': False,
        'pca_components': 20
    }

    run_configs_sgch['use_pca'] = use_pca
    run_configs_dict = run_configs_sgch
    for bin in range(2, 9, 2):
        run_configs_dict['num_bins'] = bin
        for grid_size in range(2, 9, 2):
            run_configs_dict['grid_rows'] = grid_size
            run_configs_dict['grid_cols'] = grid_size
            run_configs = SimpleNamespace(**run_configs_dict)
            cvpr_execute_searches.main(run_configs)   


def run_eoh_tests(use_pca=False):
    ### eoh config
    run_configs_eoh = {
        'descriptor_type': 'eoh',
        'descriptor_name': 'Edge Orientation Histogram (EOH)',
        'eoh_bins': 2,
        'grid_rows': 2,
        'grid_cols': 2,
        'use_pca': False,
        'pca_components': 30
    }

    run_configs_eoh['use_pca'] = use_pca
    run_configs_dict = run_configs_eoh
    for bin in range(2, 9, 2):
        run_configs_dict['eoh_bins'] = bin
        for grid_size in range(2, 9, 2):
            run_configs_dict['grid_rows'] = grid_size
            run_configs_dict['grid_cols'] = grid_size
            run_configs = SimpleNamespace(**run_configs_dict)
            cvpr_execute_searches.main(run_configs)  


def run_eoh_plus_ch_tests(use_pca=False):

    ### eoh + grid color histogram config
    run_configs_eoh_plus_ch = {
            'descriptor_type': 'eoh_plus_ch',
            'descriptor_name': 'EOH + Grid Color Histogram',
            'eoh_bins': 2,
            'gch_bins': 2,
            'grid_rows': 2,
            'grid_cols': 2,
            # Weights for EOH and Color Histogram combination
            'weight_eoh': 0.5,
            'weight_color': 0.5,
            'use_pca': False,
            'pca_components': 1
        }

    run_configs_eoh_plus_ch['use_pca'] = use_pca
    run_configs_dict = run_configs_eoh_plus_ch
    for eoh_bin in range(2, 9, 2):
        run_configs_dict['eoh_bins'] = eoh_bin
        for gch_bin in range(2, 9, 2):
            run_configs_dict['gch_bins'] = gch_bin
            for grid_size in range(2, 9, 2):
                run_configs_dict['grid_rows'] = grid_size
                run_configs_dict['grid_cols'] = grid_size
                run_configs = SimpleNamespace(**run_configs_dict)
                cvpr_execute_searches.main(run_configs)


if __name__ == "__main__":

    try:
        # run_gch_tests()
        # run_spatial_gch_tests()
        # run_eoh_tests()
        run_eoh_plus_ch_tests()

    
    except Exception as e:
        print("An error occurred during execution:")
        traceback.print_exc()