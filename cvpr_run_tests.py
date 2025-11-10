import cvpr_execute_searches
from types import SimpleNamespace
import traceback


def call_searcher_with_configs(run_configs_dict):

    print(f"\nRunning search with configs: {run_configs_dict}")
    run_configs = SimpleNamespace(**run_configs_dict)
    cvpr_execute_searches.main(run_configs)


def run_gch_tests(use_pca=False, pca_components_list=[20, 40, 60, 80, 100]):
    ### globarl_color_histogram config
    run_configs_gch = {
        'descriptor_type': 'global_color_histogram', 
        'descriptor_name': 'Global Color Histogram',
        'num_bins': 2,
        'use_pca': False,
        'pca_components':20
    }

    run_configs_gch['use_pca'] = use_pca
    run_configs_dict = run_configs_gch
    bins_list = range(2, 21, 2)

    for bin in bins_list:
        run_configs_dict['num_bins'] = bin

        if use_pca:
            for pca_comp in pca_components_list:
                run_configs_dict['pca_components'] = pca_comp
                call_searcher_with_configs(run_configs_dict)

        else:
            call_searcher_with_configs(run_configs_dict)


def run_spatial_gch_tests(use_pca=False, pca_components_list=[20, 40, 60, 80, 100]):
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
    bins_list = range(2, 17, 2)
    grid_sizes_list = [4, 8, 12]

    for bin in bins_list:
        run_configs_dict['num_bins'] = bin
        for grid_size in grid_sizes_list:
            run_configs_dict['grid_rows'] = grid_size
            run_configs_dict['grid_cols'] = grid_size

            if use_pca:
                for pca_comp in pca_components_list:
                    run_configs_dict['pca_components'] = pca_comp
                    call_searcher_with_configs(run_configs_dict)

            else:
                call_searcher_with_configs(run_configs_dict)  


def run_eoh_tests(use_pca=False, pca_components_list=[20, 40, 60, 80, 100]):
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
    eoh_bins_list = [2, 4, 8, 12, 16, 20, 24]
    grid_sizes_list = [2, 4, 8, 12, 16]

    for bin in eoh_bins_list:
        run_configs_dict['eoh_bins'] = bin
        for grid_size in grid_sizes_list:
            run_configs_dict['grid_rows'] = grid_size
            run_configs_dict['grid_cols'] = grid_size

            if use_pca:
                for pca_comp in pca_components_list:
                    run_configs_dict['pca_components'] = pca_comp
                    call_searcher_with_configs(run_configs_dict)

            else:
                call_searcher_with_configs(run_configs_dict)


def run_eoh_plus_ch_tests(use_pca=False, pca_components_list=[20, 40, 60, 80, 100]):

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
    eoh_bins_list = [4, 8, 12, 16]
    gch_bins_list = [4, 8, 12, 16]
    grid_sizes_list = [4, 8, 12, 16] 
    weight_eoh_list = [0.2, 0.4, 0.5, 0.6, 0.8]

    for eoh_bin in eoh_bins_list:
        run_configs_dict['eoh_bins'] = eoh_bin
        for gch_bin in gch_bins_list:
            run_configs_dict['gch_bins'] = gch_bin
            for grid_size in grid_sizes_list:
                run_configs_dict['grid_rows'] = grid_size
                run_configs_dict['grid_cols'] = grid_size
                for weight_eoh in weight_eoh_list:
                    run_configs_dict['weight_eoh'] = weight_eoh
                    run_configs_dict['weight_color'] = 1.0 - weight_eoh

                    if use_pca:
                        for pca_comp in pca_components_list:
                            run_configs_dict['pca_components'] = pca_comp
                            call_searcher_with_configs(run_configs_dict)

                    else:
                        call_searcher_with_configs(run_configs_dict)


if __name__ == "__main__":

    try:
        run_gch_tests(use_pca=False)
        # run_spatial_gch_tests(False)
        # run_eoh_tests()
        # run_eoh_plus_ch_tests()

    
    except Exception as e:
        print("\n An error occurred during execution:")
        traceback.print_exc()