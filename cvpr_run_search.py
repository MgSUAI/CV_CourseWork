import cvpr_execute_searches
from types import SimpleNamespace
import traceback


def call_searcher_with_configs(run_configs_dict):

    print(f"\nRunning search with configs: {run_configs_dict}")
    run_configs = SimpleNamespace(**run_configs_dict)
    cvpr_execute_searches.main(run_configs)


def run_gch_search():
    ### globarl_color_histogram config
    run_configs_gch = {
        'descriptor_type': 'global_color_histogram', 
        'descriptor_name': 'Global Color Histogram',
        'num_bins': 2,
        'use_pca': False,
        'pca_components':20
    }

    call_searcher_with_configs(run_configs_gch)



def run_spatial_gch_search():
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

    call_searcher_with_configs(run_configs_sgch)

                
def run_eoh_search():
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

    call_searcher_with_configs(run_configs_eoh)


def run_eoh_plus_ch_search():

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

    call_searcher_with_configs(run_configs_eoh_plus_ch)


if __name__ == "__main__":

    try:
        run_gch_search()
        # run_spatial_gch_search()
        # run_eoh_search()
        # run_eoh_plus_ch_search()
    
    except Exception as e:
        print("\n An error occurred during execution:")
        traceback.print_exc()