#### HOW to run


1. Folder location of the image dataset should be specified in config.py file. 
Update "dataset_folder" variable with the path to MSRC_ObjCategImageDatabase_v2 folder, which
is the parent folder of the "Images" sub_folder. This can either be:
    a. absolute path (eg. /home/user/xyz/documents/MSRC_ObjCategImageDatabase_v2), or
    b. or, relative path to the python scripts (eg. my/dataset/MSRC_ObjCategImageDatabase_v2)

2. run the following to install python packages
$ pip install -r requirements.txt  

3. Run the following command to run tests of a combination of all descriptors, distance measures, 
and quantizations:
    $ python3 cvpr_run_tests.py 
Output metrics and plots will be stored in a new output/ folder. This is a very long script since
it loops through all possible combinations

4. To run a a single set of descriptiors, distance measures and a single set of quantizations with 
and without PCA run:
    $ python3 cvpr_run_searches.py
The different quantization parameters can be updated in the Python dictionaries available in this script.
Output will be stored in output/ folder
