from hofnet.examples import example_path
from hofnet.utils import prepare_data

# Example paths for CIFs and dataset
root_cifs = "./data/HOF_cif/cife"
root_dataset = "./data/HOF_cif/total"

# Run data preparation
prepare_data(root_cifs, root_dataset, downstream=None)