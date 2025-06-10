from google.cloud import storage
import xarray
import dataclasses
from gencast import gencast
from common import data_utils


task_config = gencast.TASK
def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))

gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "gencast/"

print("Loading data...")

# Get available datasets
dataset_file = 'source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc'

with gcs_bucket.blob(dir_prefix+f"dataset/{dataset_file}").open("rb") as f:
  example_batch = xarray.load_dataset(f).compute()

print("Data loaded successfully.")
print("\nShape of the loaded data (example_batch):")
for var_name, var_data in example_batch.data_vars.items():
    print(f"  {var_name}: {var_data.shape}")

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets
print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.removesuffix(".nc")).items()]))


print("Extracting inputs, targets, and forcings...")
train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", "12h"), # Only 1AR training.
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", f"{(example_batch.dims['time']-2)*12}h"), # All but 2 input frames.
    **dataclasses.asdict(task_config))

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)