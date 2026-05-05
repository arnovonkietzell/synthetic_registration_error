# Synthetic Registration Error

Code to support the creation of synthetic electroanatomic mapping data, and using these synthetic maps to calculate Synthetic Registration Error (SRE). This is a source library of utilities, execution is via WIPs available at EP Workbench's [Work-In-Progress marketplace](link): 

## Pre-requisites

- **Conda** (or your preferred Python environment) is installed.
- The [OpenEP library](link) and its dependencies are installed in your environment. To install `openep` and its dependencies in a new environment name `sre`:

```bash
conda create -n sre python=3.10 pip
conda activate sre

git clone git@github.com:ecci-cvs/openep-py.git
cd openep-py
python -m pip install -e .
```

## How to Install a WIP

1. Ensure all pre-requisites are installed on your computer (see above).

2. Clone this repository:

```bash
git clone <repository-url>
```
   
3. Go to the directory you just cloned and do the following:

```bash
pip install -r requirements.txt

```
Your conda environment is now ready!

## Defining boundaries for clipping 

This is an optional preprocessing step if automatic clipping of mesh boundaries is desired, e.g. to test the effect of boundary inconsistency on registration. The input should be a cardiac surface `.vtk` file, loaded into EP Workbench. Use EP Workbench's `Region selector` tool, together with the `calculate_boundary_distance` WIP, to define boundaries and calculate distances from these boundaries (used for automatic clipping at any given distance). More details [here](link).

## Defining noise regions

This is an optional preprocessing step if region-dependent noise is desired, e.g. a larger noise amplitude in the appendage to simulate incomplete mapping. The input should be a cardiac surface `.vtk` file, loaded into EP Workbench. Use EP Workbench's `Region selector` tool, together with the `noise_regions` WIP, to define regions for specifiying independent noise fields. More details [here](link).

## Visualising synthetic data creation 

This step opens a GUI visualising the creation of synthetic data, and the effect of different parameters. It allows tuning of synthetic data parameters by visually comparing synthetic electroanatomic maps with real maps, possibly from the same patient. The input should be a cardiac surface mesh, loaded into EP Workbench. If automatic boundary clipping and/or region-dependent noise is desired, this mesh should have been preprocessed as described above. Use the `parameter_tuning_GUI` WIP to interactively create a synthetic map from this mesh. More details [here](link).

## Running registration experiment 

This allows end-to-end calculation of a Synthetic Registration Error field from a cardiac surface. The input should be a cardiac surface mesh, loaded into EP Workbench. If automatic boundary clipping and/or region-dependent noise is desired, this mesh should have been preprocessed as described above. Also, if fibrosis transfer metrics are to be quantified, the mesh should contain an 'IIR' field as point data. Use the `registration_experiment` WIP to create synthetic mapping data, register this to the original geometry, and calculate Synthetic Registration Error (saved as a field on the registered mesh). More details [here](link).

