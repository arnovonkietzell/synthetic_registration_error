# Synthetic Registration Error

Code to support the creation of synthetic electroanatomic mapping data, and using these synthetic maps to calculate Synthetic Registration Error (SRE). This repository contains both a source library of utilities `sre_utils` and a collection `wips` of WIP modules designed to be run in the EP Workbench software. 

## Pre-requisites

- **Conda** (or your preferred Python environment) is installed.
- The [OpenEP library](https://github.com/ecci-cvs/openep-py) and its dependencies are installed in your environment:

```bash
conda create -n sre python=3.10 pip
git clone https://github.com/ecci-cvs/openep-py
cd openep-py
python -m pip install -e .
```
- (Recommended) EP Workbench, minimal version `v1.1.0-beta.1-260505`. Download instructions available [here](https://openep.discourse.group/t/downloading-ep-workbench-beta-for-academic-use/149).

## How to run these WIPs

1. Ensure all pre-requisites are installed on your computer (see above).

2. Clone this repository and install requirements:

```bash
conda activate sre
git clone https://github.com/arnovonkietzell/synthetic_registration_error
cd synthetic_registration_error
pip install -r requirements.txt
which python #save this env python path for later
```

3. Running a WIP through EP Workbench:

- Open EP Workbench and navigate to Work-in-Progress > Marketplace.
- Select your WIP and press "clone".
- On the WIP Bar, select Info > Edit. 
- Set Interpreter as the path to your python environment (from step 2), and Root Dir as the path to `synthetic_registration_error` (this repository). Save these settings and close the WIP editor window.
- You are now ready to run the WIP - check each WIP for individual instructions.

4. Running a WIP outside EP Workbench:

It is possible to run these WIPs in `debug` mode outside EP Workbench. However, any workflows relying on graphical interaction with the data (e.g. selection boundaries to clip, landmark points for registration) require EP Workbench. To run in `debug` mode requires you to open the WIP's `main.py` code and edit `root_dir` to your preferred input/output directory. Then:

```bash
conda activate sre
cd synthetic_registration_error
python -m wips.<wip_name>.main
```

## Overview of WIPs ##
Here is a brief overview of the WIPs included in this repository. For more information refer to the individual WIP `README`s.

- `noise_regions`: This is an optional pre-processing step, necessary if region-dependent noise is desired in the synthetic electroanatomic map.
- `calculate_boundary_distance`: This is an optional pre-processing step, necessary if automatic clipping of mesh boundaries is desired.
- `parameter_tuning_gui`: This is an optional visualisation step, allowing viewers to interactively probe the effect of different parameters on the synthetic electroanatomic map output.
- `registration_data`: This step creates three meshes necessary for evaluation of Synthetic Registration Error: a synthetic map, an image source mesh, and an image registration mesh.
- `registration_eval`: Following registration of the synthetic map with the image registration mesh using EP Workbench, this step calculates the Synthetic Registration Error field, and optionally evaluates the accuracy of fibrosis transfer.