# Parameter Tuning GUI

This is an optional visualisation step, allowing viewers to interactively probe the effect of different parameters on the synthetic electroanatomic map output.

## Prerequisites 

Refer to the `synthetic_registration_error` [GitHub page](https://github.com/arnovonkietzell/synthetic_registration_error) for general instructions for this set of WIPs. Make sure to follow the steps on how to set the interpreter and root directory.

## Running this WIP

- The input to this WIP (`case_1`) should be an image-based cardiac surface mesh. If region-dependent noise or automatic clipping are desired, this should be the output of the `noise_region` or `calculate_boundary_distance` WIPs (run sequentially if both are desired).
- Running the WIP will open a window to a GUI allowing you to experiment with the following functionalities:
    - Mesh resampling using decimation or ACVD
    - Adding deformations at three scales (large-scale, small-scale, corruptive) with different parameters - optionally region-dependent
    - Automatic clipping of different boundaries at different displacements
    - Smoothing of the auto-clipped boundaries
    - Exporting the output mesh or chosen parameters to file 
- On closing the GUI window, a new case representing the final synthetic map will be added to EP Workbench