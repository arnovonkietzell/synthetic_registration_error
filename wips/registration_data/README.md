# Registration Data

This step creates three meshes necessary for evaluation of Synthetic Registration Error: a synthetic map, an image source mesh, and an image registration mesh.

## Prerequisites 

Refer to the `synthetic_registration_error` [GitHub page](https://github.com/arnovonkietzell/synthetic_registration_error) for general instructions for this set of WIPs. Make sure to follow the steps on how to set the interpreter and root directory.

## Running this WIP

- The input to this WIP (`case_1`) should be an image-based cardiac surface mesh. If automatic clipping is desired, this should be the output of the `calculate_boundary_distance` WIP. Region-dependent noise is not currently implemented in this WIP.
- You can select a number of parameters altering the appearance of the synthetic map. To gain an understanding of the effect of these parameters, use the `parameter_tuning_gui` WIP.
- The output will be three new meshes: a synthetic map, an image source mesh, and an image registration mesh.