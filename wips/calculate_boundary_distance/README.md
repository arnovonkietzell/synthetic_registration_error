# Calculate Boundary Distance

This is an optional pre-processing step, necessary if automatic clipping of mesh boundaries is desired.

## Prerequisites 

Refer to the `synthetic_registration_error` [GitHub page](https://github.com/arnovonkietzell/synthetic_registration_error) for general instructions for this set of WIPs. Make sure to follow the steps on how to set the interpreter and root directory.

## Running this WIP
- The input to this WIP (`case_1`) should be an image-based cardiac surface mesh, with boundaries (e.g. at the mitral valve or appendage) selected using EP Workbench's Region Selector tool.
- The output will be a WIP with additional fields `<boundary_number>_boundary_distance` for each boundary, which will be used for automatic clipping later.

