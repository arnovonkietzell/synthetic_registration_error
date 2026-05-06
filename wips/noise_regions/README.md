# Noise Regions

This is an optional pre-processing step,necessary if region-dependent noise is desired in the synthetic electroanatomic map.

## Prerequisites 

Refer to the `synthetic_registration_error` [GitHub page](https://github.com/arnovonkietzell/synthetic_registration_error) for general instructions for this set of WIPs. Make sure to follow the steps on how to set the interpreter and root directory.

## Running this WIP
- The input to this WIP (`case_1`) should be an image-based cardiac surface mesh, with regions of independent noise defined using EP Workbench's Region Selector tool.
- The output will be a WIP with an additional field `noise_region`, which will be used in creating a synthetic electroanatomic map later.