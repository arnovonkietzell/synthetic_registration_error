# Registration Eval

Following registration of the synthetic map with the image registration mesh using EP Workbench, this step calculates the Synthetic Registration Error field, and optionally evaluates the accuracy of fibrosis transfer.

## Prerequisites 

Refer to the `synthetic_registration_error` [GitHub page](https://github.com/arnovonkietzell/synthetic_registration_error) for general instructions for this set of WIPs. Make sure to follow the steps on how to set the interpreter and root directory.

## Running this WIP

- The inputs to this WIP (`case_1`, `case_2`, `case_3`) should be the outputs from the `registration_data` WIP: a synthetic map, an image source mesh, and an image registration mesh. The synthetic map should have been registered with the image registration mesh using EP Workbench's registration tool.
- You can choose whether to calculate metrics quantifying the accuracy of fibrosis transfer. In this case, the input meshes must contain a `point_data` field `IIR`. The output will be to a `.csv` file at the specified path.
- The output of the WIP will be the synthetic map, with a new field `Registration Error`. If `fibrosis_transfer_eval` was ticked, it will also contain a field `Fibrosis Classification` showing where fibrosis was transferred correctly: True Negative = 0, False Positive = 1, False Negative = 2, True Positive = 3. 
