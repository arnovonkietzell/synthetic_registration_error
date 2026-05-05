import openep

from sre_utils import mesh_preprocessing as mp, mesh_creators as mc, experiment_handlers as eh

# Boilerplate to run in debug mode or in EP Workbench WIP environment
try:
    case = cases[case_1]
    debug = False
    root_name = case_1.rsplit("__", 1)[0]

except:
    debug = True
    root_dir = '/Users/s1807328/Desktop/'
    root_name = 'test'
    case = openep.load_openep_mat(f'{root_dir}/{root_name}.mat')

    #synthetic data parameters
    n_source_points=5000
    clip_distance=5.0
    large_amplitude=1.5
    large_spacing=10.0
    small_amplitude=0.25
    small_spacing=3.5
    corruptive_amplitude=0.25
    rotation_angle=45.0
    translation_distance=50.0
    scaling_range=1.2

    #registration parameters
    n_registration_points=7500
    reg_method = 'rigid CPD' # options: 'rigid landmark', 'affine landmark', 'rigid CPD', 'affine CPD', 'deformable CPD'

    #output parameters
    fibrosis_transfer_eval = False
    fibrosis_stats_path = ('path/to/fibrosis/stats.csv')
    return_intermediate_meshes = False

def main():

    #load in mesh and ensure noise_region field exists
    preprocessed_mesh = mp.mesh_from_case(case)
    preprocessed_mesh = mp.ensure_noise_region(preprocessed_mesh)

    #create data for registration
    synthetic_map, source_mri_mesh = mc.make_synthetic_map_and_source_mri_mesh(preprocessed_mesh,
                                                                              n_points=n_source_points,
                                                                              clip_distance=clip_distance,
                                                                              large_amplitude=large_amplitude,
                                                                              large_spacing=large_spacing,
                                                                              small_amplitude=small_amplitude,
                                                                              small_spacing=small_spacing,
                                                                              corruptive_amplitude=corruptive_amplitude,
                                                                              rotation_angle=rotation_angle,
                                                                              translation_distance=translation_distance,
                                                                              scaling_range=scaling_range)
    registration_mri_mesh = mc.make_registration_mesh(preprocessed_mesh,
                                                      n_points=n_registration_points)

    source_mri_case = mp.case_from_mesh(source_mri_mesh, name=f'{root_name}__source_mri')
    registration_mri_case = mp.case_from_mesh(registration_mri_mesh, name=f'{root_name}__registration_mri')
    synthetic_map_case = mp.case_from_mesh(synthetic_map, name=f'{root_name}__synthetic_map')

    # perform registration
    reg = eh.RegistrationExperiment(mri_mesh_eval=source_mri_mesh,
                                 mri_case_register=registration_mri_case,
                                 eam_case_register=synthetic_map_case)

    if reg_method == 'rigid landmark':
        reg.register_landmarks('similarity')
    elif reg_method == 'affine landmark':
        reg.register_landmarks('affine')
    elif reg_method == 'rigid CPD':
        reg.register_cpd('rigid')
    elif reg_method == 'affine CPD':
        reg.register_cpd('affine')
    elif reg_method == 'deformable CPD':
        reg.register_cpd('deformable, prereg_rigid=True')

    # evaluate results
    _ = reg.calculate_registration_error()
    if fibrosis_transfer_eval:
        _ = reg.fibrosis_prediction_metrics(save_path=fibrosis_stats_path)

    # create case from registered mesh
    registered_mesh = reg.eam_mesh_post_registration
    registered_case = mp.case_from_mesh(registered_mesh, name=f'{root_name}__registered_map')
    
    # output registered mesh as case, along with intermediate meshes if specified
    if debug:
        openep.io.writers.export_openep_mat(registered_case, f'{root_dir}/{root_name}__registered_map.mat')

        if return_intermediate_meshes:
            openep.io.writers.export_openep_mat(synthetic_map_case, f'{root_dir}/{root_name}__synthetic_map.mat')
            openep.io.writers.export_openep_mat(source_mri_case, f'{root_dir}/{root_name}__source_mri.mat')
            openep.io.writers.export_openep_mat(registration_mri_case, f'{root_dir}/{root_name}__registration_mri.mat')


    if debug == False:
            
        out_cases[f'{root_name}__registered_case'] = registered_case

        if return_intermediate_meshes:
            out_cases[f'{root_name}__synthetic_map'] = synthetic_map_case
            out_cases[f'{root_name}__source_mri'] = source_mri_case
            out_cases[f'{root_name}__registration_mri'] = registration_mri_case

if __name__ == "__main__":
    main()