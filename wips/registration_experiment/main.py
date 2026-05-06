import openep

from sre_utils import mesh_preprocessing as mp, experiment_handlers as eh

# Boilerplate to run in debug mode or in EP Workbench WIP environment
try:
    source_mri_case = cases[case_1]
    registration_mri_case = cases[case_2]
    synthetic_map = cases[case_3]
    debug = False

except:
    debug = True
    root_dir = '/Users/s1807328/Desktop/'
    source_mri_case = openep.load_openep_mat(f'{root_dir}/test__source_mri.mat')
    registration_mri_case = openep.load_openep_mat(f'{root_dir}/test__registration_mri.mat')
    synthetic_map = openep.load_openep_mat(f'{root_dir}/test__synthetic_map.mat')

    #output parameters
    fibrosis_transfer_eval = False
    fibrosis_stats_path = ('path/to/fibrosis/stats.csv')

def main():

    # cases to meshes
    source_mri_mesh = mp.mesh_from_case(source_mri_case)
    registration_mri_mesh = mp.mesh_from_case(registration_mri_case)
    synthetic_map_mesh = mp.mesh_from_case(synthetic_map)

    # evaluate registration
    reg = eh.RegistrationEvaluator(mri_mesh_eval=source_mri_mesh,
                                   mri_mesh_register=registration_mri_mesh,
                                   eam_mesh_post_registration=synthetic_map_mesh)

    _ = reg.calculate_registration_error()
    if fibrosis_transfer_eval:
        _ = reg.fibrosis_prediction_metrics(save_path=fibrosis_stats_path)
    
    # output registered mesh as case, along with intermediate meshes if specified
    if debug:
        new_name = 'test__map_with_errors'
        registered_case = mp.case_from_mesh(reg.eam_mesh_post_registration, name=new_name)
        openep.io.writers.export_openep_mat(registered_case, f'{root_dir}/{new_name}.mat')

    else:
        new_name = f'{case_1.rsplit("__", 1)[0]}__map_with_errors'
        registered_case = mp.case_from_mesh(reg.eam_mesh_post_registration, name=new_name)
        out_cases[new_name] = registered_case

if __name__ == "__main__":
    main()