from sre_utils import mesh_preprocessing as mp
import openep

# Boilerplate to run in debug mode or in EP Workbench WIP environment
try:
    case = cases[case_1]
    debug = False
except:
    root_dir = '/Users/s1807328/Desktop/'
    case = openep.load_openep_mat(f'{root_dir}/test.mat')
    debug = True

def main():

    #read in the mesh from the case
    mesh = mp.mesh_from_case(case)

    #assign distance fields to the mesh
    new_mesh = mp.assign_distance_fields(mesh)

    if debug:
        new_name = 'test__boundary_distance'
        new_case = mp.case_from_mesh(new_mesh, name=new_name)
        openep.io.writers.export_openep_mat(new_case, f'{root_dir}/{new_name}.mat')

    else:
        new_name = f'{case_1.rsplit("__", 1)[0]}__boundary_distance'
        new_case = mp.case_from_mesh(new_mesh, name=new_name)
        out_cases[new_name] = new_case

if __name__ == "__main__":
    main()