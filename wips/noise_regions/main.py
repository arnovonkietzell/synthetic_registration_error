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

    #create point data for the noise regions by sampling the cell data
    new_mesh = mesh.sample(mesh)
    new_mesh.point_data['noise_region'] = new_mesh.point_data['cell_region']
    new_mesh.point_data.remove('cell_region')
    new_mesh.cell_data.remove('cell_region')

    if debug:
        new_name = 'test__noise_regions'
        new_case = mp.case_from_mesh(new_mesh, name=new_name)
        openep.io.writers.export_openep_mat(new_case, f'{root_dir}/{new_name}.mat')

    else:
        new_name = f'{case_1.rsplit("__", 1)[0]}__noise_regions'
        new_case = mp.case_from_mesh(new_mesh, name=new_name)
        out_cases[new_name] = new_case

if __name__ == "__main__":
    main()