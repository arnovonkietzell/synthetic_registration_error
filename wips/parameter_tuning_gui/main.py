from sre_utils import mesh_preprocessing as mp, mesh_visualiser as mv, mesh_creators as mc
import sys
from PyQt5.QtWidgets import QApplication
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

    app = QApplication(sys.argv)

    # Create SyntheticEAM instance with the loaded mesh
    mesh = mp.mesh_from_case(case)
    mesh = mp.ensure_noise_region(mesh)
    synthetic_eam = mc.SyntheticEAM(mesh)

    # Create and show the visualiser
    window = mv.MeshVisualiser(synthetic_eam)
    window.show()

    # Run the application event loop (blocks until window is closed)
    exit_code = app.exec_()

    # After the window is closed, create a new case from the modified mesh
    synthetic_map = window.synthetic_eam.clipped_mesh

    if debug:
        new_name = 'test__synthetic_map'
        new_case = mp.case_from_mesh(synthetic_map, name=new_name)
        openep.io.writers.export_openep_mat(new_case, f'{root_dir}/{new_name}.mat')

    else:
        new_name = f'{case_1.rsplit("__", 1)[0]}__synthetic_map'
        new_case = mp.case_from_mesh(synthetic_map, name=new_name)
        out_cases[new_name] = new_case
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()