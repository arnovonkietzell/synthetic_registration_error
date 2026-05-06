import numpy as np
import pandas as pd
from copy import deepcopy
from sre_utils.mesh_creators import closest_point_indices
from sklearn.metrics import f1_score, precision_score, recall_score

class RegistrationExperiment:
    """Orchestrates registration of a synthetic EAM mesh to an MRI mesh.

    Holds three mesh references that serve distinct roles in the pipeline:

    - ``mri_mesh_eval``: ground-truth evaluation mesh passed to
      :class:`RegistrationEvaluator` after registration.
    - ``mri_mesh_register``: the mesh actually passed to the registration
      algorithm (may differ in resolution from the eval mesh).
    - ``eam_mesh``: the unregistered EAM mesh (source for registration).

    After calling one of the ``register_*`` methods,
    ``eam_mesh_post_registration`` holds the transformed EAM mesh and can be
    passed to :class:`RegistrationEvaluator` for error and fibrosis metrics.
    """

    def __init__(self, mri_mesh_eval, mri_case_register, eam_case_register):
        """Initialise the experiment from pre-processed case objects.

        Args:
            mri_mesh_eval: PyVista mesh used as the ground-truth reference for
                evaluation.  Stored as a copy so downstream processing does not
                affect it.
            mri_case_register: OpenEP Case whose mesh is used as the
                registration target.  Must have an ``'IIR'`` field if
                fibrosis metrics are to be computed.
            eam_case_register: OpenEP Case whose mesh is the registration
                source (the EAM to be aligned).
        """
        self.mri_case_register = mri_case_register
        self.eam_case_register = eam_case_register

        self.mri_mesh_eval = mri_mesh_eval.copy()

        # Build registration meshes from their respective cases
        self.mri_mesh_register = mri_case_register.create_mesh()
        self.mri_mesh_register.point_data['IIR'] = self.mri_case_register.fields['IIR']
        self.eam_mesh = eam_case_register.create_mesh()

        self.eam_mesh_post_registration = None  # Populated by a register_* call

    def register_cpd(self, method, n_iter=50, prereg_rigid=False, deformable_params=(0.1, 0.1)):
        """Register the EAM mesh to the MRI mesh using Coherent Point Drift (CPD).

        Optionally applies a rigid pre-registration pass before the main
        registration step.  The result is stored in
        ``self.eam_mesh_post_registration``.

        Args:
            method (str): CPD variant for the main registration step, e.g.
                ``'rigid'``, ``'affine'``, or ``'deformable'``.
            n_iter (int): Number of optimisation steps for each CPD pass.
            prereg_rigid (bool): If ``True``, run a rigid CPD pass on the EAM
                points before the main registration.
            deformable_params (tuple[float, float]): ``(alpha, beta)``
                regularisation parameters passed to the deformable CPD solver.
        """

        from OpenEPGUI.view.mesh_tools import registration_ui

        mri_points = np.array(self.mri_mesh_register.points)
        eam_points = np.array(self.eam_mesh.points)

        # Optional rigid pre-registration to bring the meshes into rough alignment
        if prereg_rigid:
            pars = {
                    'pars': {
                        'method': 'rigid',
                        'n_steps': n_iter,
                        'kwargs': {}
                    }
            }
            w = registration_ui.CPDRegistrationWorker(pars)
            eam_points = w.run(conn=None, source_points=eam_points, target_points=mri_points)

        # Main CPD registration pass
        pars = {
                'pars': {
                    'method': method,
                    'n_steps': n_iter,
                    'kwargs': {
                        'alpha': deformable_params[0],
                        'beta': deformable_params[1],
                    }
                }
        }
        w = registration_ui.CPDRegistrationWorker(pars)
        transformed_points = w.run(conn=None, source_points=eam_points, target_points=mri_points)

        # Store the registered mesh with updated point positions
        eam_mesh_transformed = self.eam_mesh.copy()
        eam_mesh_transformed.points = transformed_points
        self.eam_mesh_post_registration = eam_mesh_transformed

    def register_landmarks(self, method, n_landmarks=6):
        """Register the EAM mesh to the MRI mesh using paired landmark points.

        Reads the last ``n_landmarks`` landmarks from each case, fits a
        transform of the requested type, applies it to a deep copy of the EAM
        case, and stores the resulting mesh in
        ``self.eam_mesh_post_registration``.

        Args:
            method (str): Transform type — ``'rigid'``, ``'similarity'``, or
                ``'affine'``.
            n_landmarks (int): Number of landmark pairs to use, taken from the
                end of each case's landmark list.
        """

        from OpenEPGUI.model.mesh_tools import LandmarkRegistrationModel

        # Take the last n_landmarks points from each case's landmark list
        landmarks_0 = self.mri_case_register.electric.landmark_points.points[-n_landmarks:]
        landmarks_1 = self.eam_case_register.electric.landmark_points.points[-n_landmarks:]

        model = LandmarkRegistrationModel()

        # Set exclusive boolean flags to select the transform type
        if method == 'rigid':
            model.rigid = True
            model.similarity = False
            model.affine = False

        elif method == 'similarity':
            model.similarity = True
            model.rigid = False
            model.affine = False

        elif method == 'affine':
            model.affine = True
            model.rigid = False
            model.similarity = False

        # Compute transform that maps EAM landmarks onto MRI landmarks
        transform_matrix = model.get_transform_matrix(landmarks_1, landmarks_0)
        case_1_transform = deepcopy(self.eam_case_register)
        case_1_transform.transform(transform_matrix)

        self.eam_mesh_post_registration = case_1_transform.create_mesh()

    def save_registered_eam(self, filepath):
        """Save the post-registration EAM mesh to disk.

        Args:
            filepath (str): Destination file path.  The format is inferred from
                the file extension (e.g. ``'.vtk'``, ``'.vtp'``).

        Raises:
            ValueError: If ``register_cpd`` or ``register_landmarks`` has not
                been called yet.
        """
        if self.eam_mesh_post_registration is not None:
            self.eam_mesh_post_registration.save(filepath)
        else:
            raise ValueError("EAM mesh has not been registered yet.")


class RegistrationEvaluator:
    """Evaluates the quality of a completed EAM-to-MRI registration.

    Takes the three mesh objects produced by :class:`RegistrationExperiment`
    after registration and provides methods for computing registration error
    and fibrosis classification metrics.

    Args:
        eam_mesh_post_registration: PyVista mesh of the EAM after registration.
        mri_mesh_eval: Ground-truth PyVista mesh used for error and fibrosis
            metric computation.
        mri_mesh_register: PyVista mesh that was used as the registration
            target; must carry an ``'IIR'`` point-data field for fibrosis
            metrics.
    """

    def __init__(self, eam_mesh_post_registration, mri_mesh_eval, mri_mesh_register):
        self.eam_mesh_post_registration = eam_mesh_post_registration
        self.mri_mesh_eval = mri_mesh_eval
        self.mri_mesh_register = mri_mesh_register

    def closest_reg_points(self):
        """Find the closest point on the MRI registration mesh for each post-registration EAM point.

        Returns:
            np.ndarray: Shape ``(n_eam_points, 3)`` array of 3-D coordinates,
            where each row is the nearest point on ``mri_mesh_register`` to the
            corresponding point in ``eam_mesh_post_registration``.
        """
        indices = closest_point_indices(self.eam_mesh_post_registration.points,
                                        self.mri_mesh_register.points)
        closest_points = self.mri_mesh_register.points[indices]
        return closest_points

    def calculate_registration_error(self):
        """Compute per-point registration error and attach it to the post-registration mesh.

        Calculates the Euclidean distance between each point on the evaluation
        MRI mesh and its corresponding projected point on the registration mesh,
        stores the result as ``'Registration Error'`` in
        ``eam_mesh_post_registration.point_data``, and returns it.

        Returns:
            np.ndarray: Shape ``(n_points,)`` array of per-point distances in
            the same units as the mesh coordinates.
        """
        projected_points = self.closest_reg_points()

        distances = np.linalg.norm(self.mri_mesh_eval.points - projected_points, axis=1)

        self.eam_mesh_post_registration.point_data['Registration Error'] = distances

        return distances

    def fibrosis_prediction_metrics(self, fibrosis_threshold=1.2, save_path=None):
        """Evaluate fibrosis classification accuracy against the ground-truth IIR field.

        Maps post-registration EAM points onto the MRI registration mesh,
        thresholds both predicted and ground-truth IIR values at
        ``fibrosis_threshold`` to produce binary fibrosis labels, then
        computes Dice coefficient, precision, and recall.

        A ``'Fibrosis Classification'`` field is attached to
        ``eam_mesh_post_registration`` using the encoding:
        TN=0, FP=1, FN=2, TP=3.

        Args:
            fibrosis_threshold (float): IIR value above which a point is
                classified as fibrotic.
            save_path (str, optional): If provided, writes results to a
                single-row CSV at this path.

        Returns:
            dict: ``{'Dice Coefficient': float, 'Precision': float,
            'Recall': float}``.
        """
        # Map EAM post-registration points to their nearest neighbours on the MRI mesh
        closest_points = closest_point_indices(self.eam_mesh_post_registration.points,
                                               self.mri_mesh_register.points)

        true_iir = self.mri_mesh_eval.point_data['IIR']
        pred_iir = self.mri_mesh_register.point_data['IIR'][closest_points]

        # Threshold IIR to produce binary fibrosis labels
        true_fibrosis = true_iir >= fibrosis_threshold
        pred_fibrosis = pred_iir >= fibrosis_threshold

        # Encode classification outcome: TN=0, FP=1, FN=2, TP=3
        classification_results = true_fibrosis * 2 + pred_fibrosis
        self.eam_mesh_post_registration.point_data['Fibrosis Classification'] = classification_results

        dice = f1_score(true_fibrosis, pred_fibrosis)
        precision = precision_score(true_fibrosis, pred_fibrosis)
        recall = recall_score(true_fibrosis, pred_fibrosis)

        results_dict = {
            'Dice Coefficient': dice,
            'Precision': precision,
            'Recall': recall
        }

        if save_path is not None:
            results_df = pd.DataFrame([results_dict])
            results_df.to_csv(save_path, index=False)

        return results_dict
