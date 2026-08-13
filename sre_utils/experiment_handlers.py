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

    Registration is performed by one of ``register_cpd`` (Coherent Point
    Drift), ``register_icp`` (Iterative Closest Point), or
    ``register_landmarks`` (paired landmark points), all backed by
    :mod:`openep.case.registration`. After calling one of these
    ``register_*`` methods, ``eam_mesh_post_registration`` holds the
    transformed EAM mesh and ``registration_transform`` holds the fitted
    :class:`openep.case.transforms.Transform`; the mesh can be passed to
    :class:`RegistrationEvaluator` for error and fibrosis metrics.
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
        self.registration_transform = None  # openep.case.transforms.Transform fitted by the last register_* call

    def register_cpd(
        self,
        method,
        n_iter=50,
        prereg_rigid=False,
        deformable_params=(0.1, 0.1),
        verbose=False,
        decimate=False,
        decimate_n_points=1000,
        decimate_method='pyvista',
    ):
        """Register the EAM mesh to the MRI mesh using Coherent Point Drift (CPD).

        Optionally applies a rigid pre-registration pass before the main
        registration step.  The result is stored in
        ``self.eam_mesh_post_registration``.

        Args:
            method (str): CPD variant for the main registration step, e.g.
                ``'rigid'``, ``'similarity'``, ``'affine'``, or ``'deformable'``.
            n_iter (int): Number of optimisation steps for each CPD pass.
            prereg_rigid (bool): If ``True``, run a rigid CPD pass on the EAM
                points before the main registration.
            deformable_params (tuple[float, float]): ``(alpha, beta)``
                regularisation parameters passed to the deformable CPD solver.
                Ignored unless ``method == 'deformable'``.
            verbose (bool): If ``True``, print iteration progress for each
                CPD pass.
            decimate (bool): If ``True``, fit each CPD pass on decimated
                copies of the EAM/MRI meshes instead of their full-resolution
                points, then apply the fitted transform back to the
                full-resolution EAM mesh. Much faster on dense meshes - see
                :class:`openep.case.registration.CPDRegistration`.
            decimate_n_points (int): Target point count for the decimated
                working copies. Ignored unless ``decimate=True``.
            decimate_method (str): Decimation method to use - see
                :func:`openep.mesh.decimation.decimate_mesh`. Ignored unless
                ``decimate=True``.
        """

        from openep.case.registration import CPDRegistration

        mri_mesh = self.mri_mesh_register
        eam_mesh = self.eam_mesh
        decimate_kwargs = dict(
            decimate=decimate,
            decimate_n_points=decimate_n_points,
            decimate_method=decimate_method,
        )

        # Optional rigid pre-registration to bring the meshes into rough alignment
        if prereg_rigid:
            prereg_transform = CPDRegistration(
                source_points=eam_mesh.points,
                target_points=mri_mesh.points,
                method='rigid',
                n_iterations=n_iter,
                source_mesh=eam_mesh,
                target_mesh=mri_mesh,
                progress_callback=self._cpd_progress_printer('prereg-rigid', n_iter) if verbose else None,
                **decimate_kwargs,
            ).run()

            eam_mesh = eam_mesh.copy()
            eam_mesh.points = prereg_transform.apply(eam_mesh.points)

        # Main CPD registration pass, fit on the (possibly pre-registered) EAM mesh
        cpd_kwargs = {'alpha': deformable_params[0], 'beta': deformable_params[1]} if method == 'deformable' else {}
        transform = CPDRegistration(
            source_points=eam_mesh.points,
            target_points=mri_mesh.points,
            method=method,
            n_iterations=n_iter,
            source_mesh=eam_mesh,
            target_mesh=mri_mesh,
            progress_callback=self._cpd_progress_printer(method, n_iter) if verbose else None,
            **decimate_kwargs,
            **cpd_kwargs,
        ).run()

        self._apply_registration_transform(transform, base_mesh=eam_mesh)

    @staticmethod
    def _cpd_progress_printer(stage, n_iter):
        """Build a `progress_callback` for `CPDRegistration` that prints iteration progress."""

        def callback(iteration, source_points):
            print(f"[CPD:{stage}] iteration {iteration}/{n_iter}")

        return callback

    def register_icp(self, method='rigid', max_iterations=100, max_correspondence_distance=None):
        """Register the EAM mesh to the MRI mesh using Iterative Closest Point (ICP).

        The result is stored in ``self.eam_mesh_post_registration``.

        Args:
            method (str): ``'rigid'`` (translation and rotation only) or
                ``'similarity'`` (translation, rotation, and isotropic
                scaling).
            max_iterations (int): Maximum number of ICP iterations.
            max_correspondence_distance (float, optional): Maximum distance
                between a source/target point pair for it to be treated as a
                correspondence. Defaults to 10% of the MRI mesh's
                bounding-box diagonal (see
                :class:`openep.case.registration.ICPRegistration`).
        """

        from openep.case.registration import ICPRegistration

        mri_points = np.array(self.mri_mesh_register.points)
        eam_points = np.array(self.eam_mesh.points)

        transform = ICPRegistration(
            source_points=eam_points,
            target_points=mri_points,
            method=method,
            max_iterations=max_iterations,
            max_correspondence_distance=max_correspondence_distance,
        ).run()

        self._apply_registration_transform(transform)

    def _apply_registration_transform(self, transform, base_mesh=None):
        """Apply a fitted Transform to a copy of ``base_mesh`` and store the result.

        Args:
            transform (openep.case.transforms.Transform): Transform mapping
                ``base_mesh`` points onto the MRI registration mesh.
            base_mesh (pyvista.PolyData, optional): Mesh whose points
                ``transform`` was fitted to map. Defaults to ``self.eam_mesh``
                - pass an intermediate mesh here if an earlier pass (e.g. a
                CPD rigid pre-registration) has already been applied ahead of
                ``transform``.
        """

        base_mesh = self.eam_mesh if base_mesh is None else base_mesh
        eam_mesh_transformed = base_mesh.copy()
        eam_mesh_transformed.points = transform.apply(eam_mesh_transformed.points)

        self.registration_transform = transform
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
        from openep.case.registration import LandmarkRegistration

        # Take the last n_landmarks points from each case's landmark list
        landmarks_0 = self.mri_case_register.electric.landmark_points.points[-n_landmarks:]
        landmarks_1 = self.eam_case_register.electric.landmark_points.points[-n_landmarks:]

        # Fit the transform that maps EAM landmarks onto MRI landmarks
        transform = LandmarkRegistration(
            source_points=landmarks_1,
            target_points=landmarks_0,
            method=method,
        ).run()

        case_transformed = deepcopy(self.eam_case_register)
        case_transformed.transform(transform)

        self.registration_transform = transform
        self.eam_mesh_post_registration = case_transformed.create_mesh()

    def save_registered_eam(self, filepath):
        """Save the post-registration EAM mesh to disk.

        Args:
            filepath (str): Destination file path.  The format is inferred from
                the file extension (e.g. ``'.vtk'``, ``'.vtp'``).

        Raises:
            ValueError: If none of the ``register_*`` methods has been
                called yet.
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

    def fibrosis_prediction_metrics(self, fibrosis_thresholds=(1.2,), save_path=None):
        """Evaluate fibrosis classification accuracy against the ground-truth IIR field.

        Maps post-registration EAM points onto the MRI registration mesh, then
        for each threshold in ``fibrosis_thresholds`` thresholds both predicted
        and ground-truth IIR values to produce binary fibrosis labels and
        computes Dice coefficient, precision, and recall.

        ``'True IIR'`` and ``'Pred IIR'`` fields (the raw, unthresholded
        values) are attached to ``eam_mesh_post_registration`` to allow
        post-hoc analysis at arbitrary thresholds. A per-threshold
        ``'Fibrosis Label (IIR>{threshold})'`` field is also attached
        using the encoding: TN=0, FP=1, FN=2, TP=3.

        Args:
            fibrosis_thresholds (Sequence[float]): IIR values above which a
                point is classified as fibrotic. One set of metrics is
                computed per threshold.
            save_path (str, optional): If provided, writes results to a CSV
                at this path with one row per threshold.

        Returns:
            dict: Keyed by threshold, each value a dict of
            ``{'Dice Coefficient': float, 'Precision': float,
            'Recall': float}``.
        """
        # Map EAM post-registration points to their nearest neighbours on the MRI mesh
        closest_points = closest_point_indices(self.eam_mesh_post_registration.points,
                                               self.mri_mesh_register.points)

        true_iir = self.mri_mesh_eval.point_data['IIR']
        pred_iir = self.mri_mesh_register.point_data['IIR'][closest_points]

        self.eam_mesh_post_registration.point_data['True IIR'] = true_iir
        self.eam_mesh_post_registration.point_data['Pred IIR'] = pred_iir
        if 'IIR' in self.eam_mesh_post_registration.point_data:
            self.eam_mesh_post_registration.point_data.remove('IIR')

        results_by_threshold = {}
        for fibrosis_threshold in fibrosis_thresholds:
            # Threshold IIR to produce binary fibrosis labels
            true_fibrosis = true_iir >= fibrosis_threshold
            pred_fibrosis = pred_iir >= fibrosis_threshold

            # Encode classification outcome: TN=0, FP=1, FN=2, TP=3
            classification_results = true_fibrosis * 2 + pred_fibrosis
            self.eam_mesh_post_registration.point_data[
                f'Fibrosis Label (IIR>{fibrosis_threshold})'
            ] = classification_results

            dice = f1_score(true_fibrosis, pred_fibrosis)
            precision = precision_score(true_fibrosis, pred_fibrosis)
            recall = recall_score(true_fibrosis, pred_fibrosis)

            results_by_threshold[fibrosis_threshold] = {
                'Dice Coefficient': dice,
                'Precision': precision,
                'Recall': recall
            }

        if save_path is not None:
            results_df = pd.DataFrame([
                {'IIR Threshold': threshold, **metrics}
                for threshold, metrics in results_by_threshold.items()
            ])
            results_df.to_csv(save_path, index=False)

        return results_by_threshold
