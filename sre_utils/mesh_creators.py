import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RBFInterpolator, splprep, splev
import pyacvd


def sample_points(mesh, target_points=None, target_resolution=None, fields='all'):
    """Reduce mesh density to a target point count, preserving selected fields.

    Either ``target_points`` or ``target_resolution`` must be provided.  When
    ``target_resolution`` is given it is converted to an equivalent point count
    using the mesh surface area.  If the target already exceeds the current
    point count, no decimation is performed.  Fields are transferred to the
    decimated mesh by nearest-neighbour lookup.

    Args:
        mesh: PyVista mesh to decimate.
        target_points (int, optional): Desired number of vertices after decimation.
        target_resolution (float, optional): Desired average inter-vertex spacing
            in millimetres; converted to ``target_points`` internally.
        fields (str or list, optional): Point-data field names to transfer.
            Pass ``'all'`` (default) to transfer every field.

    Returns:
        pyvista.PolyData: Decimated mesh with the requested fields preserved.
    """

    if target_resolution is not None:
        # Convert spatial resolution (mm) to target point count based on mesh area
        target_points = mesh.area / (target_resolution ** 2)

    remove_fraction = 1 - (target_points / mesh.n_points)

    if remove_fraction < 0 or remove_fraction >= 1:
        # No decimation needed if target exceeds current point count
        sampled_mesh = mesh
    else:
        sampled_mesh = mesh.decimate(remove_fraction)

        # Preserve given point data fields by mapping from original mesh
        idx = closest_point_indices(sampled_mesh.points, mesh.points)

        if fields == 'all':
            fields = mesh.point_data.keys()

        for field in fields:
            sampled_mesh[field] = mesh[field][idx]

    return sampled_mesh


def remesh_pyacvd(mesh, target_resolution=None, target_points=None, fields='all'):
    """Remesh to a uniform point distribution using ACVD clustering.

    Unlike simple decimation, ACVD produces a more isotropic point distribution
    by iterative clustering.  A single subdivision pass is applied before
    clustering to ensure sufficient resolution for the target point count.
    Fields are transferred to the new mesh by nearest-neighbour lookup.

    Args:
        mesh: PyVista mesh to remesh.
        target_resolution (float, optional): Desired average inter-vertex spacing
            in millimetres; converted to ``target_points`` internally.
        target_points (int, optional): Desired number of vertices.
        fields (str or list, optional): Point-data field names to transfer.
            Pass ``'all'`` (default) to transfer every field.

    Returns:
        pyvista.PolyData: Remeshed mesh with the requested fields preserved.
    """
    if target_resolution is not None:
        # Convert spatial resolution (mm) to target point count based on mesh area
        target_points = int(mesh.area / (target_resolution ** 2))

    # Create a copy to avoid modifying the original mesh
    mesh_copy = mesh.copy()

    clus = pyacvd.Clustering(mesh_copy)
    clus.subdivide(1) # Subdivide once to ensure sufficient resolution for target point count
    clus.cluster(target_points)

    new_mesh = clus.create_mesh()

    # Preserve given fields by mapping from original mesh
    idx = closest_point_indices(new_mesh.points, mesh.points)

    if fields == 'all':
        fields = mesh.point_data.keys()

    for field in fields:
        new_mesh[field] = mesh[field][idx]

    return new_mesh


def closest_point_indices(source_points, target_points):
    """Return the index into ``target_points`` of the nearest neighbour for each source point.

    Uses a cKDTree for efficient spatial queries, making it suitable for large
    point clouds.

    Args:
        source_points (np.ndarray): Query points, shape ``(n, 3)``.
        target_points (np.ndarray): Reference point cloud, shape ``(m, 3)``.

    Returns:
        np.ndarray: Integer array of shape ``(n,)`` with indices into
        ``target_points``.
    """
    # Find nearest neighbour indices using KD-tree for efficient spatial queries
    kd = cKDTree(target_points)
    _, closest_indices = kd.query(source_points, k=1)
    return closest_indices


def interpolate_closest_points(source_mesh, target_mesh, scalar):
    """Transfer a scalar point-data field from source to target mesh via nearest neighbours.

    For each point in ``target_mesh``, the value is taken from the closest
    point in ``source_mesh``.  The result is stored directly in
    ``target_mesh.point_data``.

    Args:
        source_mesh: PyVista mesh that holds the field to transfer.
        target_mesh: PyVista mesh to receive the transferred field.
        scalar (str): Name of the point-data field to transfer.

    Returns:
        The modified ``target_mesh`` with the transferred field added to
        ``point_data``.
    """
    # Transfer scalar data from source mesh to target mesh via nearest neighbours
    closest_indices = closest_point_indices(target_mesh.points, source_mesh.points)
    target_mesh.point_data[scalar] = source_mesh.point_data[scalar][closest_indices]

    return target_mesh


def random_unit_vectors(n):
    """Generate ``n`` random unit vectors uniformly distributed on the unit sphere.

    Uses the standard method of sampling from a 3-D isotropic Gaussian and
    then normalising, which produces a uniform distribution on S².

    Args:
        n (int): Number of unit vectors to generate.

    Returns:
        np.ndarray: Shape ``(n, 3)`` array of unit vectors.
    """
    # Generate n random unit vectors uniformly distributed on unit sphere
    vecs = np.random.randn(n, 3)
    # Normalize each vector to unit length
    vecs /= np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    return vecs


class NoiseField:
    """Parameterised spatial noise field for mesh vertex displacement.

    Supports three noise models:

    - ``'tangential'``: high-frequency noise along surface tangents, applied
      directly on the high-resolution mesh without spatial smoothing.
      ``spacing`` is not used by this model.
    - ``'perpendicular'``: smooth in/out deformation along surface normals,
      generated on a low-resolution proxy mesh and interpolated with RBF.
    - ``'isotropic'``: smooth random-direction deformation, generated on a
      low-resolution proxy mesh and interpolated with RBF.

    Noise can be restricted to a specific anatomical region via ``region_id``,
    or applied globally by setting ``region_id=None``.
    """

    def __init__(self, region_id, amplitude, spacing, noise_model):
        """Initialise noise field parameters.

        Noise vectors are not computed at construction time; call
        :meth:`calculate_noise_vectors` followed by :meth:`scale_noise_vectors`
        to populate them.

        Args:
            region_id (int or None): Region to restrict noise to, or ``None``
                for global application.
            amplitude (float): Noise strength in millimetres.
            spacing (float or None): Spatial frequency in millimetres, used as
                the low-resolution proxy spacing for ``'perpendicular'`` and
                ``'isotropic'`` models.  Ignored by ``'tangential'``.
            noise_model (str): One of ``'tangential'``, ``'perpendicular'``,
                or ``'isotropic'``.
        """
        # Initialize noise field parameters
        self.region_id = region_id  
        self.amplitude = amplitude  
        self.spacing = spacing      
        self.noise_model = noise_model 

        # Noise vectors computed later
        self.unscaled_noise_vectors = None
        self.scaled_noise_vectors = None

    def calculate_noise_vectors(self, mesh):
        """Compute and store unscaled noise vectors for the given mesh.

        Populates ``self.unscaled_noise_vectors`` (shape ``(n_points, 3)``)
        according to the ``noise_model``.  Call :meth:`scale_noise_vectors`
        afterwards to apply amplitude scaling.

        Args:
            mesh: PyVista mesh to generate noise vectors for.  Must have
                ``point_normals`` computed and, if ``region_id`` is not
                ``None``, a ``'noise_region'`` field in ``point_data``.
        """
        # Tangential noise: mesh corruption along surface tangents
        if self.noise_model == 'tangential':

            # Generate noise amplitudes directly on high-resolution mesh
            high_res_noise = np.random.randn(mesh.n_points)
            if self.region_id is not None:
                # Mask noise to only affect specified region
                high_res_noise[mesh.point_data['noise_region'] != self.region_id] = 0

            # Create tangent vectors by cross product with random vectors
            random_vectors = np.random.randn(*mesh.points.shape)
            tangents = np.cross(mesh.point_normals, random_vectors)
            tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]
            self.unscaled_noise_vectors = tangents * high_res_noise[:, np.newaxis]

        # Perpendicular noise: in/out deformation normal to surface
        elif self.noise_model == 'perpendicular':

            # Generate smooth noise by sampling low-res mesh then interpolating
            low_res_mesh = remesh_pyacvd(mesh, target_resolution=self.spacing)
            low_res_noise = np.random.randn(low_res_mesh.n_points)

            if self.region_id is not None:
                # Transfer region labels to low-res mesh for masking
                low_res_mesh = interpolate_closest_points(mesh, low_res_mesh, 'noise_region')
                low_res_noise[low_res_mesh.point_data['noise_region'] != self.region_id] = 0

            # Smoothly interpolate noise values to high-res mesh using RBF
            interp = RBFInterpolator(low_res_mesh.points, low_res_noise, kernel='gaussian', epsilon=1/self.spacing)
            high_res_noise = interp(mesh.points)

            # Apply noise along surface normals (in/out deformation)
            self.unscaled_noise_vectors = mesh.point_normals * high_res_noise[:, np.newaxis]

        # Isotropic noise: random directions for volumetric deformation
        elif self.noise_model == 'isotropic':

            # Generate smooth vector field by sampling low-res mesh then interpolating
            low_res_mesh = remesh_pyacvd(mesh, target_resolution=self.spacing)
            low_res_noise = np.random.randn(low_res_mesh.n_points)

            if self.region_id is not None:
                # Transfer region labels to low-res mesh for masking
                low_res_mesh = interpolate_closest_points(mesh, low_res_mesh, 'noise_region')
                low_res_noise[low_res_mesh.point_data['noise_region'] != self.region_id] = 0

            # Create random direction vectors scaled by noise amplitude
            low_res_noise_vectors = random_unit_vectors(low_res_mesh.n_points) * low_res_noise[:, np.newaxis]
            # Smoothly interpolate vector field to high-res mesh
            interp = RBFInterpolator(low_res_mesh.points, low_res_noise_vectors, kernel='gaussian', epsilon=1/self.spacing)
            self.unscaled_noise_vectors = interp(mesh.points)

    def scale_noise_vectors(self):
        """Apply amplitude scaling to produce the final displacement vectors.

        Multiplies ``self.unscaled_noise_vectors`` by ``self.amplitude`` and
        stores the result in ``self.scaled_noise_vectors``.
        :meth:`calculate_noise_vectors` must be called first.
        """
        self.scaled_noise_vectors = self.amplitude * self.unscaled_noise_vectors


class SyntheticEAM:
    """Orchestrates the full synthetic EAM mesh generation pipeline.

    Holds references to the mesh at each processing stage and manages
    collections of :class:`NoiseField` objects and boundary clipping
    parameters.  The intended pipeline order is:

    1. :meth:`resample_mesh` — reduce mesh resolution.
    2. :meth:`add_noise_field` / :meth:`apply_noise_fields` — deform mesh.
    3. :meth:`apply_boundary_clips` — trim mesh at anatomical boundaries.
    4. :meth:`rotate_mesh`, :meth:`scale_mesh`, :meth:`translate_mesh`
       — augment with rigid/similarity transforms.
    """

    def __init__(self, mesh):
        """Initialise the pipeline from a source mesh.

        Scans ``mesh.point_data`` for distance fields (names ending in
        ``'distance'``) to populate :attr:`boundary_clips`, and creates
        empty noise-field slots for every unique region ID found in the
        ``'noise_region'`` field.

        Args:
            mesh: PyVista mesh that has been pre-processed by
                :func:`mesh_preprocessing.assign_distance_fields`.
        """
        # Initialize mesh processing pipeline with different stages
        self.og_mesh = mesh.copy()          # Original unmodified mesh
        self.resampled_mesh = None          # Decimated version
        self.deformed_mesh = None           # After noise application
        self.clipped_mesh = None            # After boundary clipping
        self.transformed_mesh = None        # After rotation/scaling/translation

        # Initialize noise field storage for each region and noise type
        self.noise_regions = {}
        self.noise_regions[None] = {'large': None, 'small': None, 'corruptive': None}  # Global region
        for region_id in self.get_region_ids():
            # Create empty slot for each anatomical region
            self.noise_regions[region_id] = {'large': None, 'small': None, 'corruptive': None}

        # Initialize boundary clipping parameters based on detected distance fields
        self.boundary_clips = {}
        for name in self.og_mesh.point_data:
            if name.endswith('distance'):
                boundary_name = name.replace('_distance', '')
                self.boundary_clips[boundary_name] = 0.0

    def get_boundary_names(self):
        """Return the names of all detected boundary loops.

        Returns:
            list[str]: Keys of :attr:`boundary_clips`, e.g.
            ``['boundary_1', 'boundary_2']``.
        """
        # Return list of available boundary loop names for GUI
        return list(self.boundary_clips.keys())

    def resample_mesh(self, target_points, mesh=None, method='decimate'):
        """Resample the mesh to a target number of points.

        Stores the result in :attr:`resampled_mesh`.  Uses the original mesh
        when ``mesh`` is not supplied.

        Args:
            target_points (int): Desired number of vertices after resampling.
            mesh: Source mesh to resample.  Defaults to :attr:`og_mesh`.
            method (str): ``'decimate'`` for simple decimation or ``'acvd'``
                for isotropic ACVD remeshing.
        """
        mesh = mesh or self.og_mesh

        # Reduce mesh resolution by decimating to target point count
        if method == 'decimate':
            self.resampled_mesh = sample_points(mesh, target_points=target_points)
        elif method == 'acvd':
            self.resampled_mesh = remesh_pyacvd(mesh, target_points=target_points)

    def get_region_ids(self):
        """Return the sorted list of unique anatomical region IDs.

        Reads the ``'noise_region'`` point-data field from the original mesh.

        Returns:
            list: Unique integer region IDs present in the mesh.
        """
        return list(np.unique(self.og_mesh.point_data['noise_region']))

    def add_noise_field(self, region_id, amplitude, spacing, noise_model, noise_type, mesh=None):
        """Create, compute, and register a noise field for the given region and slot.

        Constructs a :class:`NoiseField`, generates its vectors on ``mesh``,
        scales them, and stores the result in
        ``self.noise_regions[region_id][noise_type]``.

        Args:
            region_id (int or None): Target region, or ``None`` for global noise.
            amplitude (float): Noise strength in millimetres.
            spacing (float or None): Spatial frequency in millimetres (unused
                for the ``'tangential'`` noise model).
            noise_model (str): One of ``'tangential'``, ``'perpendicular'``,
                or ``'isotropic'``.
            noise_type (str): Storage slot — one of ``'large'``, ``'small'``,
                or ``'corruptive'``.
            mesh: Mesh to compute noise vectors on.  Defaults to
                :attr:`resampled_mesh` then :attr:`og_mesh`.
        """
        mesh = mesh or self.resampled_mesh or self.og_mesh

        # Create and configure a new noise field for specified region and type
        nf = NoiseField(region_id, amplitude, spacing, noise_model)
        nf.calculate_noise_vectors(mesh)
        nf.scale_noise_vectors()
        self.noise_regions[region_id][noise_type] = nf

    def clear_noise_fields(self, regions='all'):
        """Reset noise-field slots to ``None`` for the specified regions.

        Args:
            regions (str or list): Pass ``'all'`` (default) to reset every
                region including the global slot (``None``), or supply a list
                of specific region IDs.
        """
        # Remove noise fields for specified region or all regions
        if regions == 'all':
            regions = self.get_region_ids()
            regions.append(None)  # Include global region

        for region_id in regions:
            # Reset storage for each region
            self.noise_regions[region_id] = {'large': None, 'small': None, 'corruptive': None}

    def update_noise_spacing(self, region_id, noise_type, new_spacing, mesh=None):
        """Update the spatial frequency of a noise field and regenerate its vectors.

        Args:
            region_id (int or None): Target region.
            noise_type (str): Slot to update — ``'large'``, ``'small'``, or
                ``'corruptive'``.
            new_spacing (float): New spatial frequency in millimetres.
            mesh: Mesh to recompute noise vectors on.  Defaults to
                :attr:`resampled_mesh` then :attr:`og_mesh`.
        """
        mesh = mesh or self.resampled_mesh or self.og_mesh

        # Update spacing parameter and regenerate noise vectors
        self.noise_regions[region_id][noise_type].spacing = new_spacing
        self.noise_regions[region_id][noise_type].calculate_noise_vectors(mesh)
        self.noise_regions[region_id][noise_type].scale_noise_vectors()

    def update_noise_model(self, region_id, noise_type, new_noise_model, mesh=None):
        """Change the noise model of a noise field and regenerate its vectors.

        Args:
            region_id (int or None): Target region.
            noise_type (str): Slot to update — ``'large'``, ``'small'``, or
                ``'corruptive'``.
            new_noise_model (str): New model — ``'tangential'``,
                ``'perpendicular'``, or ``'isotropic'``.
            mesh: Mesh to recompute noise vectors on.  Defaults to
                :attr:`resampled_mesh` then :attr:`og_mesh`.
        """
        mesh = mesh or self.resampled_mesh or self.og_mesh

        # Update noise model and regenerate vectors with new model
        self.noise_regions[region_id][noise_type].noise_model = new_noise_model
        self.noise_regions[region_id][noise_type].calculate_noise_vectors(mesh)
        self.noise_regions[region_id][noise_type].scale_noise_vectors()

    def update_noise_amplitude(self, region_id, noise_type, new_amplitude):
        """Update the amplitude of a noise field and rescale its vectors.

        Does not regenerate noise vectors; only rescales the existing
        ``unscaled_noise_vectors`` by the new amplitude.

        Args:
            region_id (int or None): Target region.
            noise_type (str): Slot to update — ``'large'``, ``'small'``, or
                ``'corruptive'``.
            new_amplitude (float): New noise strength in millimetres.
        """
        # Update amplitude and rescale existing noise vectors
        self.noise_regions[region_id][noise_type].amplitude = new_amplitude
        self.noise_regions[region_id][noise_type].scale_noise_vectors()

    def apply_noise_fields(self, mesh=None):
        """Combine all active noise fields and apply the total displacement to the mesh.

        Sums ``scaled_noise_vectors`` across every non-``None`` noise field,
        adds the result to the mesh vertex positions, and stores the outcome
        in :attr:`deformed_mesh`.

        Args:
            mesh: Base mesh to deform.  Defaults to :attr:`resampled_mesh`
                then :attr:`og_mesh`.
        """
        mesh = mesh or self.resampled_mesh or self.og_mesh

        # Combine all noise fields and apply to original mesh
        total_noise = np.zeros_like(mesh.points)

        # Sum contributions from all active noise fields
        for region in self.noise_regions.values():
            for nf in region.values():
                if nf is not None and nf.scaled_noise_vectors is not None:
                    total_noise += nf.scaled_noise_vectors

        # Create deformed mesh by displacing original points
        self.deformed_mesh = mesh.copy()
        self.deformed_mesh.points += total_noise

    def extract_loops(self, mesh=None):
        """Extract the boundary edge loops of the mesh as separate meshes.

        Identifies all open boundary edges, computes connected components, and
        returns each connected loop as an individual mesh.

        Args:
            mesh: Mesh to extract loops from.  Defaults to :attr:`deformed_mesh`
                then :attr:`og_mesh`.

        Returns:
            list[pyvista.PolyData]: One mesh per connected boundary loop.
        """
        # Use deformed mesh if available, otherwise use original mesh
        mesh = mesh or self.deformed_mesh or self.og_mesh

        # Extract only boundary edges (holes in mesh)
        edges = mesh.extract_feature_edges(boundary_edges=True,
                                            feature_edges=False,
                                            manifold_edges=False,
                                            non_manifold_edges=False)

        # Group connected edges into separate loops
        components = edges.connectivity()

        #Handle case where no boundary edges are found (e.g. if mesh is already clipped)
        #if components has 'RegionId'
        if 'RegionId' in components.point_data:
            num_regions = int(components.point_data['RegionId'].max() + 1)
        else:
            num_regions = 0

        # Create separate mesh for each connected loop
        loops = [
            components.threshold(value=(i, i), scalars='RegionId')
            for i in range(num_regions)
        ]

        return loops

    def add_boundary_clip(self, loop_name, clip_distance):
        """Register a clipping distance for the named boundary.

        Args:
            loop_name (str): Key in :attr:`boundary_clips` identifying the
                boundary (e.g. ``'boundary_1'``).
            clip_distance (float): Signed distance threshold.  Points with a
                distance field value below this threshold are removed.
        """
        self.boundary_clips[loop_name] = clip_distance

    def update_clip_distance(self, loop_name, new_clip_distance):
        """Update the clipping distance for the named boundary.

        Args:
            loop_name (str): Key in :attr:`boundary_clips`.
            new_clip_distance (float): New signed distance threshold.
        """
        # Update clipping distance parameter
        self.boundary_clips[loop_name] = new_clip_distance

    def clear_boundary_clips(self):
        """Reset all boundary clip distances to zero.

        Setting the distance to ``0.0`` effectively disables clipping for that
        boundary (no points are removed).
        """
        # Remove all configured boundary clipping operations
        for boundary_name in self.boundary_clips:
            self.boundary_clips[boundary_name] = 0.0  # Reset clip distance to zero

    @staticmethod
    def sort_loop_points_sequentially(loop):
        """Sort loop vertices into a consistent traversal order using greedy nearest-neighbour.

        Starting from the first vertex, repeatedly selects the closest
        unvisited vertex.  Used by :meth:`smooth_loop_points` to ensure the
        B-spline is fitted along the loop rather than across it.

        Note: O(n²) complexity — may be slow for very large loops.

        Args:
            loop: PyVista mesh containing the boundary loop vertices.

        Returns:
            np.ndarray: Shape ``(n_points, 3)`` array of points in traversal
            order.
        """
        points = loop.points
        n_points = len(points)

        # Start with the first point
        sorted_indices = [0]
        remaining_indices = list(range(1, n_points))
        current_point = points[0]

        # Greedily find the nearest unvisited point at each step
        while remaining_indices:
            distances = [np.linalg.norm(points[i] - current_point) for i in remaining_indices]
            nearest_idx = remaining_indices[np.argmin(distances)]
            sorted_indices.append(nearest_idx)
            remaining_indices.remove(nearest_idx)
            current_point = points[nearest_idx]

        return points[sorted_indices]

    @staticmethod
    def smooth_loop_points(loop, smoothing_factor=0.1, n_curve_points=1000):
        """Fit a smooth B-spline to a boundary loop and project the loop points onto it.

        Sorts loop vertices sequentially, fits a periodic parametric B-spline,
        evaluates it at ``n_curve_points`` uniform parameter values, then snaps
        each original loop point to its nearest position on the smooth curve.

        Args:
            loop: PyVista mesh containing boundary loop vertices.
            smoothing_factor (float): Spline smoothing weight; higher values
                produce a smoother (less faithful) curve.  Scaled internally
                by the number of points.
            n_curve_points (int): Number of points used to discretise the
                smooth curve before projecting.

        Returns:
            np.ndarray: Shape ``(n_points, 3)`` projected points.
        """
        # Sort points sequentially around the loop to prevent zigzag
        sorted_points = SyntheticEAM.sort_loop_points_sequentially(loop)

        # For closed loop, need to ensure periodicity
        # Add the first point at the end to close the loop
        closed_points = np.vstack([sorted_points, sorted_points[0]])

        # Fit parametric B-spline to the closed loop
        tck, _ = splprep([closed_points[:, 0], closed_points[:, 1], closed_points[:, 2]],
                        s=smoothing_factor * len(closed_points),
                        per=True)  # per=True for periodic/closed curve

        # Generate smooth curve points
        u_smooth = np.linspace(0, 1, n_curve_points)
        smooth_curve = splev(u_smooth, tck)
        smooth_curve_points = np.column_stack(smooth_curve)

        # Project original points onto the smooth curve
        # Find closest point on smooth curve for each original point
        distances = cdist(loop.points, smooth_curve_points)
        closest_indices = np.argmin(distances, axis=1)
        projected_points = smooth_curve_points[closest_indices]

        return projected_points

    def smooth_boundaries(self, mesh=None, smoothing_factor=0.1, n_curve_points=1000):
        """Smooth all open boundary loops of the mesh in-place.

        Extracts each boundary loop, fits a smooth B-spline via
        :meth:`smooth_loop_points`, and moves each boundary vertex in the main
        mesh to its projection on that spline.

        Args:
            mesh: Mesh whose boundaries should be smoothed.  Defaults to
                :attr:`clipped_mesh`.
            smoothing_factor (float): Passed through to
                :meth:`smooth_loop_points`.
            n_curve_points (int): Passed through to
                :meth:`smooth_loop_points`.
        """
        # Find all boundary loops
        mesh = mesh or self.clipped_mesh
        loops = self.extract_loops(mesh)

        # Apply smoothing at every boundary
        for loop in loops:
            smoothed_loop_points = self.smooth_loop_points(loop, smoothing_factor, n_curve_points)

            for i, loop_point in enumerate(loop.points):

                # Find the closest point in the main mesh
                distances = np.linalg.norm(mesh.points - loop_point, axis=1)
                closest_idx = np.argmin(distances)

                # Set point to its smoothed version
                mesh.points[closest_idx] = smoothed_loop_points[i]

        self.clipped_mesh = mesh

    def apply_boundary_clips(self, mesh=None):
        """Apply all registered boundary clip operations sequentially.

        For each entry in :attr:`boundary_clips`, thresholds the mesh on the
        corresponding distance field, retaining only the region where the
        distance exceeds the clip threshold.  The clipped result is stored in
        :attr:`clipped_mesh`.

        Args:
            mesh: Mesh to clip.  Defaults to :attr:`deformed_mesh` then
                :attr:`og_mesh`.
        """
        # Apply all configured boundary clipping operations sequentially
        # Use deformed mesh if available, otherwise use original mesh
        mesh = mesh or self.deformed_mesh or self.og_mesh

        clipped_mesh = mesh

        # Apply each boundary clip operation
        for loop_name, distance in self.boundary_clips.items():
            # Keep points within distance threshold of boundary
            epsilon = 1e-6  # Small tolerance to avoid numerical issues
            clipped_mesh = clipped_mesh.threshold(value=distance+epsilon, scalars=f'{loop_name}_distance', invert=False).extract_surface()

        self.clipped_mesh = clipped_mesh

    def rotate_mesh(self, angle, mesh=None):
        """Apply a rotation of ``angle`` degrees about a random axis.

        Generates a random unit-vector axis, converts ``angle`` to a rotation
        vector, and rotates all mesh points.  Stores the result in
        :attr:`transformed_mesh`.

        Args:
            angle (float): Rotation magnitude in degrees.
            mesh: Mesh to rotate.  Defaults to :attr:`clipped_mesh`.
        """
        # Apply random rotation to mesh for data augmentation
        # Use clipped mesh if no mesh provided
        mesh = mesh or self.clipped_mesh

        # Generate random rotation axis and convert angle to rotation vector
        axis = random_unit_vectors(1)[0]
        rotvec = np.radians(angle) * axis  # Convert degrees to radians

        # Create rotation matrix from axis-angle representation
        rotation = R.from_rotvec(rotvec).as_matrix()

        # Apply rotation to all mesh points
        rotated_points = np.dot(mesh.points, rotation.T)
        mesh.points = rotated_points

        self.transformed_mesh = mesh

    def scale_mesh(self, scale_factor, mesh=None):
        """Uniformly scale the mesh about its centroid.

        Stores the result in :attr:`transformed_mesh`.

        Args:
            scale_factor (float): Uniform scale factor (>1 enlarges, <1 shrinks).
            mesh: Mesh to scale.  Defaults to :attr:`clipped_mesh`.
        """
        # Apply uniform scaling about mesh centroid
        # Use clipped mesh if no mesh provided
        mesh = mesh or self.clipped_mesh

        # Scale relative to centroid to maintain mesh position
        centroid = mesh.center
        scaled_points = (mesh.points - centroid) * scale_factor + centroid
        mesh.points = scaled_points

        self.transformed_mesh = mesh

    def translate_mesh(self, distance, mesh=None):
        """Translate the mesh by ``distance`` in a random direction.

        Stores the result in :attr:`transformed_mesh`.

        Args:
            distance (float): Translation magnitude in millimetres.
            mesh: Mesh to translate.  Defaults to :attr:`clipped_mesh`.
        """
        # Apply random translation for data augmentation
        # Use clipped mesh if no mesh provided
        mesh = mesh or self.clipped_mesh

        # Move mesh in random direction by specified distance
        direction = random_unit_vectors(1)[0]
        mesh.points = mesh.points + direction * distance

        self.transformed_mesh = mesh

    def export_parameters_to_csv(self, filepath):
        """Export all active noise-field and boundary-clip parameters to a CSV file.

        Writes one row per active noise field and one row per boundary clip.
        Does nothing and returns ``False`` if there are no parameters to export.

        Args:
            filepath (str or path-like): Destination CSV file path.

        Returns:
            bool: ``True`` if the file was written, ``False`` if there was
            nothing to export.
        """
        import csv

        rows = []

        # Export noise field parameters
        for region_id, noise_types in self.noise_regions.items():
            region_name = "Whole mesh" if region_id is None else f"Region {region_id}"

            for noise_type, noise_field in noise_types.items():
                if noise_field is not None:
                    rows.append({
                        'Type': 'Noise Field',
                        'Region': region_name,
                        'Noise Type': noise_type,
                        'Amplitude': noise_field.amplitude,
                        'Spacing': noise_field.spacing,
                        'Model': noise_field.noise_model,
                        'Boundary': '',
                        'Clip Distance': ''
                    })

        # Export boundary clip parameters
        for boundary_name, clip_distance in self.boundary_clips.items():
            rows.append({
                'Type': 'Boundary Clip',
                'Region': '',
                'Noise Type': '',
                'Amplitude': '',
                'Spacing': '',
                'Model': '',
                'Boundary': boundary_name,
                'Clip Distance': clip_distance
            })

        # Write to CSV file
        if rows:  # Only write if there are parameters to export
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = ['Type', 'Region', 'Noise Type', 'Amplitude', 'Spacing', 'Model',
                             'Boundary', 'Clip Distance']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return True
        return False


def make_synthetic_map_and_source_mri_mesh(source_mesh,
                                      n_points=5000,
                                      clip_distance=5.0,
                                      large_amplitude=1.5,
                                      large_spacing=10.0,
                                      small_amplitude=0.25,
                                      small_spacing=3.5,
                                      corruptive_amplitude=0.25,
                                      rotation_angle=45.0,
                                      translation_distance=50.0,
                                      scaling_range=1.2):
    """Create a paired synthetic EAM map and corresponding MRI-derived mesh.

    Applies the full synthetic EAM generation pipeline to ``source_mesh``:
    random boundary clipping, boundary smoothing (two passes), downsampling,
    noise-field deformation, and a random similarity transform
    (scale → rotate → translate).

    Args:
        source_mesh: Pre-processed PyVista mesh (output of
            :func:`mesh_preprocessing.assign_distance_fields`).
        n_points (int): Target vertex count after downsampling.
        clip_distance (float): Boundary clip magnitude in millimetres; the
            sign is chosen randomly to clip either side of each boundary.
        large_amplitude (float): Amplitude of the coarse isotropic noise in mm.
        large_spacing (float): Spatial frequency of the coarse noise in mm.
        small_amplitude (float): Amplitude of the fine isotropic noise in mm.
        small_spacing (float): Spatial frequency of the fine noise in mm.
        corruptive_amplitude (float): Amplitude of the tangential
            (surface-corrupt) noise in mm.
        rotation_angle (float): Rotation magnitude in degrees.
        translation_distance (float): Translation magnitude in mm.
        scaling_range (float): Scale factor applied as either ``scaling_range``
            or ``1/scaling_range``, chosen randomly.

    Returns:
        tuple[pyvista.PolyData, pyvista.PolyData]:
        - ``synthetic_map``: Deformed and transformed mesh simulating an EAM.
        - ``source_mri_mesh``: Downsampled source mesh without deformation.
    """
    synth = SyntheticEAM(source_mesh)

    # Clip mesh
    for name in synth.get_boundary_names():
        # Randomly clip either positive or negative side of boundary
        distance = np.random.choice([-clip_distance, clip_distance])
        synth.add_boundary_clip(name, clip_distance=distance)
    synth.apply_boundary_clips(mesh=synth.og_mesh)

    # Two smoothing passes to reduce jagged boundary artefacts
    synth.smooth_boundaries()
    synth.smooth_boundaries()

    # Resample mesh to target number of points
    synth.resample_mesh(target_points=n_points, method='decimate', mesh=synth.clipped_mesh)

    # Add noise fields
    synth.add_noise_field(region_id=None, amplitude=large_amplitude, spacing=large_spacing, noise_model='isotropic', noise_type='large')
    synth.add_noise_field(region_id=None, amplitude=small_amplitude, spacing=small_spacing, noise_model='isotropic', noise_type='small')
    synth.add_noise_field(region_id=None, amplitude=corruptive_amplitude, spacing=None, noise_model='tangential', noise_type='corruptive')
    synth.apply_noise_fields(mesh=synth.resampled_mesh)

    # Apply transformations
    scale_factor = np.random.choice([1/scaling_range, scaling_range])
    synth.scale_mesh(scale_factor=scale_factor, mesh=synth.deformed_mesh)
    synth.rotate_mesh(angle=rotation_angle, mesh=synth.transformed_mesh)
    synth.translate_mesh(distance=translation_distance, mesh=synth.transformed_mesh)

    # Get synthetic map
    synthetic_map = synth.transformed_mesh
    source_mri_mesh = synth.resampled_mesh

    return synthetic_map, source_mri_mesh


def make_registration_mesh(source_mesh,
                      n_points=7500):
    """Create a clean registration-target mesh from the source MRI mesh.

    Clips all boundaries at distance 0 (no offset clipping), applies one pass
    of boundary smoothing, and resamples to ``n_points`` vertices using the
    isotropic ACVD remesher.  No noise deformation is applied.

    Args:
        source_mesh: Pre-processed PyVista mesh (output of
            :func:`mesh_preprocessing.assign_distance_fields`).
        n_points (int): Target vertex count after resampling.

    Returns:
        pyvista.PolyData: Clean, uniformly sampled registration target.
    """
    synth = SyntheticEAM(source_mesh)

    # Clip mesh
    for name in synth.get_boundary_names():
        # Clip at boundary to create clean registration target
        synth.add_boundary_clip(name, clip_distance=0)
    synth.apply_boundary_clips(mesh=synth.og_mesh)
    synth.smooth_boundaries()

    # Resample mesh to target number of points
    synth.resample_mesh(target_points=n_points, method='acvd', mesh=synth.clipped_mesh)

    reg_mesh = synth.resampled_mesh

    return reg_mesh
