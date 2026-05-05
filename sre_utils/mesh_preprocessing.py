import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import openep


def get_points_by_region(mesh):
    """Return a mapping from region ID to the unique point indices that belong to it.

    Iterates over every cell in the mesh, reads the ``cell_region`` cell-data
    field to determine which region each cell belongs to, and collects all
    unique vertex indices for that region.

    Args:
        mesh: PyVista mesh with a ``cell_region`` integer array in ``cell_data``.

    Returns:
        dict[int, np.ndarray]: Maps each region ID to a sorted array of unique
        point indices that are part of at least one cell in that region.
    """
    region_points = {}

    for i in range(mesh.n_cells):
        cell = mesh.get_cell(i)
        region_id = mesh.cell_data['cell_region'][i]
        points = cell.point_ids
        if region_id not in region_points:
            region_points[int(region_id)] = []  # int cast keeps dict keys as native Python ints
        region_points[region_id].append(points)

    for region_id in region_points:
        region_points[region_id] = np.unique(np.concatenate(region_points[region_id]))  # flatten and deduplicate

    return region_points


def get_boundary_points(region_points):
    """Find the points shared between region 0 (background) and each named region.

    Region 0 is treated as the background/surrounding tissue. Any point that
    belongs to both region 0 and another region lies on the interface between
    the two, and is therefore a boundary point for that region.

    Args:
        region_points (dict[int, array-like]): Maps region ID to an array of
            point indices, as returned by :func:`get_points_by_region`.

    Returns:
        dict[int, list[int]]: Maps each non-zero region ID to the list of
        point indices that it shares with region 0.
    """
    boundary_points = {}
    region_0_points = set(region_points[0])  # set for O(1) membership lookups

    for i in range(1, len(region_points)):
        other_region_points = set(region_points[i])
        boundary_points[i] = list(region_0_points.intersection(other_region_points))  # shared vertices lie on the interface

    return boundary_points


def build_edge_graph(mesh):
    """Build a symmetric weighted adjacency matrix (CSR) for the mesh vertices.

    Parses the packed ``faces`` array of a PyVista mesh (format:
    ``[n_verts, v0, v1, …, n_verts, v0, v1, …]``), extracts every unique
    polygon edge, and assigns it a weight equal to the Euclidean distance
    between the two endpoint vertices.

    Args:
        mesh: PyVista mesh whose ``faces`` attribute follows the packed-integer
            format and whose ``points`` array has shape ``(n_points, 3)``.

    Returns:
        scipy.sparse.csr_matrix: Symmetric ``(n_points × n_points)`` adjacency
        matrix where entry ``[i, j]`` is the Euclidean length of the edge
        between vertices *i* and *j*, or 0 if no such edge exists.
    """
    pts = mesh.points
    faces = mesh.faces
    n_points = mesh.n_points

    # parse faces array: format [n0, v0, v1, ..., n1, v0, v1, ...]
    i = 0
    edges = set()
    faces_len = faces.size
    while i < faces_len:
        n = int(faces[i])
        verts = faces[i+1:i+1+n]
        # add each polygon edge (loop)
        for j in range(n):
            a = int(verts[j])
            b = int(verts[(j+1) % n])
            if a != b:
                edges.add((min(a,b), max(a,b)))
        i += 1 + n

    # build arrays for sparse matrix
    rows = []
    cols = []
    data = []
    for a, b in edges:
        pa = pts[a]
        pb = pts[b]
        w = np.linalg.norm(pa - pb)
        rows.append(a); cols.append(b); data.append(w)
        rows.append(b); cols.append(a); data.append(w)  # symmetric

    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_points, n_points))
    return adj.tocsr()


def geodesic_distance_to_boundary(mesh, boundary_point_indices):
    """Compute the shortest geodesic distance from every vertex to the nearest boundary point.

    Uses Dijkstra's algorithm on the mesh edge graph, treating edge lengths as
    weights. When multiple boundary points are supplied, the returned distance
    for each vertex is the minimum across all sources.

    Args:
        mesh: PyVista mesh passed through to :func:`build_edge_graph`.
        boundary_point_indices (array-like of int): Indices of the vertices
            that define the boundary (source points for Dijkstra).

    Returns:
        np.ndarray: Shape ``(n_points,)``. Entry *i* is the geodesic distance
        from vertex *i* to the nearest boundary point.

    Raises:
        ValueError: If ``boundary_point_indices`` is empty.
    """
    if len(boundary_point_indices) == 0:
        raise ValueError("boundary_point_indices is empty")

    adj = build_edge_graph(mesh)

    dists_from_each = dijkstra(csgraph=adj, directed=False, indices=np.asarray(boundary_point_indices))
    # If only one source, shape will be (n_vertices,), convert to (1, n_vertices)
    if dists_from_each.ndim == 1:
        dists_from_each = dists_from_each[np.newaxis, :]

    # For each vertex, take the minimum distance across all boundary sources
    distances = np.min(dists_from_each, axis=0)
    return distances


def assign_distance_fields(mesh):
    """Compute signed geodesic distance fields for every non-background region and attach them as point data.

    For each region (excluding region 0), calculates the geodesic distance from
    every mesh vertex to the region boundary. Distances are negated for points
    that lie *inside* the region, producing a signed distance field where
    negative values indicate interior membership.  The resulting array is stored
    in ``mesh.point_data`` under the key ``boundary_<region_id>_distance``.

    The ``cell_region`` cell-data array is removed from the mesh once all
    distance fields have been assigned, as it is no longer needed.

    Args:
        mesh: PyVista mesh with ``cell_region`` cell data, as produced by
            :func:`mesh_from_case`.

    Returns:
        The input mesh with distance fields added to ``point_data`` and
        ``cell_region`` removed from ``cell_data``.
    """
    region_points = get_points_by_region(mesh)
    boundary_points = get_boundary_points(region_points)

    distance_arrays = {}

    for region_id, b_points in boundary_points.items():
        distances = geodesic_distance_to_boundary(mesh, b_points)
        distance_arrays[region_id] = distances
        distance_arrays[region_id][region_points[region_id]] *= -1  # sign interior points negative to form signed distance field

        field_name = f'boundary_{region_id}_distance'
        mesh.point_data[field_name] = distance_arrays[region_id]

    # clear cell region from cell data
    mesh.cell_data.remove('cell_region')

    return mesh


def mesh_from_case(case):
    """Construct a PyVista mesh from a SyntheticEAM case object.

    Creates the base geometry via ``case.create_mesh()``, then transfers all
    non-``None`` fields from ``case.fields`` to the appropriate mesh data
    container: arrays whose length matches ``n_points`` go to ``point_data``;
    arrays whose length matches ``n_cells`` go to ``cell_data``.

    Args:
        case (SyntheticEAM): A fully configured synthetic EAM case.

    Returns:
        pyvista.PolyData: Mesh with case fields transferred to ``point_data``
        or ``cell_data``.
    """
    mesh = case.create_mesh()

    # transfer fields and ensure noise_region is point data
    for field in case.fields:
        if case.fields[field] is None:
            continue
        if len(case.fields[field]) == mesh.n_points:  # per-vertex field
            mesh.point_data[field] = case.fields[field]
        elif len(case.fields[field]) == mesh.n_cells:  # per-cell field
            mesh.cell_data[field] = case.fields[field]

    return mesh


def case_from_mesh(mesh, name='processed_case'):
    """Convert a processed PyVista mesh back into an OpenEP Case.

    Constructs a minimal OpenEP ``Case`` from the mesh geometry and then
    copies all arrays in ``mesh.point_data`` into ``case.fields``.  Electric
    and ablation sub-structures are initialised empty.

    Args:
        mesh: PyVista mesh whose ``faces`` array follows the packed-integer
            format (``[n_verts, v0, v1, …]``).
        name (str): Name assigned to the resulting ``Case`` object.
            Defaults to ``'processed_case'``.

    Returns:
        openep.data_structures.case.Case: Case with geometry and point fields
        populated from the mesh.
    """
    case = openep.data_structures.case.Case(
        name=name,
        points=mesh.points,
        indices=mesh.faces.reshape((-1, 4))[:, 1:],  # drop leading count column to get (n_faces, 3) vertex indices
        fields=openep.data_structures.surface.Fields(),
        electric=openep.data_structures.electric.Electric(),
        ablation=openep.data_structures.ablation.Ablation()
    )

    # transfer point data fields to case
    for field in mesh.point_data:
        case.fields[field] = mesh.point_data[field]

    return case


def ensure_noise_region(mesh):
    """Ensure the ``noise_region`` point-data field exists on the mesh.

    If ``noise_region`` is not already present in ``mesh.point_data``, it is
    initialised to an all-zero integer array of length ``n_points``, indicating
    that no point belongs to a noise region by default.

    Args:
        mesh: PyVista mesh to check and potentially modify.

    Returns:
        The input mesh, with ``noise_region`` guaranteed to be present in
        ``point_data``.
    """
    if 'noise_region' not in mesh.point_data:
        mesh.point_data['noise_region'] = np.zeros(mesh.n_points, dtype=int)

    return mesh
