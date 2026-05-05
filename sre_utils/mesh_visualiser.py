from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QSlider, QComboBox, QPushButton, QFileDialog, QMessageBox,
    QFrame, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal
from pyvistaqt import QtInteractor
import sys
import csv
import pyvista as pv
from sre_utils.mesh_creators import SyntheticEAM


class FloatSlider(QWidget):
    """A QWidget wrapper that exposes a float-valued horizontal slider.

    Qt's ``QSlider`` only supports integer values.  ``FloatSlider`` bridges
    this by mapping the float range to an integer range internally using
    ``step`` as the resolution, and emitting a custom ``valueChangedFloat``
    signal carrying the converted float value.

    Signals:
        valueChangedFloat (float): Emitted whenever the slider moves, carrying
            the current float value.
    """

    valueChangedFloat = pyqtSignal(float)  # Custom signal for float value changes

    def __init__(self, min_val, max_val, init_val, step, label_text="", parent=None):
        """Create a labelled float slider.

        Args:
            min_val (float): Minimum slider value.
            max_val (float): Maximum slider value.
            init_val (float): Initial slider value.
            step (float): Resolution of the slider; the float range is divided
                into ``(max_val - min_val) / step`` integer ticks.
            label_text (str): Text shown to the left of the slider.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.step = step
        self.label_text = label_text

        # Convert float range to integer range for underlying QSlider
        self.int_min = int(round(min_val / step))
        self.int_max = int(round(max_val / step))
        self.int_init = int(round(init_val / step))

        # UI setup
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(QLabel(label_text), 0, 0)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(self.int_min, self.int_max)
        self.slider.setValue(self.int_init)
        self.slider.setMinimumHeight(15)
        self.slider.setMinimumWidth(200)
        layout.addWidget(self.slider, 0, 1)

        self.value_label = QLabel(f"{init_val:.2f}")
        self.value_label.setAlignment(Qt.AlignRight)
        self.value_label.setFixedWidth(60)
        layout.addWidget(self.value_label, 0, 2)

        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, int_val):
        """Convert the raw integer tick to a float, update the label, and emit the signal.

        Connected to ``QSlider.valueChanged``.

        Args:
            int_val (int): Raw integer value from the underlying ``QSlider``.
        """
        float_val = int_val * self.step
        self.value_label.setText(f"{float_val:.2f}")
        self.valueChangedFloat.emit(float_val)

    def value(self):
        """Return the current float value of the slider.

        Returns:
            float: Current value, equal to the integer tick multiplied by
            ``self.step``.
        """
        return self.slider.value() * self.step

    def setValue(self, float_val):
        """Move the slider to the nearest tick for the given float value.

        This will trigger ``QSlider.valueChanged``, which calls ``_on_change``
        and emits ``valueChangedFloat`` — disconnect the signal first if
        programmatic updates should not propagate.

        Args:
            float_val (float): Target value; should be within the slider range.
        """
        self.slider.setValue(int(round(float_val / self.step)))


class MeshVisualiser(QMainWindow):
    """Interactive Qt GUI for configuring and previewing a :class:`SyntheticEAM` mesh.

    Provides controls for mesh resampling, three independent noise types
    (large-scale, small-scale, corruptive), and boundary clipping.  The 3D
    viewport is rendered with PyVista via ``pyvistaqt``.

    The GUI is a thin front-end over a :class:`mesh_creators.SyntheticEAM`
    instance, which owns all mesh data and processing logic.

    Attributes:
        synthetic_eam (SyntheticEAM): Backend mesh processing object.
        target_resample_points (int): Current target for mesh resampling.
        resample_method (str): Current resampling method (``'decimate'`` or
            ``'acvd'``).
        active_region: Region ID currently selected in the region combo box,
            or ``None`` for "Whole mesh".
        active_boundary (str or None): Boundary name currently selected, or
            ``None`` when "None" is selected.
        actor: PyVista render actor for the currently displayed mesh.
    """

    def __init__(self, synthetic_eam):
        """Initialise the visualiser window.

        Args:
            synthetic_eam (SyntheticEAM): The backend mesh processing object
                whose parameters the GUI will control.
        """
        super().__init__()
        self.setWindowTitle("Synthetic EAM Visualiser")
        self.resize(1200, 800)

        self.synthetic_eam = synthetic_eam

        # Initialize resampling parameters to match the original mesh
        self.target_resample_points = self.synthetic_eam.og_mesh.n_points
        self.resample_method = 'decimate'

        self._build_gui()

        # Track currently active region and boundary for parameter updates
        self.active_region = None
        self.active_boundary = None

    def _build_gui(self):
        """Construct all GUI widgets and lay them out.

        Creates the PyVista 3D viewport on the left and a vertical control
        panel on the right containing (in order): mesh resampling controls,
        noise configuration (per-region, three noise types), boundary clipping
        controls, and action buttons.  Also applies any pre-existing boundary
        clips and performs the initial mesh render.
        """
        # --- Main layout ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        main_widget.setLayout(layout)

        # --- PyVista interactor ---
        self.vtk_widget = QtInteractor(self)
        layout.addWidget(self.vtk_widget, stretch=3)
        self.vtk_widget.set_background("white")

        # --- Control panel ---
        controls = QWidget()
        controls.setMinimumWidth(400)
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(5)
        controls.setLayout(controls_layout)
        layout.addWidget(controls, stretch=2)

        # --- Mesh Resampling group ---
        resampling_frame = QFrame()
        resampling_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        resampling_frame.setLineWidth(3)
        resampling_layout = QGridLayout()
        resampling_layout.setContentsMargins(8, 8, 8, 8)
        resampling_layout.setSpacing(3)
        resampling_frame.setLayout(resampling_layout)

        resampling_layout.addWidget(QLabel("Mesh Resampling"), 0, 0, 1, 3)

        # Get original mesh point count for slider range
        original_points = self.synthetic_eam.og_mesh.n_points
        min_points = max(500, int(original_points * 0.1))  # At least 500 or 10% of original
        max_points = int(original_points)

        self.slider_resample_points = FloatSlider(min_points, max_points, original_points, 500, "Target points")
        # Show integer count rather than the default decimal format
        self.slider_resample_points.value_label.setText(f"{int(original_points)}")
        # Replace _on_change so the value label shows an integer instead of a decimal
        self.slider_resample_points._on_change = lambda int_val: self._on_resample_points_change(int_val)
        resampling_layout.addWidget(self.slider_resample_points, 1, 0, 1, 3)
        self.slider_resample_points.valueChangedFloat.connect(self.update_resample_points)

        resampling_layout.addWidget(QLabel("Method:"), 2, 0)
        self.combo_resample_method = QComboBox()
        self.combo_resample_method.addItems(["decimate", "acvd"])
        self.combo_resample_method.currentTextChanged.connect(self.update_resample_method)
        self.combo_resample_method.setMinimumHeight(20)
        resampling_layout.addWidget(self.combo_resample_method, 2, 1, 1, 2)

        controls_layout.addWidget(resampling_frame)

        # --- Noise configuration group ---
        noise_main_frame = QFrame()
        noise_main_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        noise_main_frame.setLineWidth(3)
        noise_main_layout = QVBoxLayout()
        noise_main_layout.setContentsMargins(8, 8, 8, 8)
        noise_main_layout.setSpacing(5)
        noise_main_frame.setLayout(noise_main_layout)

        # --- Region selector ---
        region_layout = QGridLayout()
        region_layout.addWidget(QLabel("Region:"), 0, 0)
        self.region_combo = QComboBox()

        # Build region mapping dynamically from the mesh — labels are "Region N" for each ID
        self.region_names = {"Whole mesh": None}
        for region_id in self.synthetic_eam.get_region_ids():
            self.region_names[f"Region {region_id}"] = region_id

        self.region_combo.addItems(list(self.region_names.keys()))
        self.region_combo.currentTextChanged.connect(self.update_region)
        self.region_combo.setMinimumHeight(20)
        region_layout.addWidget(self.region_combo, 0, 1, 1, 2)
        noise_main_layout.addLayout(region_layout)

        # --- Large-scale noise subgroup ---
        large_noise_frame = QFrame()
        large_noise_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        large_noise_frame.setLineWidth(1)
        large_noise_layout = QGridLayout()
        large_noise_layout.setContentsMargins(5, 5, 5, 5)
        large_noise_layout.setSpacing(2)
        large_noise_frame.setLayout(large_noise_layout)

        large_noise_layout.addWidget(QLabel("Large-scale noise"), 0, 0, 1, 3)

        self.slider_amp_large = FloatSlider(0.0, 5.0, 0.0, 0.5, "Amplitude (mm)")
        large_noise_layout.addWidget(self.slider_amp_large, 1, 0, 1, 3)
        self.slider_amp_large.valueChangedFloat.connect(lambda v: self.update_noise_param('amplitude', v, noise_type='large'))

        self.slider_space_large = FloatSlider(5.0, 15.0, 10.0, 1.0, "Spacing (mm)")
        large_noise_layout.addWidget(self.slider_space_large, 2, 0, 1, 3)
        self.slider_space_large.valueChangedFloat.connect(lambda v: self.update_noise_param('spacing', v, noise_type='large'))

        self.combo_model_large = QComboBox()
        self.combo_model_large.addItems(["isotropic", "perpendicular"])
        self.combo_model_large.currentTextChanged.connect(lambda text: self.update_noise_param('model', text, noise_type='large'))
        self.combo_model_large.setMinimumHeight(20)
        large_noise_layout.addWidget(QLabel("Noise model"), 3, 0)
        large_noise_layout.addWidget(self.combo_model_large, 3, 1, 1, 2)

        regen_large_button = QPushButton("🔄 Regenerate noise")
        regen_large_button.clicked.connect(lambda: self.regenerate_noise_type('large'))
        large_noise_layout.addWidget(regen_large_button, 4, 0, 1, 3)

        noise_main_layout.addWidget(large_noise_frame)

        # --- Small-scale noise subgroup ---
        small_noise_frame = QFrame()
        small_noise_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        small_noise_frame.setLineWidth(1)
        small_noise_layout = QGridLayout()
        small_noise_layout.setContentsMargins(5, 5, 5, 5)
        small_noise_layout.setSpacing(2)
        small_noise_frame.setLayout(small_noise_layout)

        small_noise_layout.addWidget(QLabel("Small-scale noise"), 0, 0, 1, 3)

        self.slider_amp_small = FloatSlider(0.0, 0.5, 0.0, 0.05, "Amplitude (mm)")
        small_noise_layout.addWidget(self.slider_amp_small, 1, 0, 1, 3)
        self.slider_amp_small.valueChangedFloat.connect(lambda v: self.update_noise_param('amplitude', v, noise_type='small'))

        self.slider_space_small = FloatSlider(2.0, 5.0, 3.0, 0.5, "Spacing (mm)")
        small_noise_layout.addWidget(self.slider_space_small, 2, 0, 1, 3)
        self.slider_space_small.valueChangedFloat.connect(lambda v: self.update_noise_param('spacing', v, noise_type='small'))

        self.combo_model_small = QComboBox()
        self.combo_model_small.addItems(["isotropic", "perpendicular"])
        self.combo_model_small.currentTextChanged.connect(lambda text: self.update_noise_param('model', text, noise_type='small'))
        self.combo_model_small.setMinimumHeight(20)
        small_noise_layout.addWidget(QLabel("Noise model"), 3, 0)
        small_noise_layout.addWidget(self.combo_model_small, 3, 1, 1, 2)

        regen_small_button = QPushButton("🔄 Regenerate noise")
        regen_small_button.clicked.connect(lambda: self.regenerate_noise_type('small'))
        small_noise_layout.addWidget(regen_small_button, 4, 0, 1, 3)

        noise_main_layout.addWidget(small_noise_frame)

        # --- Corruptive (tangential) noise subgroup ---
        corr_noise_frame = QFrame()
        corr_noise_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        corr_noise_frame.setLineWidth(1)
        corr_noise_layout = QGridLayout()
        corr_noise_layout.setContentsMargins(5, 5, 5, 5)
        corr_noise_layout.setSpacing(2)
        corr_noise_frame.setLayout(corr_noise_layout)

        corr_noise_layout.addWidget(QLabel("Corruptive noise (tangential)"), 0, 0, 1, 3)
        self.slider_amp_corruptive = FloatSlider(0.0, 0.5, 0.0, 0.05, "Amplitude (mm)")
        corr_noise_layout.addWidget(self.slider_amp_corruptive, 1, 0, 1, 3)
        self.slider_amp_corruptive.valueChangedFloat.connect(lambda v: self.update_noise_param('amplitude', v, noise_type='corruptive'))

        regen_corr_button = QPushButton("🔄 Regenerate noise")
        regen_corr_button.clicked.connect(lambda: self.regenerate_noise_type('corruptive'))
        corr_noise_layout.addWidget(regen_corr_button, 2, 0, 1, 3)

        noise_main_layout.addWidget(corr_noise_frame)

        controls_layout.addWidget(noise_main_frame)

        # --- Boundary clipping group ---
        clipping_frame = QFrame()
        clipping_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        clipping_frame.setLineWidth(3)
        clipping_layout = QGridLayout()
        clipping_layout.setContentsMargins(8, 8, 8, 8)
        clipping_layout.setSpacing(3)
        clipping_frame.setLayout(clipping_layout)

        clipping_layout.addWidget(QLabel("Boundary Clipping"), 0, 0, 1, 3)

        clipping_layout.addWidget(QLabel("Boundary:"), 1, 0)
        self.boundary_combo = QComboBox()
        self.boundary_combo.addItems(["None"] + self.synthetic_eam.get_boundary_names())
        self.boundary_combo.currentTextChanged.connect(self.update_boundary)
        self.boundary_combo.setMinimumHeight(20)
        clipping_layout.addWidget(self.boundary_combo, 1, 1, 1, 2)

        self.slider_clip = FloatSlider(-10.0, 10.0, -10.0, 1.0, "Clip distance (mm)")
        clipping_layout.addWidget(self.slider_clip, 2, 0, 1, 3)
        self.slider_clip.valueChangedFloat.connect(self.update_clip_distance)

        controls_layout.addWidget(clipping_frame)

        # --- Action buttons ---
        regen_button = QPushButton("🔁 Regenerate mesh")
        regen_button.clicked.connect(self.regenerate_mesh)
        controls_layout.addWidget(regen_button)

        smooth_button = QPushButton("🔧 Smooth boundaries")
        smooth_button.clicked.connect(self.smooth_boundaries)
        controls_layout.addWidget(smooth_button)

        reset_button = QPushButton("🔄 Reset all")
        reset_button.clicked.connect(self.reset_all)
        controls_layout.addWidget(reset_button)

        export_mesh_button = QPushButton("📄 Export mesh")
        export_mesh_button.clicked.connect(self.export_mesh)
        controls_layout.addWidget(export_mesh_button)

        export_button = QPushButton("💾 Export Parameters")
        export_button.clicked.connect(self.export_parameters)
        controls_layout.addWidget(export_button)

        # Push all controls to the top
        controls_layout.addStretch()

        # Apply any pre-existing boundary clips before first render
        if self.synthetic_eam.boundary_clips:
            self.synthetic_eam.apply_boundary_clips()

        # Initial render: use the most processed mesh available
        mesh_to_display = (self.synthetic_eam.clipped_mesh or
                          self.synthetic_eam.deformed_mesh or
                          self.synthetic_eam.og_mesh)
        self.actor = self.vtk_widget.add_mesh(mesh_to_display, color="lightgray")
        self.vtk_widget.reset_camera()

    # === Event Handlers ===

    def update_region(self, region_name):
        """Handle region dropdown selection and sync noise controls to the new region.

        Args:
            region_name (str): Human-readable region name from the combo box
                (e.g. "Whole mesh", "Region 0", etc.).
        """
        # Convert descriptive name to backend region ID (None = whole mesh)
        self.active_region = self.region_names[region_name]
        self.sync_noise_controls()

    def update_noise_param(self, kind, param, noise_type):
        """Update a single noise parameter for the active region and noise type.

        Creates the noise field with sensible defaults if it does not yet
        exist, then dispatches to the appropriate backend update method by
        constructing the method name dynamically from ``kind``.

        Args:
            kind (str): Parameter to update — ``'amplitude'``, ``'spacing'``,
                or ``'model'``.
            param: New parameter value (float for amplitude/spacing,
                str for model).
            noise_type (str): Target slot — ``'large'``, ``'small'``, or
                ``'corruptive'``.
        """
        default_amplitude = 0.0
        if noise_type == 'large':
            default_spacing = 10.0
            default_model = "isotropic"
        elif noise_type == 'small':
            default_spacing = 3.0
            default_model = "isotropic"
        elif noise_type == 'corruptive':
            default_spacing = None
            default_model = "tangential"

        # Create the noise field with defaults if it has not been configured yet
        if self.synthetic_eam.noise_regions[self.active_region][noise_type] is None:
            self.synthetic_eam.add_noise_field(
                region_id=self.active_region,
                amplitude=default_amplitude,
                spacing=default_spacing,
                noise_model=default_model,
                noise_type=noise_type
            )

        # Dispatch to update_noise_amplitude, update_noise_spacing, or update_noise_model
        method = getattr(self.synthetic_eam, f"update_noise_{kind}")
        method(self.active_region, noise_type, param)

        self.synthetic_eam.apply_noise_fields()
        self.synthetic_eam.apply_boundary_clips()
        self.update_display()

    def update_boundary(self, boundary_name):
        """Handle boundary dropdown selection and sync the clip-distance control.

        Args:
            boundary_name (str): Selected boundary name, or ``'None'`` when
                no boundary is active.
        """
        self.active_boundary = boundary_name if boundary_name != "None" else None
        self.sync_boundary_controls()

    def update_clip_distance(self, distance):
        """Update the clipping distance for the currently selected boundary.

        No-op if no boundary is selected.

        Args:
            distance (float): New signed clip distance in millimetres.
        """
        if self.active_boundary is not None:
            self.synthetic_eam.update_clip_distance(self.active_boundary, distance)
            self.synthetic_eam.apply_boundary_clips()
            self.update_display()

    def _on_resample_points_change(self, int_val):
        """Custom value-change handler for the resampling slider.

        Replaces the default ``FloatSlider._on_change`` to display an integer
        point count rather than a decimal in the value label.

        Args:
            int_val (int): Raw integer tick value from the underlying
                ``QSlider``.
        """
        float_val = int_val * self.slider_resample_points.step
        int_points = int(float_val)
        self.slider_resample_points.value_label.setText(f"{int_points}")
        self.update_resample_points(float_val)

    def update_resample_points(self, n_points):
        """Store the new target point count and immediately apply resampling.

        Args:
            n_points (float): Desired number of vertices (passed as float
                from ``FloatSlider``; converted to int internally).
        """
        self.target_resample_points = int(n_points)
        self.apply_resampling()

    def update_resample_method(self, method):
        """Store the new resampling method and immediately apply resampling.

        Args:
            method (str): ``'decimate'`` or ``'acvd'``.
        """
        self.resample_method = method
        self.apply_resampling()

    def apply_resampling(self):
        """Resample the mesh and recalculate all active noise fields on the new geometry.

        Reads ``self.target_resample_points`` and ``self.resample_method``,
        resamples via the backend, recalculates every active noise field's
        vectors for the new mesh resolution, then re-applies noise and clipping
        before updating the display.
        """
        self.synthetic_eam.resample_mesh(
            target_points=self.target_resample_points,
            method=self.resample_method
        )

        # Recalculate all existing noise fields for the resampled mesh
        for region in self.synthetic_eam.noise_regions.values():
            for nf in region.values():
                if nf is not None:
                    nf.calculate_noise_vectors(self.synthetic_eam.resampled_mesh)
                    nf.scale_noise_vectors()

        self.synthetic_eam.apply_noise_fields()
        self.synthetic_eam.apply_boundary_clips()
        self.update_display()

    def regenerate_mesh(self):
        """Re-randomise all noise vectors while keeping current parameters.

        Useful for sampling different realisations of the configured noise
        fields without changing amplitudes, spacings, or models.
        """
        for region in self.synthetic_eam.noise_regions.values():
            for nf in region.values():
                if nf is not None:
                    nf.calculate_noise_vectors(self.synthetic_eam.resampled_mesh)
                    nf.scale_noise_vectors()
        self.synthetic_eam.apply_noise_fields()
        self.synthetic_eam.apply_boundary_clips()
        self.update_display()

    def regenerate_noise_type(self, noise_type):
        """Re-randomise noise vectors for one noise type in the active region only.

        Args:
            noise_type (str): Slot to regenerate — ``'large'``, ``'small'``,
                or ``'corruptive'``.
        """
        if self.active_region in self.synthetic_eam.noise_regions:
            nf = self.synthetic_eam.noise_regions[self.active_region][noise_type]
            if nf is not None:
                nf.calculate_noise_vectors(self.synthetic_eam.resampled_mesh)
                nf.scale_noise_vectors()
                self.synthetic_eam.apply_noise_fields()
                self.synthetic_eam.apply_boundary_clips()
                self.update_display()

    def smooth_boundaries(self):
        """Apply one pass of boundary smoothing to the current mesh and refresh the display."""
        self.synthetic_eam.smooth_boundaries()
        self.update_display()

    def reset_all(self):
        """Clear all noise fields, boundary clips, and resampling, then reset the GUI.

        Restores the pipeline to its post-construction state: no resampling,
        no noise, default clip distances.
        """
        self.synthetic_eam.clear_noise_fields()
        self.synthetic_eam.clear_boundary_clips()

        # Discard resampled mesh so the pipeline falls back to og_mesh
        self.synthetic_eam.resampled_mesh = None

        self.synthetic_eam.apply_noise_fields()
        self.synthetic_eam.apply_boundary_clips()

        self.reset_gui_controls()
        self.update_display()

    def reset_gui_controls(self):
        """Reset all control widgets to their default values without triggering updates.

        Temporarily disconnects signals before setting widget values to prevent
        cascading backend calls during the reset.
        """
        # Reset active selections
        self.region_combo.setCurrentText("Whole mesh")
        self.active_region = None

        self.boundary_combo.setCurrentText("None")
        self.active_boundary = None

        # Disconnect signals before bulk widget updates
        self.disconnect_noise_signals()
        self.disconnect_boundary_signals()

        self.slider_amp_large.setValue(0.0)
        self.slider_space_large.setValue(10.0)
        self.combo_model_large.setCurrentText("isotropic")

        self.slider_amp_small.setValue(0.0)
        self.slider_space_small.setValue(3.0)
        self.combo_model_small.setCurrentText("isotropic")

        self.slider_amp_corruptive.setValue(0.0)

        self.slider_clip.setValue(0.0)

        self.slider_resample_points.setValue(float(self.synthetic_eam.og_mesh.n_points))
        self.combo_resample_method.setCurrentText("decimate")
        self.target_resample_points = self.synthetic_eam.og_mesh.n_points
        self.resample_method = 'decimate'

        self.connect_noise_signals()
        self.connect_boundary_signals()

    def export_mesh(self):
        """Open a save-file dialog and export the currently displayed mesh.

        Supports VTK, VTP, STL, PLY, and OBJ formats.  Shows a success or
        error message box on completion.
        """
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Export Mesh",
            "synthetic_mesh.vtk",
            "VTK Files (*.vtk);;VTP Files (*.vtp);;STL Files (*.stl);;PLY Files (*.ply);;OBJ Files (*.obj);;All Files (*)"
        )

        if file_path:
            try:
                # Use the same priority chain as update_display
                mesh_to_export = (self.synthetic_eam.clipped_mesh or
                                 self.synthetic_eam.deformed_mesh or
                                 self.synthetic_eam.og_mesh)
                mesh_to_export.save(file_path)
                QMessageBox.information(self, "Export Successful",
                                      f"Mesh exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error",
                                   f"Failed to export mesh:\n{str(e)}")

    def export_parameters(self):
        """Open a save-file dialog and export all current parameters to a CSV file.

        Exports resampling parameters, noise field settings (per region and
        noise type), and boundary clip distances.  Shows a success or error
        message box on completion.

        Note:
            Writes CSV inline to include resampling rows not covered by
            :meth:`mesh_creators.SyntheticEAM.export_parameters_to_csv`.
        """
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Export Parameters",
            "synthetic_eam_parameters.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            try:

                rows = []

                # Export mesh resampling parameters
                if hasattr(self.synthetic_eam, 'resampled_mesh') and self.synthetic_eam.resampled_mesh is not None:
                    actual_points = self.synthetic_eam.resampled_mesh.n_points
                    rows.append({
                        'Type': 'Mesh Resampling',
                        'Region': '',
                        'Noise Type': '',
                        'Amplitude': '',
                        'Spacing': '',
                        'Model': '',
                        'Boundary': '',
                        'Clip Distance': '',
                        'Target Points': self.target_resample_points,
                        'Actual Points': actual_points,
                        'Resampling Method': self.resample_method
                    })

                # Export noise field parameters
                for region_id, noise_types in self.synthetic_eam.noise_regions.items():
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
                                'Clip Distance': '',
                                'Target Points': '',
                                'Actual Points': '',
                                'Resampling Method': ''
                            })

                # Export boundary clip parameters
                for boundary_name, clip_distance in self.synthetic_eam.boundary_clips.items():
                    rows.append({
                        'Type': 'Boundary Clip',
                        'Region': '',
                        'Noise Type': '',
                        'Amplitude': '',
                        'Spacing': '',
                        'Model': '',
                        'Boundary': boundary_name,
                        'Clip Distance': clip_distance,
                        'Target Points': '',
                        'Actual Points': '',
                        'Resampling Method': ''
                    })

                if rows:
                    with open(file_path, 'w', newline='') as csvfile:
                        fieldnames = ['Type', 'Region', 'Noise Type', 'Amplitude', 'Spacing', 'Model',
                                     'Boundary', 'Clip Distance', 'Target Points', 'Actual Points', 'Resampling Method']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                    QMessageBox.information(self, "Export Successful",
                                          f"Parameters exported to:\n{file_path}")
                else:
                    QMessageBox.warning(self, "Export Warning",
                                      "No active parameters to export.")

            except Exception as e:
                QMessageBox.critical(self, "Export Error",
                                   f"Failed to export parameters:\n{str(e)}")

    def update_display(self):
        """Refresh the 3D viewport with the most-processed mesh currently available.

        Priority order: ``clipped_mesh`` → ``deformed_mesh`` → ``og_mesh``.
        Resets the camera after every update.
        """
        self.vtk_widget.remove_actor(self.actor)
        mesh_to_display = (self.synthetic_eam.clipped_mesh or
                          self.synthetic_eam.deformed_mesh or
                          self.synthetic_eam.og_mesh)
        self.actor = self.vtk_widget.add_mesh(mesh_to_display, color="lightgray")
        self.vtk_widget.reset_camera()

    def sync_noise_controls(self):
        """Update all noise control widgets to match the active region's current parameters.

        Disconnects signals before setting widget values to prevent triggering
        backend updates during the sync.
        """
        self.disconnect_noise_signals()

        noise_region = self.synthetic_eam.noise_regions[self.active_region]

        # Sync large-scale noise controls
        large_nf = noise_region['large']
        if large_nf is not None:
            self.slider_amp_large.setValue(large_nf.amplitude)
            self.slider_space_large.setValue(large_nf.spacing)
            self.combo_model_large.setCurrentText(large_nf.noise_model)
        else:
            self.slider_amp_large.setValue(0.0)
            self.slider_space_large.setValue(10.0)
            self.combo_model_large.setCurrentText("isotropic")

        # Sync small-scale noise controls
        small_nf = noise_region['small']
        if small_nf is not None:
            self.slider_amp_small.setValue(small_nf.amplitude)
            self.slider_space_small.setValue(small_nf.spacing)
            self.combo_model_small.setCurrentText(small_nf.noise_model)
        else:
            self.slider_amp_small.setValue(0.0)
            self.slider_space_small.setValue(3.0)
            self.combo_model_small.setCurrentText("isotropic")

        # Sync corruptive noise controls (amplitude only; model is fixed to tangential)
        corruptive_nf = noise_region['corruptive']
        if corruptive_nf is not None:
            self.slider_amp_corruptive.setValue(corruptive_nf.amplitude)
        else:
            self.slider_amp_corruptive.setValue(0.0)

        self.connect_noise_signals()

    def sync_boundary_controls(self):
        """Update the clip-distance slider to match the active boundary's current value.

        Disconnects the signal before updating to prevent triggering a clip
        recomputation.
        """
        self.disconnect_boundary_signals()

        if self.active_boundary is not None and self.active_boundary in self.synthetic_eam.boundary_clips:
            distance = self.synthetic_eam.boundary_clips[self.active_boundary]
            self.slider_clip.setValue(distance)
        else:
            self.slider_clip.setValue(-10.0)  # Slider minimum — indicates no active clip

        self.connect_boundary_signals()

    def disconnect_noise_signals(self):
        """Disconnect all noise control signals to suppress updates during GUI sync."""
        self.slider_amp_large.valueChangedFloat.disconnect()
        self.slider_space_large.valueChangedFloat.disconnect()
        self.combo_model_large.currentTextChanged.disconnect()
        self.slider_amp_small.valueChangedFloat.disconnect()
        self.slider_space_small.valueChangedFloat.disconnect()
        self.combo_model_small.currentTextChanged.disconnect()
        self.slider_amp_corruptive.valueChangedFloat.disconnect()

    def connect_noise_signals(self):
        """Reconnect all noise control signals after a GUI sync operation."""
        self.slider_amp_large.valueChangedFloat.connect(lambda v: self.update_noise_param('amplitude', v, noise_type='large'))
        self.slider_space_large.valueChangedFloat.connect(lambda v: self.update_noise_param('spacing', v, noise_type='large'))
        self.combo_model_large.currentTextChanged.connect(lambda text: self.update_noise_param('model', text, noise_type='large'))
        self.slider_amp_small.valueChangedFloat.connect(lambda v: self.update_noise_param('amplitude', v, noise_type='small'))
        self.slider_space_small.valueChangedFloat.connect(lambda v: self.update_noise_param('spacing', v, noise_type='small'))
        self.combo_model_small.currentTextChanged.connect(lambda text: self.update_noise_param('model', text, noise_type='small'))
        self.slider_amp_corruptive.valueChangedFloat.connect(lambda v: self.update_noise_param('amplitude', v, noise_type='corruptive'))

    def disconnect_boundary_signals(self):
        """Disconnect the boundary control signal to suppress updates during GUI sync."""
        self.slider_clip.valueChangedFloat.disconnect()

    def connect_boundary_signals(self):
        """Reconnect the boundary control signal after a GUI sync operation."""
        self.slider_clip.valueChangedFloat.connect(self.update_clip_distance)


if __name__ == "__main__":

    app = QApplication(sys.argv)

    # Prompt user to select an input mesh file
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(
        None,
        "Select Input Mesh",
        "",
        "Mesh Files (*.vtk *.vtp *.stl *.ply *.obj);;All Files (*)"
    )

    if not file_path:
        QMessageBox.warning(None, "No File Selected", "No mesh file was selected. Exiting.")
        sys.exit(0)

    try:
        mesh = pv.read(file_path)
        synthetic_eam = SyntheticEAM(mesh)
        window = MeshVisualiser(synthetic_eam)
        window.show()
        sys.exit(app.exec_())

    except Exception as e:
        QMessageBox.critical(None, "Error Loading Mesh", f"Failed to load mesh:\n{str(e)}")
        sys.exit(1)
