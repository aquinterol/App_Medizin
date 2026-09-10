import os

os.environ["ETS_TOOLKIT"] = "qt4"
os.environ["ETS_QT4_IMPORTS"] = "1"
os.environ["QT_API"] = "pyqt5"

from mayavi import mlab
from pyface.qt import QtGui
import traitsui
import matplotlib
matplotlib.use('Qt5Agg')
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QDialog, QMessageBox, QTableWidgetItem
from mayavi.core.ui.api import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from tvtk.pyface.scene_editor import SceneEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from ui_pw import Ui_Widget
from ui_sw import Ui_Dialog  
from ui_fwhm import Ui_Form  
from ui_popup import Ui_Form as Ui_ScatterDialog
from ui_cd import Ui_Form as Ui_CDDialog
import scipy.io
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Function for interpolating the data
def get_interpolated_plane(data, coords, coord_mm, axis):
        coord_m = coord_mm / 1000.0 # Convert input mm to meters
        idx = np.searchsorted(coords, coord_m) # Coordinates (self.x/y/z) are in meters
        if idx == 0:
            idx0, idx1 = 0, 1
        elif idx >= len(coords):
            idx0, idx1 = len(coords) - 2, len(coords) - 1
        else:
            idx0, idx1 = idx - 1, idx
        
        if idx1 >= len(coords): # Edge case handling
            idx1 = len(coords) - 1
            idx0 = idx1 - 1
            
        c0, c1 = coords[idx0], coords[idx1]
        alpha = (coord_m - c0) / (c1 - c0) if (c1 - c0) != 0 else 0
        
        if axis == 'x':
            img0 = data[idx0, :, :]
            img1 = data[idx1, :, :]
        elif axis == 'y':
            img0 = data[:, idx0, :]
            img1 = data[:, idx1, :]
        elif axis == 'z':
            img0 = data[:, :, idx0]
            img1 = data[:, :, idx1]
        else:
            raise ValueError('Invalid axis')
        img_interp = (1 - alpha) * img0 + alpha * img1
        return img_interp


class VisualizationWidget(HasTraits):
    scene = Instance(MlabSceneModel, ())
    view = View(
        Item(
            'scene',
            editor=SceneEditor(scene_class=MayaviScene),
            show_label=False,
        ),
        resizable=True,
    )

    def cleanup(self):
        if self.scene is not None:
            self.scene.stop()
            self.scene = None

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        # Connect the scatter button to open the dialog
        self.ui.scatter.clicked.connect(self.open_scatter_dialog)

        # Button state
        self.scatter_activado = False

        # Connect the buttons, events, and slider
        self.ui.openFile.clicked.connect(self.open_file)
        self.ui.comboBox.setEnabled(False)
        self.ui.comboBox.currentIndexChanged.connect(self.plot_volume)
        self.ui.horizontalSlider.setEnabled(False)
        self.ui.horizontalSlider.valueChanged.connect(self.update_slice_position)
        self.ui.Figure2D.clicked.connect(self.open_popup_dialog)
        self.ui.CompData.clicked.connect(self.open_comparison_dialog)
       
        
        # Connect the units combobox
        self.ui.comboBox_2.currentIndexChanged.connect(self.update_units)

        # Create the Mayavi visualization for the main layout
        self.visualization = VisualizationWidget()
        self.visualization_control = self.visualization.edit_traits(parent=self, kind='subpanel').control
        self.ui.viLayout.addWidget(self.visualization_control)

        # Create visualization widgets for the tabs
        self.visualization_xy = self._create_visualization(self.ui.tabWidget.widget(0))
        self.visualization_yz = self._create_visualization(self.ui.tabWidget.widget(1))
        self.visualization_xz = self._create_visualization(self.ui.tabWidget.widget(2))

        # Variable to store the data
        self.data = None  # Store converted data
        self.data_shape = None
        self.plane_xy = None
        self.plane_yz = None
        self.plane_xz = None
        self.current_tab_index = 0
        
        # Variables to store the x, y, z coordinates
        self.x = None
        self.y = None
        self.z = None
        self.lambda_value = 1.0  # Default value

        # Connect the tab change event
        self.ui.tabWidget.currentChanged.connect(self.tab_changed)    

    def _create_visualization(self, parent_widget):
        vis_widget = VisualizationWidget()
        layout = QVBoxLayout()
        parent_widget.setLayout(layout)
        layout.addWidget(vis_widget.edit_traits(parent=self, kind='subpanel').control)
        return vis_widget

    @staticmethod
    def get_param_value(mat_data, field_name):
        try:
            value = mat_data['param'][field_name][0][0][0]
            
            # --- TYPE CORRECTION ---
            # Make sure the value is a Python scalar
            if isinstance(value, np.ndarray):
                value = value.item() # Extract the scalar from an array (e.g. np.array([1.5]) -> 1.5)
            # --- END TYPE CORRECTION ---

            return value
        except (KeyError, IndexError):
            return "Parameter wasn't found."    

    def tab_changed(self, index):
        self.current_tab_index = index
        if self.data is not None:
            self.update_slider_range()
            # Update the slider position to reflect the current plane
            self.update_slice_position(self.ui.horizontalSlider.value())

    def update_slider_range(self):
        if self.data is None:
            return

        # Adjust the slider range according to the current dimension
        if self.current_tab_index == 0:  # XY (control on Z)
            max_val = self.data.shape[2] - 1
        elif self.current_tab_index == 1:  # YZ (control on X)
            max_val = self.data.shape[0] - 1
        else:  # XZ (control on Y)
            max_val = self.data.shape[1] - 1

        # Preserve the relative position when switching tabs
        current_value = self.ui.horizontalSlider.value()
        old_max = self.ui.horizontalSlider.maximum()
        
        # If the slider already has a range, calculate the relative position
        if old_max > 0:
            relative_position = current_value / old_max
            new_value = int(relative_position * max_val)
        else:
            # Otherwise, use the midpoint
            new_value = max_val // 2
        
        # Configure the slider range and value
        self.ui.horizontalSlider.setRange(0, max_val)
        self.ui.horizontalSlider.setValue(new_value)
        self.update_position_label(new_value)

    def update_units(self):
        # Avoid update if no data is loaded
        if self.data is None:
            return

        # Get the selected unit and scale factor
        selected_unit = self.ui.comboBox_2.currentText()
        if selected_unit == "Milimeters":
            scale_factor = 1000
            xlabel = 'X (mm)'
            ylabel = 'Y (mm)'
            zlabel = 'Z (mm)'
        else:  # Wavelength
            scale_factor = 1 / self.lambda_value if self.lambda_value != 0 else 1
            xlabel = r'X ($\lambda$)' 
            ylabel = r'Y ($\lambda$)'
            zlabel = r'Z ($\lambda$)'

        try:
            # Update main visualization axes
            mlab.clf(figure=self.visualization.scene.mayavi_scene)
            src = mlab.pipeline.scalar_field(self.data, figure=self.visualization.scene.mayavi_scene)
            
            # Update the visualization based on comboBox selection
            choice = self.ui.comboBox.currentText()
            if choice == "Isosurface":
                mlab.contour3d(self.data, contours=8, opacity=0.5)
            elif choice == "Volume rendering":
                mlab.pipeline.volume(src)

            # Use the real x, y, z values for the axes if available
            if self.x is not None and self.y is not None and self.z is not None:
                x_min, x_max = self.x[0], self.x[-1]
                y_min, y_max = self.y[0], self.y[-1]
                z_min, z_max = self.z[0], self.z[-1]
                
                x_min *= scale_factor
                x_max *= scale_factor
                y_min *= scale_factor
                y_max *= scale_factor
                z_min *= scale_factor
                z_max *= scale_factor
                
                axes = mlab.axes(
                    xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                    ranges=np.array([x_min, x_max, y_min, y_max, z_min, z_max]).flatten()
                )
            else:
                # Fall back to indices if there are no real coordinates
                axes = mlab.axes(
                    xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                    ranges=[0, self.data.shape[0]*scale_factor, 
                        0, self.data.shape[1]*scale_factor, 
                        0, self.data.shape[2]*scale_factor]
                )
            
            mlab.colorbar(orientation='vertical')
            
            # Update tab visualizations with the new units
            self.update_tab_visualizations(scale_factor, xlabel, ylabel, zlabel)
            
            # Preserve the current slider position but update its label with new units
            current_pos = self.ui.horizontalSlider.value()
            units = 'mm' if selected_unit == "Milimeters" else r'$\lambda$'
            self.update_position_label(current_pos, units)

        except Exception as e:
            self.ui.textInfo.append(f"Error updating axis labels: {e}")

    def update_position_label(self, position, units=None):
        if self.data is None:
            return
        
        # If no units are provided, use the currently selected ones
        if units is None:
            units = 'mm' if self.ui.comboBox_2.currentText() == "Milimeters" else r'$\lambda$'
            
        # Calculate the scale factor based on the units
        if units == 'mm':
            scale_factor = 1000 
        else:
            scale_factor = 1 / self.lambda_value 
            
        # Get the total size of the current axis and the position
        if self.current_tab_index == 0:  # XY (control on Z)
            axis_name = 'Z'
            total_size = self.data.shape[2]
            current_pos = position
            
            # If we have real z values, use those instead of the index
            if self.z is not None and position < len(self.z):
                real_pos = self.z[position].astype(float) * scale_factor
                if isinstance(real_pos, np.ndarray):
                 real_pos = real_pos.item()  # Convert to scalar if it's a single value array
                plane_text = f"XY Plane at Z = {real_pos:.2f} {units}"
            else:
                plane_text = f"XY Plane at Z = {current_pos} {units}"
                
        elif self.current_tab_index == 1:  # YZ (control on X)
            axis_name = 'X'
            total_size = self.data.shape[0]
            current_pos = position
            
            # If we have real x values, use those instead of the index
            if self.x is not None and position < len(self.x):
                real_pos = self.x[position].astype(float) * scale_factor
                if isinstance(real_pos, np.ndarray):
                 real_pos = real_pos.item()  # Convert to scalar if it's a single value array
                plane_text = f"YZ Plane at X = {real_pos:.2f} {units}"
            else:
                plane_text = f"YZ Plane at X = {current_pos} {units}"
                
        else:  # XZ (control on Y)
            axis_name = 'Y'
            total_size = self.data.shape[1]
            current_pos = position
            
            # If we have real y values, use those instead of the index
            if self.y is not None and position < len(self.y):
                real_pos = self.y[position].astype(float) * scale_factor
                if isinstance(real_pos, np.ndarray):
                 real_pos = real_pos.item()  # Convert to scalar if it's a single value array
                plane_text = f"XZ Plane at Y = {real_pos:.2f} {units}"
            else:
                plane_text = f"XZ Plane at Y = {current_pos} {units}"

        # Update the label with the axis and current position
        self.ui.label.setText(plane_text)
        
        # Also update textInfo
        current_info = self.ui.textInfo.toPlainText()
        info_lines = current_info.split('\n')
        position_line = f"Current Position: {plane_text} ({position}/{total_size-1})"
        
        position_found = False
        for i, line in enumerate(info_lines):
            if "Current Position:" in line:
                info_lines[i] = position_line
                position_found = True
                break
                
        if not position_found:
            info_lines.append("")
            info_lines.append(position_line)
        
        self.ui.textInfo.setText('\n'.join(info_lines))

    def update_slice_position(self, position):
        if self.data is None:
            return

        # Get the current units
        selected_unit = 'mm' if self.ui.comboBox_2.currentText() == "Milimeters" else r'$\lambda$'
        self.update_position_label(position, selected_unit)

        try:
            # Make sure the planes exist before updating them
            if self.current_tab_index == 0:  # XY
                if self.plane_xy and hasattr(self.plane_xy, 'ipw'):
                    self.plane_xy.ipw.slice_position = position
            elif self.current_tab_index == 1:  # YZ
                if self.plane_yz and hasattr(self.plane_yz, 'ipw'):
                    self.plane_yz.ipw.slice_position = position
            else:  # XZ
                if self.plane_xz and hasattr(self.plane_xz, 'ipw'):
                    self.plane_xz.ipw.slice_position = position
        except Exception as e:
            self.ui.textInfo.append(f"Error updating slice position: {e}")

    def update_tab_visualizations(self, scale_factor=None, xlabel=None, ylabel=None, zlabel=None):
        if self.data is None:
            return

        try:
            # If no parameters are provided, use default values
            if scale_factor is None or xlabel is None or ylabel is None or zlabel is None:
                selected_unit = self.ui.comboBox_2.currentText()
                if selected_unit == "Milimeters":
                    scale_factor = 1000
                    xlabel = 'X (mm)'
                    ylabel = 'Y (mm)'
                    zlabel = 'Z (mm)'
                else:  # Wavelength
                    scale_factor = 1 / self.lambda_value if self.lambda_value != 0 else 1
                    xlabel = r'X ($\lambda$)'
                    ylabel = r'Y ($\lambda$)'
                    zlabel = r'Z ($\lambda$)'
            
            # Configure cut planes at the current slider positions, or by default in the middle
            slice_x = self.ui.horizontalSlider.value() if self.current_tab_index == 1 else self.data.shape[0] // 2
            slice_y = self.ui.horizontalSlider.value() if self.current_tab_index == 2 else self.data.shape[1] // 2
            slice_z = self.ui.horizontalSlider.value() if self.current_tab_index == 0 else self.data.shape[2] // 2
            
            # Make sure the cut indices are within bounds
            slice_x = max(0, min(slice_x, self.data.shape[0] - 1))
            slice_y = max(0, min(slice_y, self.data.shape[1] - 1))
            slice_z = max(0, min(slice_z, self.data.shape[2] - 1))
            
            # Determine the ranges for the axes
            if self.x is not None and self.y is not None and self.z is not None:
                x_min, x_max = self.x[0], self.x[-1]
                y_min, y_max = self.y[0], self.y[-1]
                z_min, z_max = self.z[0], self.z[-1]
                
                # Apply scale factor if necessary
                x_min *= scale_factor
                x_max *= scale_factor
                y_min *= scale_factor
                y_max *= scale_factor
                z_min *= scale_factor
                z_max *= scale_factor
            else:
                # Fall back to indices if there are no real coordinates
                x_min, x_max = 0, self.data.shape[0] * scale_factor
                y_min, y_max = 0, self.data.shape[1] * scale_factor
                z_min, z_max = 0, self.data.shape[2] * scale_factor
            
            # Create a NumPy array for the ranges
            ranges = np.array([x_min, x_max, y_min, y_max, z_min, z_max]).flatten()

            # XY Visualization (XY Plane)
            mlab.clf(figure=self.visualization_xy.scene.mayavi_scene)
            src_xy = mlab.pipeline.scalar_field(self.data, figure=self.visualization_xy.scene.mayavi_scene)
            self.plane_xy = mlab.pipeline.image_plane_widget(src_xy, 
                plane_orientation='z_axes', 
                slice_index=slice_z, 
                figure=self.visualization_xy.scene.mayavi_scene
            )
            self.plane_xy.ipw.interaction = 0
            mlab.axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, 
                    ranges=ranges,
                    figure=self.visualization_xy.scene.mayavi_scene)
            mlab.colorbar(orientation='vertical')
            self.visualization_xy.scene.camera.view_up = [0, 1, 0]
            self.visualization_xy.scene.camera.elevation(-90)

            # YZ Visualization (YZ Plane)
            mlab.clf(figure=self.visualization_yz.scene.mayavi_scene)
            src_yz = mlab.pipeline.scalar_field(self.data, figure=self.visualization_yz.scene.mayavi_scene)
            self.plane_yz = mlab.pipeline.image_plane_widget(src_yz, 
                plane_orientation='x_axes', 
                slice_index=slice_x, 
                figure=self.visualization_yz.scene.mayavi_scene
            )
            self.plane_yz.ipw.interaction = 0
            mlab.axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, 
                    ranges=ranges,
                    figure=self.visualization_yz.scene.mayavi_scene)
            mlab.colorbar(orientation='vertical')
            self.visualization_yz.scene.camera.view_up = [0, 1, 0]
            self.visualization_yz.scene.camera.azimuth(90)

            # XZ Visualization (XZ Plane)
            mlab.clf(figure=self.visualization_xz.scene.mayavi_scene)
            src_xz = mlab.pipeline.scalar_field(self.data, figure=self.visualization_xz.scene.mayavi_scene)
            self.plane_xz = mlab.pipeline.image_plane_widget(src_xz, 
                plane_orientation='y_axes', 
                slice_index=slice_y, 
                figure=self.visualization_xz.scene.mayavi_scene
            )
            self.plane_xz.ipw.interaction = 0
            mlab.axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, 
                    ranges=ranges,
                    figure=self.visualization_xz.scene.mayavi_scene)
            mlab.colorbar(orientation='vertical')
            self.visualization_xz.scene.camera.view_up = [0, 0, 1]
            self.visualization_xz.scene.camera.azimuth(90)

            # Update the position label with the correct units
            units = 'mm' if 'mm' in xlabel else r'$\lambda$'
            self.update_position_label(self.ui.horizontalSlider.value(), units)

        except Exception as e:
            self.ui.textInfo.append(f"Error updating tab visualizations: {e}")        

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "MAT files (*.mat)")
        if file_path:
            try:
                mat_data = scipy.io.loadmat(file_path)
                self.data = mat_data.get('data', None)
                
                # Load the x, y, z coordinates if available
                self.x = mat_data.get('x', None)
                self.y = mat_data.get('y', None)
                self.z = mat_data.get('z', None)

                # Load the x, y, z coordinates if available
                self.xs = mat_data.get('xs', None)
                self.ys = mat_data.get('ys', None)
                self.zs = mat_data.get('zs', None)

                
                # Get lambda_value from the file
                self.lambda_value = self.get_param_value(mat_data, 'lambda')
                if isinstance(self.lambda_value, str) or self.lambda_value == 0 or self.lambda_value is None:
                    self.lambda_value = 1.0  # Default value if not found
                
                # Convert to one-dimensional arrays if necessary
                if self.x is not None and len(self.x.shape) > 1:
                    self.x = self.x.ravel()
                if self.y is not None and len(self.y.shape) > 1:
                    self.y = self.y.ravel()
                if self.z is not None and len(self.z.shape) > 1:
                    self.z = self.z.ravel()

                if self.data is not None:
                    self.ui.fileText.setText(f"Opened File: {file_path}")
                    self.ui.comboBox.setEnabled(True)
                    self.ui.horizontalSlider.setEnabled(True)
                    
                    # Configure the slider and update visualizations
                    self.update_slider_range()
                    self.plot_volume()
                    self.update_tab_visualizations()
                    
                    # Get the current units for the visualization
                    selected_unit = 'mm' if self.ui.comboBox_2.currentText() == "Milimeters" else r'$\lambda$'
                    
                    # Add information about the coordinates
                    x_info = f"X range: [{self.x[0]:.4f} to {self.x[-1]:.4f}]" if self.x is not None else "X coordinates not found"
                    y_info = f"Y range: [{self.y[0]:.4f} to {self.y[-1]:.4f}]" if self.y is not None else "Y coordinates not found"
                    z_info = f"Z range: [{self.z[0]:.4f} to {self.z[-1]:.4f}]" if self.z is not None else "Z coordinates not found"
                    
                    info_text = f"""
                    File information:
                    ---------------------
                    Name: {file_path.split('/')[-1]}
                    Units: {selected_unit}
                    Lambda: {self.lambda_value}
                    
                    Coordinate Information:
                    ---------------------
                    {x_info}
                    {y_info}
                    {z_info}
                    
                    Parameters Information:
                    ---------------------
                    Src Type: {self.get_param_value(mat_data, 'srcType')}
                    Data Name: {self.get_param_value(mat_data, 'dataName')}
                    Logaritmic Compression: {self.get_param_value(mat_data, 'logCompression')}
                    """
                    self.ui.textInfo.setText(info_text)
                else:
                    self.ui.textInfo.setText("Error: No valid data in file.")
            except Exception as e:
                self.ui.textInfo.setText(f"Error loading file: {e}")

    def plot_volume(self):
        """Plots in the main layout"""
        if self.data is None:
            self.ui.textInfo.setText("No data loaded.")
            return

        try:
            # Clear the main visualization scene
            if not self.scatter_activado:
             mlab.clf(figure=self.visualization.scene.mayavi_scene)
            
            # Configure the background and create the scalar field
            self.visualization.scene.background = (0.2, 0.2, 0.2)
            src = mlab.pipeline.scalar_field(self.data, figure=self.visualization.scene.mayavi_scene)
            
            # Select the visualization type based on the comboBox
            choice = self.ui.comboBox.currentText()
            if choice == "Isosurface":
                 mlab.contour3d(self.data, contours=8, opacity=0.5)
            elif choice == "Volume rendering":
                mlab.pipeline.volume(mlab.pipeline.scalar_field(self.data, vmin=0, vmax=0.8))

            
            # Determine the labels and ranges based on real values if available
            selected_unit = self.ui.comboBox_2.currentText()
            if selected_unit == "Milimeters":
                scale_factor = 1000 
            else:
                scale_factor = 1 / self.lambda_value 
            xlabel = 'X (mm)' if selected_unit == "Milimeters" else r'X ($\lambda$)'
            ylabel = 'Y (mm)' if selected_unit == "Milimeters" else r'Y ($\lambda$)'
            zlabel = 'Z (mm)' if selected_unit == "Milimeters" else r'Z ($\lambda$)'
            
            if self.x is not None and self.y is not None and self.z is not None and self.scatter_activado is False:
                x_min, x_max = self.x[0] * scale_factor, self.x[-1] * scale_factor
                y_min, y_max = self.y[0] * scale_factor, self.y[-1] * scale_factor
                z_min, z_max = self.z[0] * scale_factor, self.z[-1] * scale_factor
                
                axes = mlab.axes(
                    xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                    ranges=np.array([x_min, x_max, y_min, y_max, z_min, z_max]).flatten()
                )
            else:
                mlab.axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

            # Configure colorbar and adjust the view
            mlab.colorbar(orientation='vertical', nb_labels=5)
            self.visualization.scene.camera.zoom(1.5)
            self.visualization.scene.render()    

            if self.scatter_activado:
                
                # Map the real coordinates (xs, ys, zs) to pixel indices
                if self.x is not None and self.y is not None and self.z is not None:
                    # Calculate the pixel indices corresponding to the real coordinates
                    x_indices = np.interp(self.xs, (self.x.min(), self.x.max()), (0, self.data.shape[0] - 1))
                    y_indices = np.interp(self.ys, (self.y.min(), self.y.max()), (0, self.data.shape[1] - 1))
                    z_indices = np.interp(self.zs, (self.z.min(), self.z.max()), (0, self.data.shape[2] - 1))
                else:
                    # If there are no real coordinates, use the indices directly
                    x_indices = self.xs
                    y_indices = self.ys
                    z_indices = self.zs

                # Add the points to the scene
                mlab.points3d(
                    x_indices, y_indices, z_indices, 
                    scale_factor=15.0,  # Adjust this value to change the point size
                    color=(1, 0, 0),  # Red color
                    figure=self.visualization.scene.mayavi_scene,  # Existing Mayavi scene 
                )
                # Hide the scene axes
                self.scatter_activado = False 

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting volume: {e}")

    def open_scatter_dialog(self):
        try:
            reply = QMessageBox.question(
                self,
                "Data for the Scatter",
                "Do you want to upload new data for scatter plot? (Selecting 'No' will use current data)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Select file", "", 
                    "CSV files (*.csv);;Text files (*.txt);;All files (*)"
                )
                if file_path:
                    import numpy as np
                    arr = np.loadtxt(file_path, delimiter=',')
                    if arr.shape[1] != 3:
                        QMessageBox.critical(self, "Error", "File must have tree columns (xs, ys, zs).")
                        return
                    # Assign directly to the class attributes
                    self.xs, self.ys, self.zs = arr[:,0], arr[:,1], arr[:,2]
                else:
                    return
            
            if self.data is None:
                QMessageBox.warning(self, "Error", "No data loaded. Please open a .mat file first.")
                return

            self.scatter_activado = not self.scatter_activado
            self.plot_volume()
            
            # Pass self.lambda_value to the ScatterDialog constructor
            self.scatter_dialog = ScatterDialog(
                self.xs, self.ys, self.zs, 
                self.x, self.y, self.z, 
                self.data, 
                self.lambda_value # <-- Added lambda value
            )
            
            self.scatter_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open the file: {str(e)}")

    def open_popup_dialog(self):
         try:
             if self.data is None:
                 QMessageBox.warning(self, "Error", "No data loaded. Please open a .mat file first.")
                 return
             dlg = PopupDialog(
                 parent=self,
                 data=self.data,
                 x=self.x, y=self.y, z=self.z,
                 xs=self.xs, ys=self.ys, zs=self.zs,
             )
             dlg.exec_()        
         except Exception as e:
             QMessageBox.critical(self, "Error", f"Error: {str(e)}") 

    def open_comparison_dialog(self):
        try:
            if self.data is None:
                QMessageBox.warning(self, "Error", "No data loaded. Please open a .mat file first.")
                return
            
            # The dialog will automatically ask for the file when it opens
            dlg = CDDialog(
                parent=self,
                data=self.data,
                x=self.x, y=self.y, z=self.z,
                xs=self.xs, ys=self.ys, zs=self.zs,
                lambda_value=self.lambda_value
            )
            dlg.exec_()        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error: {str(e)}")           

class ProfilePlotMixin:
    """
    Reusable mixin
        sources: 
            'data', 'x', 'y', 'z'      -> volume and axes in METERS
            'xs', 'ys', 'zs'           -> scatterer coordinates (meters)
            'lambda_value'             -> scalar
            'label'                    -> name for legend/messages
            'color'                    -> matplotlib color (or None = auto)

        mode_combo: the QComboBox that toggles "Simulation peaks " /
            "Manual input " (self.ui.combbprincipal in ScatterDialog,
            self.ui.combbprincipal_2 in CDDialog).
    """

    # ---------------------------------------------------------------- init
    def _init_profile_plots(self, sources, mode_combo):
        self.sources = sources
        self._mode_combo = mode_combo
        self.zoom_window = 4.0  # always in mm

        # Normalize lambda_value for each source
        for src in self.sources:
            lv = src.get('lambda_value', 1.0)
            if isinstance(lv, np.ndarray):
                lv = lv.item()
            if not lv:
                lv = 1.0
            src['lambda_value'] = lv

        if hasattr(self.ui, 'checkBox'):
            self.ui.checkBox.stateChanged.connect(self._on_normalize_or_units_changed)
        else:
            print("WARNING: 'self.ui.checkBox'.")

        if hasattr(self.ui, 'checkunits'):
            self.ui.checkunits.stateChanged.connect(self._on_normalize_or_units_changed)
        else:
            print("WARNING: 'self.ui.checkunits'.")

        self.valor_guardado_x = 0.0
        self.valor_guardado_y = 0.0
        self.valor_guardado_z = 0.0

        self.ui.combbsimu.setEnabled(False)
        self.ui.InputIndex.setEnabled(False)
        self.ui.buttonTable.setEnabled(False)
        self.ui.buttonGraph.setEnabled(False)
        self.ui.buttonTable.clicked.connect(self._open_table_dialog)

        self._setup_matplotlib_canvases()

        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.ui.InputIndex.textChanged.connect(self._on_input_index_changed)
        self.ui.InputIndey.textChanged.connect(self._on_input_index_changed)
        self.ui.InputIndez.textChanged.connect(self._on_input_index_changed)
        self.ui.buttonGraph.clicked.connect(self._update_manual_graphs)

        self._simu_connection = None
        self._populate_simulation_points()

        if hasattr(self.ui, 'InputZoom'):
            self.ui.InputZoom.editingFinished.connect(self._update_zoom_window)
        else:
            print("WARNING: 'self.ui.InputZoom'.")

        self._on_mode_changed(self._mode_combo.currentIndex())

    # ------------------------------------------------------------- canvas
    def _setup_matplotlib_canvases(self):
        """Set up Matplotlib canvases for X, Y, Z with NavigationToolbar"""
        self.figure_x = Figure(figsize=(5, 4), dpi=100)
        self.canvas_x = FigureCanvas(self.figure_x)
        self.toolbar_x = NavigationToolbar2QT(self.canvas_x, self.ui.FrameX)
        layout_x = QVBoxLayout(self.ui.FrameX)
        layout_x.addWidget(self.toolbar_x)
        layout_x.addWidget(self.canvas_x)

        self.figure_y = Figure(figsize=(5, 4), dpi=100)
        self.canvas_y = FigureCanvas(self.figure_y)
        self.toolbar_y = NavigationToolbar2QT(self.canvas_y, self.ui.FrameY)
        layout_y = QVBoxLayout(self.ui.FrameY)
        layout_y.addWidget(self.toolbar_y)
        layout_y.addWidget(self.canvas_y)

        self.figure_z = Figure(figsize=(5, 4), dpi=100)
        self.canvas_z = FigureCanvas(self.figure_z)
        self.toolbar_z = NavigationToolbar2QT(self.canvas_z, self.ui.FrameZ)
        layout_z = QVBoxLayout(self.ui.FrameZ)
        layout_z.addWidget(self.toolbar_z)
        layout_z.addWidget(self.canvas_z)

        self.figure_x.set_tight_layout(True)
        self.figure_y.set_tight_layout(True)
        self.figure_z.set_tight_layout(True)

    # --------------------------------------------------------- units
    def _units_are_lambda(self):
        return hasattr(self.ui, 'checkunits') and self.ui.checkunits.isChecked()

    def _get_axis_labels(self):
        if self._units_are_lambda():
            return r'X ($\lambda$)', r'Y ($\lambda$)', r'Z ($\lambda$)'
        return 'X (mm)', 'Y (mm)', 'Z (mm)'

    def _get_source_coords(self, src):
        """x,y,z coordinates of a source already scaled to the current unit."""
        if self._units_are_lambda():
            lv = src['lambda_value']
            return (src['x'].flatten() / lv,
                    src['y'].flatten() / lv,
                    src['z'].flatten() / lv)
        return (src['x'].flatten() * 1000.0,
                src['y'].flatten() * 1000.0,
                src['z'].flatten() * 1000.0)

    def _scale_factor(self, lambda_value):
        """Factor to convert mm -> current unit (mm or lambda)."""
        if self._units_are_lambda():
            return 1.0 / (lambda_value * 1000.0)
        return 1.0

    # ----------------------------------------------------- simulated points
    def _populate_simulation_points(self):
        self.ui.combbsimu.clear()

        for src in self.sources:
            for key in ('xs', 'ys', 'zs'):
                val = src.get(key)
                if val is not None:
                    src[key] = np.array(val).flatten()

        # Use as reference the first source that actually has scatterers
        ref = next((s for s in self.sources if s.get('xs') is not None), None)
        n_points = len(ref['xs']) if ref is not None else 0
        for index in range(n_points):
            self.ui.combbsimu.addItem(f"Point {index}")

    def _ensure_pixel_indices(self, src):
        """Calculates xs_pixels/ys_pixels/zs_pixels for a source (or None if not applicable)."""
        if src.get('xs') is None or src.get('x') is None:
            src['xs_pixels'] = None
            return
        x, y, z = src['x'], src['y'], src['z']
        src['xs_pixels'] = np.round(np.interp(
            src['xs'], (x.min(), x.max()), (0, x.shape[0] - 1))).astype(int)
        src['ys_pixels'] = np.round(np.interp(
            src['ys'], (y.min(), y.max()), (0, y.shape[0] - 1))).astype(int)
        src['zs_pixels'] = np.round(np.interp(
            src['zs'], (z.min(), z.max()), (0, z.shape[0] - 1))).astype(int)

    # ------------------------------------------------------------- inputs
    def _on_input_index_changed(self):
        """Updates the value when the user changes the manual text (always in mm)."""
        try:
            self.valor_guardado_x = float(self.ui.InputIndex.text().strip())
        except ValueError:
            self.valor_guardado_x = 0.0
        try:
            self.valor_guardado_y = float(self.ui.InputIndey.text().strip())
        except ValueError:
            self.valor_guardado_y = 0.0
        try:
            self.valor_guardado_z = float(self.ui.InputIndez.text().strip())
        except ValueError:
            self.valor_guardado_z = 0.0

    def _update_zoom_window(self):
        if not hasattr(self.ui, 'InputZoom'):
            return
        texto = self.ui.InputZoom.text().strip()
        try:
            valor = float(texto)
            if valor <= 0:
                raise ValueError
            self.zoom_window = valor
        except ValueError:
            QMessageBox.warning(self, "Wrong number", "The value must be positive.")
            self.ui.InputZoom.setText(str(self.zoom_window))
            return

        if self._mode_combo.currentText() == "Simulation peaks " and self.ui.combbsimu.isEnabled():
            idx = self.ui.combbsimu.currentIndex()
            if self._simu_connection is not None:
                self._simu_connection(idx)

    def _on_normalize_or_units_changed(self):
        """Redraws the current graphs when normalization or units change."""
        choice = self._mode_combo.currentText()
        if choice == "Simulation peaks ":
            if self._simu_connection is not None:
                idx = self.ui.combbsimu.currentIndex()
                if idx >= 0:
                    self._simu_connection(idx)
        elif choice == "Manual input ":
            self._update_manual_graphs()

    # ------------------------------------------------------------ FWHM
    def find_fwhm_points(self, profile, axis):
        """
        Returns: left endpoint (mm), right endpoint (mm), FWHM (mm),
        half-power level (dB). 'axis' must be in meters.
        """
        profile = np.array(profile)
        axis = np.array(axis).flatten()
        max_idx = np.argmax(profile)
        max_value = profile[max_idx]
        half_power_db = max_value - 6

        left_idx = max_idx
        while left_idx > 0 and profile[left_idx - 1] > half_power_db:
            left_idx -= 1
        right_idx = max_idx
        while right_idx < len(profile) - 1 and profile[right_idx + 1] > half_power_db:
            right_idx += 1

        def interpolate_point(idx1, idx2):
            if idx1 < 0 or idx1 >= len(axis) or idx2 < 0 or idx2 >= len(axis):
                return axis[max_idx]
            if idx1 >= len(profile) or idx2 >= len(profile):
                return axis[max_idx]
            x1_m, y1 = axis[idx1], profile[idx1]
            x2_m, y2 = axis[idx2], profile[idx2]
            if y1 == y2:
                return x1_m
            return x1_m + (half_power_db - y1) * (x2_m - x1_m) / (y2 - y1)

        if left_idx > 0:
            left_x_m = interpolate_point(left_idx, left_idx - 1)
        else:
            left_x_m = axis[left_idx]
        if right_idx < len(profile) - 1:
            right_x_m = interpolate_point(right_idx, right_idx + 1)
        else:
            right_x_m = axis[right_idx]

        fwhm_m = abs(right_x_m - left_x_m)
        return left_x_m * 1000.0, right_x_m * 1000.0, fwhm_m * 1000.0, half_power_db

    @staticmethod
    def _find_peaks_and_values(profile):
        max_value = np.max(profile)
        min_value = np.min(profile)
        signal_range = max_value - min_value

        if signal_range == 0:
            return np.array([np.argmax(profile)]), [profile[np.argmax(profile)]]

        adaptive_prominence = signal_range * 0.1
        height_threshold = min_value + signal_range * 0.1
        peaks, _ = find_peaks(profile, prominence=adaptive_prominence,
                               height=height_threshold, distance=10)
        if len(peaks) == 0:
            adaptive_prominence = signal_range * 0.05
            height_threshold = min_value + signal_range * 0.05
            peaks, _ = find_peaks(profile, prominence=adaptive_prominence,
                                   height=height_threshold, distance=5)
        if len(peaks) == 0:
            peaks = np.array([np.argmax(profile)])

        peak_values = profile[peaks]
        order = np.argsort(peak_values)[::-1]
        peaks = peaks[order][:5]
        return peaks, [profile[p] for p in peaks]

    # ------------------------------------------------------- mode (combo)
    def _on_mode_changed(self, index):
        choice = self._mode_combo.currentText()

        self.figure_x.clear()
        self.figure_y.clear()
        self.figure_z.clear()
        self.canvas_x.draw()
        self.canvas_y.draw()
        self.canvas_z.draw()

        self.ui.combbsimu.setEnabled(False)
        self.ui.InputIndex.setEnabled(False)
        self.ui.InputIndey.setEnabled(False)
        self.ui.InputIndez.setEnabled(False)
        if hasattr(self.ui, 'InputZoom'):
            self.ui.InputZoom.setEnabled(False)

        if choice == "Simulation peaks ":
            self.ui.combbsimu.setEnabled(True)
            self.ui.buttonTable.setEnabled(True)
            self.ui.buttonGraph.setEnabled(False)
            if hasattr(self.ui, 'InputZoom'):
                self.ui.InputZoom.setEnabled(True)

            valid = [s for s in self.sources if s.get('x') is not None and s.get('xs') is not None]
            if not valid:
                QMessageBox.critical(self, "Error",
                    "Simulation peaks (xs) not available in any loaded file, "
                    "or coordinate ranges (x, y, z) are missing.")
                return

            for src in self.sources:
                self._ensure_pixel_indices(src)

            if self._simu_connection is not None:
                try:
                    self.ui.combbsimu.currentIndexChanged.disconnect(self._simu_connection)
                except TypeError:
                    pass

            self._simu_connection = lambda idx: self._update_profile_plots_from_index(idx)
            self.ui.combbsimu.currentIndexChanged.connect(self._simu_connection)

            if self.ui.combbsimu.count() > 0:
                self._simu_connection(0)

        elif choice == "Manual input ":
            self.ui.InputIndex.setEnabled(True)
            self.ui.InputIndey.setEnabled(True)
            self.ui.InputIndez.setEnabled(True)
            self.ui.buttonGraph.setEnabled(True)
            self.ui.buttonTable.setEnabled(False)

            try:
                current_text_x = self.ui.InputIndex.text().strip()
                self.valor_guardado_x = float(current_text_x) if current_text_x else 0.0
                current_text_y = self.ui.InputIndey.text().strip()
                self.valor_guardado_y = float(current_text_y) if current_text_y else 0.0
                current_text_z = self.ui.InputIndez.text().strip()
                self.valor_guardado_z = float(current_text_z) if current_text_z else 0.0
            except ValueError:
                self.valor_guardado_x = 0.0
                self.valor_guardado_y = 0.0
                self.valor_guardado_z = 0.0
                self.ui.InputIndex.setText('0')
                self.ui.InputIndey.setText('0')
                self.ui.InputIndez.setText('0')

            if any(s.get('data') is not None and s.get('x') is not None for s in self.sources):
                self._update_manual_graphs()

        elif choice == "Find peaks ":
            pass

    # --------------------------------------------------- profiles (peaks)
    def _update_profile_plots_from_index(self, point_index):
        """'Simulation peaks' mode: draws the profile of the selected point
        for all sources that have that index available."""
        entries = []
        for src in self.sources:
            xp = src.get('xs_pixels')
            if xp is None or point_index >= len(xp):
                continue
            ix = src['xs_pixels'][point_index]
            iy = src['ys_pixels'][point_index]
            iz = src['zs_pixels'][point_index]

            x_profile = src['data'][:, iy, iz]
            y_profile = src['data'][ix, :, iz]
            z_profile = src['data'][ix, iy, :]

            if hasattr(self.ui, 'checkBox') and self.ui.checkBox.isChecked():
                if x_profile.size > 0: x_profile = x_profile - np.max(x_profile)
                if y_profile.size > 0: y_profile = y_profile - np.max(y_profile)
                if z_profile.size > 0: z_profile = z_profile - np.max(z_profile)

            xc, yc, zc = self._get_source_coords(src)
            entries.append((src, {
                'x_coords': xc, 'x_profile': x_profile,
                'y_coords': yc, 'y_profile': y_profile,
                'z_coords': zc, 'z_profile': z_profile,
            }))

        if not entries:
            return
        self._draw_profiles(entries, mode='fwhm')

    # -------------------------------------------------- profiles (manual)
    def _update_manual_graphs(self):
        """'Manual input' mode: draws the profile at (x_mm, y_mm, z_mm) for
        all sources whose coordinate range allows it."""
        x_mm = self.valor_guardado_x
        y_mm = self.valor_guardado_y
        z_mm = self.valor_guardado_z

        entries = []
        warnings = []

        for src in self.sources:
            if src.get('data') is None or src.get('x') is None or src.get('y') is None or src.get('z') is None:
                continue

            x_range_mm = (src['x'].flatten().min() * 1000, src['x'].flatten().max() * 1000)
            y_range_mm = (src['y'].flatten().min() * 1000, src['y'].flatten().max() * 1000)
            z_range_mm = (src['z'].flatten().min() * 1000, src['z'].flatten().max() * 1000)

            out_of_range = []
            if not (x_range_mm[0] <= x_mm <= x_range_mm[1]):
                out_of_range.append(f"X ({x_mm:.2f} mm) out of range [{x_range_mm[0]:.2f}, {x_range_mm[1]:.2f}] mm")
            if not (y_range_mm[0] <= y_mm <= y_range_mm[1]):
                out_of_range.append(f"Y ({y_mm:.2f} mm) out of range [{y_range_mm[0]:.2f}, {y_range_mm[1]:.2f}] mm")
            if not (z_range_mm[0] <= z_mm <= z_range_mm[1]):
                out_of_range.append(f"Z ({z_mm:.2f} mm) out of range [{z_range_mm[0]:.2f}, {z_range_mm[1]:.2f}] mm")

            if out_of_range:
                warnings.append(f"{src.get('label', 'File')}: " + "; ".join(out_of_range))
                continue

            x_m, y_m, z_m = x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0
            x_index = int(np.interp(x_m, src['x'].flatten(), np.arange(src['data'].shape[0])))
            y_index = int(np.interp(y_m, src['y'].flatten(), np.arange(src['data'].shape[1])))
            z_index = int(np.interp(z_m, src['z'].flatten(), np.arange(src['data'].shape[2])))

            x_profile = src['data'][:, y_index, z_index]
            y_profile = src['data'][x_index, :, z_index]
            z_profile = src['data'][x_index, y_index, :]

            if hasattr(self.ui, 'checkBox') and self.ui.checkBox.isChecked():
                if x_profile.size > 0: x_profile = x_profile - np.max(x_profile)
                if y_profile.size > 0: y_profile = y_profile - np.max(y_profile)
                if z_profile.size > 0: z_profile = z_profile - np.max(z_profile)

            xc, yc, zc = self._get_source_coords(src)
            entries.append((src, {
                'x_coords': xc, 'x_profile': x_profile,
                'y_coords': yc, 'y_profile': y_profile,
                'z_coords': zc, 'z_profile': z_profile,
            }))

        if warnings:
            QMessageBox.warning(self, "Out of Range Error",
                                 "Values out of range:\n" + "\n".join(warnings))
        if not entries:
            return

        self._draw_profiles(
            entries, mode='peaks',
            title_x=f'X Profile at Y={y_mm:.2f}mm, Z={z_mm:.2f}mm',
            title_y=f'Y Profile at X={x_mm:.2f}mm, Z={z_mm:.2f}mm',
            title_z=f'Z Profile at X={x_mm:.2f}mm, Y={y_mm:.2f}mm',
        )

    # ------------------------------------------------------ drawing (core)
    def _draw_profiles(self, entries, mode, title_x=None, title_y=None, title_z=None):
        """
        entries: list of (src, profile_dict) — profile_dict carries
                 x_coords/x_profile, y_coords/y_profile, z_coords/z_profile.
        mode: 'fwhm'  -> single peak + FWHM markers (Simulation peaks)
              'peaks' -> multiple annotated peaks (Manual input)
        Draws all sources overlaid on the same X/Y/Z axes.
        """
        self.figure_x.clear()
        self.figure_y.clear()
        self.figure_z.clear()

        ax_x = self.figure_x.add_subplot(111)
        ax_y = self.figure_y.add_subplot(111)
        ax_z = self.figure_z.add_subplot(111)

        xlabel, ylabel, zlabel = self._get_axis_labels()
        unit_label = r'$\lambda$' if self._units_are_lambda() else 'mm'
        multi = len(entries) > 1

        self._pick_artists = {'x': [], 'y': [], 'z': []}
        self._peak_info = {'x': [], 'y': [], 'z': []}

        for axis_key, ax, axis_label, default_title in (
            ('x', ax_x, xlabel, title_x),
            ('y', ax_y, ylabel, title_y),
            ('z', ax_z, zlabel, title_z),
        ):
            xlims = []
            fwhm_titles = []

            for src, prof in entries:
                coords = prof[f'{axis_key}_coords']
                profile = prof[f'{axis_key}_profile']
                color = src.get('color')
                label = src.get('label', 'File')

                ax.plot(coords, profile, color=color, label=label if multi else None)

                if mode == 'fwhm':
                    axis_m = src[axis_key]
                    left_mm, right_mm, fwhm_mm, half_power = self.find_fwhm_points(profile, axis_m)
                    sf = self._scale_factor(src['lambda_value'])
                    max_idx = int(np.argmax(profile))
                    peak_scaled = coords[max_idx]
                    left_s, right_s = left_mm * sf, right_mm * sf
                    fwhm_s = fwhm_mm * sf

                    dots, = ax.plot(peak_scaled, profile[max_idx], 'o', color=color, picker=5)
                    ax.plot([left_s, right_s], [half_power, half_power], '--', color=color)
                    ax.plot([left_s], [half_power], 'v', color=color)
                    ax.plot([right_s], [half_power], 'v', color=color)

                    self._pick_artists[axis_key].append(dots)
                    self._peak_info[axis_key].append({
                        'coords': np.array([peak_scaled]),
                        'values': [profile[max_idx]],
                        'label': label,
                    })
                    fwhm_titles.append(f"FWHM = {fwhm_s:.3f} {unit_label}" if multi
                                        else f"FWHM = {fwhm_s:.3f} {unit_label}")

                    zoom_s = self.zoom_window * sf
                    xlims.append((peak_scaled - zoom_s, peak_scaled + zoom_s))
                else:  # 'peaks'
                    peaks, values = self._find_peaks_and_values(profile)
                    peak_coords = coords[peaks]
                    dots, = ax.plot(peak_coords, values, 'o', color=color, picker=5)
                    for pc, v in zip(peak_coords, values):
                        ax.annotate(f'{v:.1f}dB', (pc, v), xytext=(0, 10),
                                    textcoords='offset points', ha='center',
                                    fontsize=8, color=color)
                    self._pick_artists[axis_key].append(dots)
                    self._peak_info[axis_key].append({
                        'coords': peak_coords, 'values': values, 'label': label,
                    })

            if mode == 'fwhm' and xlims:
                ax.set_xlim(min(l for l, _ in xlims), max(h for _, h in xlims))
                ax.set_title(f'{axis_key.upper()} Profile: ' + ' | '.join(fwhm_titles))
            else:
                ax.set_title(default_title or f'{axis_key.upper()} Profile')

            ax.set_xlabel(axis_label)
            ax.set_ylabel('Amplitude (dB)')
            ax.grid(True)
            if multi:
                ax.legend(fontsize=8)

        self.canvas_x.draw()
        self.canvas_y.draw()
        self.canvas_z.draw()
        self._connect_pick_events()

    def _connect_pick_events(self):
        unit_label = r'$\lambda$' if self._units_are_lambda() else 'mm'
        for axis_key, canvas_attr, cid_attr in (
            ('x', 'canvas_x', '_xpick_cid'),
            ('y', 'canvas_y', '_ypick_cid'),
            ('z', 'canvas_z', '_zpick_cid'),
        ):
            canvas = getattr(self, canvas_attr)
            old_cid = getattr(self, cid_attr, None)
            if old_cid is not None:
                try:
                    canvas.mpl_disconnect(old_cid)
                except Exception:
                    pass

            def make_handler(axis_key=axis_key, unit_label=unit_label):
                def handler(event):
                    artists = self._pick_artists.get(axis_key, [])
                    if event.artist not in artists:
                        return
                    idx = artists.index(event.artist)
                    info = self._peak_info[axis_key][idx]
                    ind = event.ind[0]
                    val_coord = info['coords'][ind]
                    val_amp = info['values'][ind]
                    QMessageBox.information(self, "Point value",
                        f"{info['label']} — {axis_key.upper()} Profile\n"
                        f"{axis_key.upper()} = {val_coord:.2f} {unit_label}\n"
                        f"Amplitude = {val_amp:.2f} dB")
                return handler

            new_cid = canvas.mpl_connect('pick_event', make_handler())
            setattr(self, cid_attr, new_cid)

    # ------------------------------------------------------------- table
    def _calculate_all_fwhms(self, src):
        fwhm_list = []
        sf = self._scale_factor(src['lambda_value'])
        xs_pixels = src.get('xs_pixels')
        if xs_pixels is None:
            return fwhm_list

        for idx in range(len(xs_pixels)):
            x_profile = src['data'][:, src['ys_pixels'][idx], src['zs_pixels'][idx]]
            y_profile = src['data'][src['xs_pixels'][idx], :, src['zs_pixels'][idx]]
            z_profile = src['data'][src['xs_pixels'][idx], src['ys_pixels'][idx], :]

            if hasattr(self.ui, 'checkBox') and self.ui.checkBox.isChecked():
                if x_profile.size > 0: x_profile = x_profile - np.max(x_profile)
                if y_profile.size > 0: y_profile = y_profile - np.max(y_profile)
                if z_profile.size > 0: z_profile = z_profile - np.max(z_profile)

            _, _, fwhm_x_mm, _ = self.find_fwhm_points(x_profile, src['x'])
            _, _, fwhm_y_mm, _ = self.find_fwhm_points(y_profile, src['y'])
            _, _, fwhm_z_mm, _ = self.find_fwhm_points(z_profile, src['z'])

            fwhm_list.append((idx, fwhm_x_mm * sf, fwhm_y_mm * sf, fwhm_z_mm * sf))

        return fwhm_list

    def _open_table_dialog(self):
        try:
            # All sources that have calculated simulation points
            valid_sources = [s for s in self.sources if s.get('xs_pixels') is not None]
            if not valid_sources:
                QMessageBox.warning(self, "Error", "No simulation points loaded.")
                return

            unit_label = r'($\lambda$)' if self._units_are_lambda() else '(mm)'
            multi = len(valid_sources) > 1

            # FWHM per source
            per_source_fwhms = [self._calculate_all_fwhms(s) for s in valid_sources]
            n_points = max(len(f) for f in per_source_fwhms)

            # Dynamic headers, grouped by AXIS instead of by
            # file: X (file 1) - X (file 2) - Y (file 1) - ...
            axis_names = ("X", "Y", "Z")
            headers = ["Point No."]
            for axis_name in axis_names:
                for s in valid_sources:
                    tag = f" — {s.get('label')}" if multi else ""
                    headers.append(f"FWHM {axis_name} {unit_label}{tag}")

            # Rows in the same order per axis. Each fwhm_list carries tuples
            # (point, fwhm_x, fwhm_y, fwhm_z) -> indices 1, 2, 3 = X, Y, Z
            rows = []
            for idx in range(n_points):
                row = [idx]
                for axis_pos in (1, 2, 3):  # 1=X, 2=Y, 3=Z within each tuple
                    for fwhm_list in per_source_fwhms:
                        if idx < len(fwhm_list):
                            row.append(fwhm_list[idx][axis_pos])
                        else:
                            row.append(None)
                rows.append(tuple(row))

            self.dlg = DialogWindow(self)
            self.dlg.set_headers(headers)
            if multi:
                self.dlg.setWindowTitle("FWHM Comparison")
            self.dlg.load_fwhm_table(rows)
            self.dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open the FWHM table: {str(e)}")


class ScatterDialog(ProfilePlotMixin, QDialog):
    # Add lambda_value to the constructor
    def __init__(self, xs, ys, zs, x=None, y=None, z=None, data=None, lambda_value=1.0):
        super().__init__()

        # Set up the dialog UI
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Keep "flat" attributes for compatibility with the rest of the code
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.x = x
        self.y = y
        self.z = z
        self.data = data
        self.lambda_value = lambda_value
        self.mi_lineedit = self.ui.InputIndex

        sources = [{
            'data': data, 'x': x, 'y': y, 'z': z,
            'xs': xs, 'ys': ys, 'zs': zs,
            'lambda_value': lambda_value,
            'label': 'File', 'color': None,
        }]

        # All the X/Y/Z profile, FWHM, peaks, zoom, and table logic
        # lives in ProfilePlotMixin (see _init_profile_plots).
        self._init_profile_plots(sources, self.ui.combbprincipal)

class DialogWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)  
        self.ui.tableWidget.verticalHeader().setVisible(False)
        self.ui.butexport.clicked.connect(self.save_table)

    # Original sizes defined in ui_fwhm.py, used as a base to
    # calculate how much to enlarge the window when there are more than 4 columns.
    _BASE_DIALOG_W, _BASE_FRAME_W, _BASE_TABLE_W = 557, 501, 511
    _BASE_COLUMNS = 4
    _EXTRA_COL_WIDTH = 120  # approximate width reserved per extra column

    def set_headers(self, headers):
        """
            1 file -> ["Point No.", "FWHM X (mm)", "FWHM Y (mm)", "FWHM Z (mm)"]
            2 files -> ["Point No.",
                           "X (mm) — File 1", "X (mm) — File 2",
                           "Y (mm) — File 1", "Y (mm) — File 2",
                           "Z (mm) — File 1", "Z (mm) — File 2"]
        """
        self.ui.tableWidget.setColumnCount(len(headers))
        for col, text in enumerate(headers):
            self.ui.tableWidget.setHorizontalHeaderItem(col, QTableWidgetItem(text))
        self._resize_for_columns(len(headers))

    def _resize_for_columns(self, n_columns):
        """
        ui_fwhm.py positions frame/tableWidget with fixed geometry (no
        layouts), so if there are more than 4 columns the window has to be
        enlarged manually so nothing gets cut off.
        """
        extra_cols = max(0, n_columns - self._BASE_COLUMNS)
        extra_width = extra_cols * self._EXTRA_COL_WIDTH

        new_dialog_w = self._BASE_DIALOG_W + extra_width
        new_frame_w = self._BASE_FRAME_W + extra_width
        new_table_w = self._BASE_TABLE_W + extra_width

        self.resize(new_dialog_w, self.height())

        frame_geo = self.ui.frame.geometry()
        self.ui.frame.setGeometry(frame_geo.x(), frame_geo.y(), new_frame_w, frame_geo.height())

        table_geo = self.ui.tableWidget.geometry()
        self.ui.tableWidget.setGeometry(table_geo.x(), table_geo.y(), new_table_w, table_geo.height())

        # Center the title and the export button in the new width
        title_geo = self.ui.F_title.geometry()
        self.ui.F_title.move((new_dialog_w - title_geo.width()) // 2, title_geo.y())

        btn_geo = self.ui.butexport.geometry()
        self.ui.butexport.move((new_dialog_w - btn_geo.width()) // 2, btn_geo.y())

    def load_fwhm_table(self, rows):
        """
        rows: list of tuples, each tuple = (point_no, fwhm_x1, fwhm_x2, ..., fwhm_y1, fwhm_y2, ..., fwhm_z1, fwhm_z2, ...)
        """
        self.ui.tableWidget.setRowCount(len(rows))
        for row_idx, row_values in enumerate(rows):
            for col_idx, value in enumerate(row_values):
                if value is None:
                    text = "N/A"
                elif col_idx == 0:
                    text = str(value)
                else:
                    text = f"{value:.3f}"
                self.ui.tableWidget.setItem(row_idx, col_idx, QTableWidgetItem(text))
        self.ui.tableWidget.resizeColumnsToContents()

    def save_table(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save table", "", "CSV Files (*.csv)")
        if path:
            with open(path, 'w', encoding='utf-8', newline='') as f: # Added newline='' for CSV
                import csv
                writer = csv.writer(f)
                
                # Write header
                headers = [self.ui.tableWidget.horizontalHeaderItem(i).text() for i in range(self.ui.tableWidget.columnCount())]
                writer.writerow(headers)
                
                # Write rows
                for row in range(self.ui.tableWidget.rowCount()):
                    row_data = []
                    for col in range(self.ui.tableWidget.columnCount()):
                        item = self.ui.tableWidget.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)           

class PopupDialog(QDialog):
    def __init__(self, parent=None, data=None, x=None, y=None, z=None, xs=None, ys=None, zs=None):
        super().__init__(parent)
        self.ui = Ui_ScatterDialog()
        self.ui.setupUi(self)
        self.ui.boxxoro.currentIndexChanged.connect(self.show_planes_together)
        self.ui.colorbar.stateChanged.connect(self.show_planes_together)
        self.ui.secondax.stateChanged.connect(self.show_planes_together)
        self.ui.direction.stateChanged.connect(self.show_planes_together)
        self.ui.scttlabel.stateChanged.connect(self.show_planes_together)

        self.data = data
        self.x = x # in meters
        self.y = y # in meters
        self.z = z # in meters
        self.xs = xs
        self.ys = ys
        self.zs = zs           

        self.ui.graph_scatter.clicked.connect(self.show_planes_together)

   
    def show_planes_together(self):
        try:
            x_val_str = self.ui.X_value_s.text()
            y_val_str = self.ui.Y_Value_s.text()
            if not x_val_str or not y_val_str:
                QMessageBox.warning(self, "Error", "Please enter values for X and Y.")
                return
            x_val_mm = float(x_val_str)
            y_val_mm = float(y_val_str)
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numbers for X and Y.")
            return
        
        if self.data is None or self.x is None or self.y is None or self.z is None:
            QMessageBox.warning(self, "Error", "Data not loaded.")
            return

        opcion = self.ui.boxxoro.currentText()
        if opcion == "Circle":
            marker_style = 'o'
        elif opcion == "Cross":
            marker_style = 'x'
        else:
            marker_style = 'o'  # Default

        # --- Interpolation for plane Y = y_val_mm ---
        img_y = get_interpolated_plane(
            self.data, self.y.flatten(), y_val_mm, 'y'
        )
        x_mm = self.x.flatten() * 1000
        z_mm = self.z.flatten() * 1000


        # --- Interpolation for plane X = x_val_mm ---
        img_x = get_interpolated_plane(
            self.data, self.x.flatten(), x_val_mm, 'x'
        )
        y_mm = self.y.flatten() * 1000
        # z_mm is already defined

        # --- Show only scatter near the plane ---
        xs_mm = self.xs.flatten() * 1000 if self.xs is not None else np.array([])
        ys_mm = self.ys.flatten() * 1000 if self.ys is not None else np.array([])
        zs_mm = self.zs.flatten() * 1000 if self.zs is not None else np.array([])

        tolerance_mm = 0.5  # tolerance in millimeters (this is the resolution we're using to project the scatter)

        # For the Y plane: only points with ys_mm near y_val_mm
        mask_y = np.abs(ys_mm - y_val_mm) < tolerance_mm
        scatter_y_x = xs_mm[mask_y]
        scatter_y_z = zs_mm[mask_y]
        scatter_y_idx = np.where(mask_y)[0]

        # For the X plane: only points with xs_mm near x_val_mm
        mask_x = np.abs(xs_mm - x_val_mm) < tolerance_mm
        scatter_x_y = ys_mm[mask_x]
        scatter_x_z = zs_mm[mask_x]
        scatter_x_idx = np.where(mask_x)[0]

            # Create a new figure and canvas
        fig = Figure(figsize=(10, 5))
        canvas = FigureCanvas(fig)
        
        # Create the subplots
        axes = fig.subplots(1, 2)

        # Plane Y = y_val_mm
        img_plot_0 = axes[0].imshow(img_y.T, cmap='gray', origin='lower', aspect='auto', 
                                    extent=[x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]])
        axes[0].scatter(scatter_y_x, scatter_y_z, c='r', marker= marker_style, label='Scatter')
        axes[0].set_title(f'Y={y_val_mm:.2f} mm')
        axes[0].set_xlabel('X (mm)')
        axes[0].set_ylabel('Z (mm)')
        axes[0].legend()
        axes[0].set_aspect('equal', adjustable='box')

        # Plane X = x_val_mm
        img_plot_1 =axes[1].imshow(img_x.T, cmap='gray', origin='lower', aspect='auto', 
                                   extent=[y_mm[0], y_mm[-1], z_mm[0], z_mm[-1]])
        axes[1].scatter(scatter_x_y, scatter_x_z, c='r', marker= marker_style, label='Scatter')
        axes[1].set_title(f'X={x_val_mm:.2f} mm')
        axes[1].set_xlabel('Y (mm)')
        axes[1].set_ylabel('Z (mm)')
        axes[1].legend()
        axes[1].set_aspect('equal', adjustable='box')

        if self.ui.secondax.isChecked():
            # Show the normal values on the Y axis
            axes[1].set_ylabel('Z (mm)')
            axes[1].tick_params(axis='y', which='both', labelleft=True, left=True)
        else:
            # Show nothing on the Y axis
            axes[1].set_ylabel("")
            axes[1].set_yticks([])  # Remove ticks
            axes[1].tick_params(axis='y', which='both', labelleft=False, left=False)  # Remove lines and labels

        if self.ui.direction.isChecked():
            axes[0].set_ylim(z_mm[0], z_mm[-1])
            axes[1].set_ylim(z_mm[0], z_mm[-1])
        else:
            axes[0].set_ylim(z_mm[-1], z_mm[0])
            axes[1].set_ylim(z_mm[-1], z_mm[0])    
        

        if self.ui.scttlabel.isChecked():
            # For the first scatter
            for idx, (x, z) in zip(scatter_y_idx, zip(scatter_y_x, scatter_y_z)):
                axes[0].text(x, z, f"No. {idx}", color='yellow', fontsize=8, ha='center', va='bottom') # Fixed to idx
            # For the second scatter
            for idx, (y, z) in zip(scatter_x_idx, zip(scatter_x_y, scatter_x_z)):
                axes[1].text(y, z, f"No. {idx}", color='yellow', fontsize=8, ha='center', va='bottom') # Fixed to idx

        if self.ui.colorbar.isChecked():
            fig.colorbar(img_plot_1, ax=axes[1], orientation='vertical', label='Amplitude (dB)')        

        fig.subplots_adjust(wspace=0.05)
        fig.tight_layout()


        # Clean up the previous layout
        if hasattr(self, 'frame_layout'):
            for i in reversed(range(self.frame_layout.count())): 
                widget = self.frame_layout.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
        else:
            # Create a vertical layout for the frame if it doesn't exist
            self.frame_layout = QVBoxLayout(self.ui.frame)
            self.ui.frame.setLayout(self.frame_layout)

        # Add the canvas to the frame
        self.frame_layout.addWidget(canvas)

        # Optional: Add a navigation toolbar
        if not hasattr(self, 'toolbar'):
            self.toolbar = NavigationToolbar2QT(canvas, self.ui.frame)
            self.frame_layout.addWidget(self.toolbar)
        else:
            # If it already exists, just add it (this could be a bug if duplicated)
            # Better to clean up and add
            self.frame_layout.addWidget(self.toolbar)

class CDDialog(ProfilePlotMixin, QDialog):
    def __init__(self, parent=None, data=None, x=None, y=None, z=None, xs=None, ys=None, zs=None,
                 lambda_value=1.0, file_label="Main File"):
        super().__init__(parent)
        self.ui = Ui_CDDialog()
        self.ui.setupUi(self)
        self.file_label = file_label  # name to display in legends/messages for file 1

        #Connect the buttons 
        self.ui.graph_scatter_cd.clicked.connect(self.show_planes_together)
        self.ui.boxcd.currentIndexChanged.connect(self.show_planes_together)
        self.ui.colorbar_cd.stateChanged.connect(self.show_planes_together)
        self.ui.secondax_cd.stateChanged.connect(self.show_planes_together)
        self.ui.direction_cd.stateChanged.connect(self.show_planes_together)
        self.ui.scttlabel_cd.stateChanged.connect(self.show_planes_together)
        
        # Data from the first file
        self.data = data
        self.x = x
        self.y = y
        self.z = z
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.lambda_value = lambda_value
        
        # Data from the second file (to compare)
        self.data_2 = None
        self.x_2 = None
        self.y_2 = None
        self.z_2 = None
        self.lambda_value_2 = 1.0
        
        # Ask for the file when opening
        if not self.load_comparison_file():
            self.close()  # Close the dialog if no file is selected
            return
        
        # Now set up the UI
        self.setup_ui()

    def load_comparison_file(self):
        """
        Opens a dialog to select the file to compare
        Returns True if loaded successfully, False if cancelled
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select File to Compare", 
            "", 
            "MAT files (*.mat)"
        )
        
        if not file_path:
            QMessageBox.warning(self, "Error", "No file selected. The dialog will close.")
            return False
        
        try:
            mat_data = scipy.io.loadmat(file_path)
            self.data_2 = mat_data.get('data', None)
            
            # Load the x, y, z coordinates
            self.x_2 = mat_data.get('x', None)
            self.y_2 = mat_data.get('y', None)
            self.z_2 = mat_data.get('z', None)

            self.xs_2 = mat_data.get('xs', None)
            self.ys_2 = mat_data.get('ys', None)
            self.zs_2 = mat_data.get('zs', None)

            # Get lambda_value from the file
            self.lambda_value_2 = MyWidget.get_param_value(mat_data, 'lambda')
            if isinstance(self.lambda_value_2, str) or self.lambda_value_2 == 0 or self.lambda_value_2 is None:
                self.lambda_value_2 = 1.0
            
            # Convert to one-dimensional arrays if necessary
            if self.x_2 is not None and len(self.x_2.shape) > 1:
                self.x_2 = self.x_2.ravel()
            if self.y_2 is not None and len(self.y_2.shape) > 1:
                self.y_2 = self.y_2.ravel()
            if self.z_2 is not None and len(self.z_2.shape) > 1:
                self.z_2 = self.z_2.ravel()

            if self.data_2 is None:
                QMessageBox.critical(self, "Error", "No valid data found in the selected file.")
                return False
            
            QMessageBox.information(self, "Success", f"File loaded successfully:\n{file_path.split('/')[-1]}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading file: {e}")
            return False

    def setup_ui(self):
        """Sets up the interface after loading the file"""
        # Configure the comboBox options (only Volume rendering and Isosurface)
        self.ui.combbprincipal.clear()
        self.ui.combbprincipal.addItem("Volume rendering")
        self.ui.combbprincipal.addItem("Isosurface")
        
        # Connect the comboBox to updating the visualizations
        self.ui.combbprincipal.currentIndexChanged.connect(self.update_both_visualizations)

        # Create the Mayavi visualization for Frame1 (first file - data)
        # The frames already have a layout from the TabWidget, so clear it first
        if hasattr(self.ui, 'Frame1'):
            self.setup_3d_visualization_frame1()
        
        # Create the Mayavi visualization for Frame2 (second file - data_2)
        if hasattr(self.ui, 'Frame2'):
            self.setup_3d_visualization_frame2()

        # Configure the "FW" tab: X/Y/Z profiles of BOTH files
        # overlaid on the same graphs, reusing ProfilePlotMixin.
        self._setup_comparison_profiles()

    def _setup_comparison_profiles(self):
        """Initializes the comparative profiles tab (FrameX/Y/Z) with
        the data from both files, using ScatterDialog's logic."""
        sources = [
            {
                'data': self.data, 'x': self.x, 'y': self.y, 'z': self.z,
                'xs': self.xs, 'ys': self.ys, 'zs': self.zs,
                'lambda_value': self.lambda_value,
                'label': self.file_label, 'color': 'tab:blue',
            },
            {
                'data': self.data_2, 'x': self.x_2, 'y': self.y_2, 'z': self.z_2,
                'xs': self.xs_2, 'ys': self.ys_2, 'zs': self.zs_2,
                'lambda_value': self.lambda_value_2,
                'label': 'Secondary File', 'color': 'tab:orange',
            },
        ]
        # combbprincipal_2 is the "Simulation peaks / Manual input" selector
        # for the FW tab (combbprincipal is already used by Volume/Isosurface)
        self._init_profile_plots(sources, self.ui.combbprincipal_2)

    # ========== FRAME 1 (Original data) ==========
    def setup_3d_visualization_frame1(self):
        """Sets up the 3D visualization in Frame1 for the first file"""
        try:
            self.visualization_1 = VisualizationWidget()
            
            # Clear the frame's existing layout if it has one
            if self.ui.Frame1.layout():
                self.clear_layout(self.ui.Frame1.layout())
            
            # Create a new layout
            layout = QVBoxLayout(self.ui.Frame1)
            self.ui.Frame1.setLayout(layout)
            
            # Add the visualization control to the frame
            self.visualization_control_1 = self.visualization_1.edit_traits(parent=self, kind='subpanel').control
            layout.addWidget(self.visualization_control_1)
            
            # Generate the 3D visualization
            self.plot_3d_volume_frame1()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error setting up 3D visualization Frame1: {e}")
    
    def plot_3d_volume_frame1(self):
        """Plots the 3D volume of the first file in Frame1"""
        if self.data is None:
            return
        
        try:
            mlab.clf(figure=self.visualization_1.scene.mayavi_scene)
            self.visualization_1.scene.background = (0.2, 0.2, 0.2)
            src = mlab.pipeline.scalar_field(self.data, figure=self.visualization_1.scene.mayavi_scene)
            
            # Get the comboBox option
            choice = self.get_visualization_choice()
            
            if choice == "Isosurface":
                mlab.contour3d(self.data, contours=8, opacity=0.5, 
                              figure=self.visualization_1.scene.mayavi_scene)
            else:  # "Volume rendering" by default
                mlab.pipeline.volume(src, figure=self.visualization_1.scene.mayavi_scene)
            
            # Determine the labels and ranges
            scale_factor = 1000
            xlabel = 'X (mm)'
            ylabel = 'Y (mm)'
            zlabel = 'Z (mm)'
            
            if self.x is not None and self.y is not None and self.z is not None:
                x_min, x_max = self.x[0] * scale_factor, self.x[-1] * scale_factor
                y_min, y_max = self.y[0] * scale_factor, self.y[-1] * scale_factor
                z_min, z_max = self.z[0] * scale_factor, self.z[-1] * scale_factor
                
                mlab.axes(
                    xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                    ranges=np.array([x_min, x_max, y_min, y_max, z_min, z_max]).flatten(),
                    figure=self.visualization_1.scene.mayavi_scene
                )
            else:
                mlab.axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                         figure=self.visualization_1.scene.mayavi_scene)
            
            mlab.colorbar(orientation='vertical', nb_labels=5)
            self.visualization_1.scene.camera.zoom(1.5)
            self.visualization_1.scene.render()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting 3D volume Frame1: {e}")
    
    # ========== FRAME 2 (Data_2 from the second file) ==========
    def setup_3d_visualization_frame2(self):
        """Sets up the 3D visualization in Frame2 for the second file"""
        try:
            self.visualization_2 = VisualizationWidget()
            
            # Clear the frame's existing layout if it has one
            if self.ui.Frame2.layout():
                self.clear_layout(self.ui.Frame2.layout())
            
            # Create a new layout
            layout = QVBoxLayout(self.ui.Frame2)
            self.ui.Frame2.setLayout(layout)
            
            # Add the visualization control to the frame
            self.visualization_control_2 = self.visualization_2.edit_traits(parent=self, kind='subpanel').control
            layout.addWidget(self.visualization_control_2)
            
            # Generate the 3D visualization
            self.plot_3d_volume_frame2()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error setting up 3D visualization Frame2: {e}")
    
    def plot_3d_volume_frame2(self):
        """Plots the 3D volume of the second file in Frame2"""
        if self.data_2 is None:
            return
        
        try:
            mlab.clf(figure=self.visualization_2.scene.mayavi_scene)
            self.visualization_2.scene.background = (0.2, 0.2, 0.2)
            src = mlab.pipeline.scalar_field(self.data_2, figure=self.visualization_2.scene.mayavi_scene)
            
            # Get the comboBox option
            choice = self.get_visualization_choice()
            
            if choice == "Isosurface":
                mlab.contour3d(self.data_2, contours=8, opacity=0.5, 
                              figure=self.visualization_2.scene.mayavi_scene)
            else:  # "Volume rendering" by default
                mlab.pipeline.volume(src, figure=self.visualization_2.scene.mayavi_scene)
            
            # Determine the labels and ranges
            scale_factor = 1000
            xlabel = 'X (mm)'
            ylabel = 'Y (mm)'
            zlabel = 'Z (mm)'
            
            if self.x_2 is not None and self.y_2 is not None and self.z_2 is not None:
                x_min, x_max = self.x_2[0] * scale_factor, self.x_2[-1] * scale_factor
                y_min, y_max = self.y_2[0] * scale_factor, self.y_2[-1] * scale_factor
                z_min, z_max = self.z_2[0] * scale_factor, self.z_2[-1] * scale_factor
                
                mlab.axes(
                    xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                    ranges=np.array([x_min, x_max, y_min, y_max, z_min, z_max]).flatten(),
                    figure=self.visualization_2.scene.mayavi_scene
                )
            else:
                mlab.axes(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel,
                         figure=self.visualization_2.scene.mayavi_scene)
            
            mlab.colorbar(orientation='vertical', nb_labels=5)
            self.visualization_2.scene.camera.zoom(1.5)
            self.visualization_2.scene.render()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting 3D volume Frame2: {e}")
    
    # ========== HELPER METHODS ==========
    def clear_layout(self, layout):
        """Clears all widgets from a layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def get_visualization_choice(self):
        """Gets the comboBox option"""
        if hasattr(self.ui, 'combbprincipal'):
            return self.ui.combbprincipal.currentText()
        return "Volume rendering"
    
    def update_both_visualizations(self):
        """Updates BOTH frames when the comboBox changes"""
        self.plot_3d_volume_frame1()
        self.plot_3d_volume_frame2()

    def show_planes_together(self):
        # --- Input validation ---
        try:
            x_val_str = self.ui.X_value_cd.text()
            y_val_str = self.ui.Y_Value_cd.text()
            if not x_val_str or not y_val_str:
                QMessageBox.warning(self, "Error", "Please enter values for X and Y.")
                return
            x_val_mm = float(x_val_str)
            y_val_mm = float(y_val_str)
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter valid numbers for X and Y.")
            return

        if self.data is None or self.x is None or self.y is None or self.z is None:
            QMessageBox.warning(self, "Error", "Data from file 1 not loaded.")
            return

        if self.data_2 is None or self.x_2 is None or self.y_2 is None or self.z_2 is None:
            QMessageBox.warning(self, "Error", "Data from file 2 not loaded.")
            return

        # --- Marker style ---
        opcion = self.ui.boxcd.currentText()
        marker_style = 'x' if opcion == "Cross" else 'o'

        # ================================================================
        # FILE 1 → frame_1
        # ================================================================
        fig1 = Figure(figsize=(10, 5))
        canvas1 = FigureCanvas(fig1)

        img_y1 = get_interpolated_plane(self.data, self.y.flatten(), y_val_mm, 'y')
        img_x1 = get_interpolated_plane(self.data, self.x.flatten(), x_val_mm, 'x')

        x_mm1 = self.x.flatten() * 1000
        y_mm1 = self.y.flatten() * 1000
        z_mm1 = self.z.flatten() * 1000

        axes1 = fig1.subplots(1, 2)

        # Scatter points for file 1
        if self.xs is not None and self.ys is not None and self.zs is not None:
            xs_mm = self.xs.flatten() * 1000
            ys_mm = self.ys.flatten() * 1000
            zs_mm = self.zs.flatten() * 1000

            # Plane Y=y_val_mm → scatter in X-Z
            mask_y1 = np.abs(ys_mm - y_val_mm) < 5  # 5 mm tolerance
            scatter_y_x1 = xs_mm[mask_y1]
            scatter_y_z1 = zs_mm[mask_y1]
            scatter_y_idx1 = np.where(mask_y1)[0]

            # Plane X=x_val_mm → scatter in Y-Z
            mask_x1 = np.abs(xs_mm - x_val_mm) < 5
            scatter_x_y1 = ys_mm[mask_x1]
            scatter_x_z1 = zs_mm[mask_x1]
            scatter_x_idx1 = np.where(mask_x1)[0]
        else:
            scatter_y_x1 = scatter_y_z1 = scatter_y_idx1 = np.array([])
            scatter_x_y1 = scatter_x_z1 = scatter_x_idx1 = np.array([])

        img_plot1_0 = axes1[0].imshow(img_y1.T, cmap='gray', origin='lower', aspect='auto',
                                    extent=[x_mm1[0], x_mm1[-1], z_mm1[0], z_mm1[-1]])
        axes1[0].scatter(scatter_y_x1, scatter_y_z1, c='r', marker=marker_style, label='Scatter')
        axes1[0].set_title(f'Main File Y={y_val_mm:.2f} mm')
        axes1[0].set_xlabel('X (mm)')
        axes1[0].set_ylabel('Z (mm)')
        axes1[0].legend()
        axes1[0].set_aspect('equal', adjustable='box')

        img_plot1_1 = axes1[1].imshow(img_x1.T, cmap='gray', origin='lower', aspect='auto',
                                    extent=[y_mm1[0], y_mm1[-1], z_mm1[0], z_mm1[-1]])
        axes1[1].scatter(scatter_x_y1, scatter_x_z1, c='r', marker=marker_style, label='Scatter')
        axes1[1].set_title(f'Main File X={x_val_mm:.2f} mm')
        axes1[1].set_xlabel('Y (mm)')
        axes1[1].legend()
        axes1[1].set_aspect('equal', adjustable='box')

        self._apply_shared_options(axes1, z_mm1, scatter_y_x1, scatter_y_z1, scatter_y_idx1,
                                    scatter_x_y1, scatter_x_z1, scatter_x_idx1,
                                    img_plot1_1, fig1)
        fig1.subplots_adjust(wspace=0.05)
        fig1.tight_layout()

        # ================================================================
        # FILE 2 → frame_2
        # ================================================================
        fig2 = Figure(figsize=(10, 5))
        canvas2 = FigureCanvas(fig2)

        img_y2 = get_interpolated_plane(self.data_2, self.y_2.flatten(), y_val_mm, 'y')
        img_x2 = get_interpolated_plane(self.data_2, self.x_2.flatten(), x_val_mm, 'x')

        x_mm2 = self.x_2.flatten() * 1000
        y_mm2 = self.y_2.flatten() * 1000
        z_mm2 = self.z_2.flatten() * 1000

        axes2 = fig2.subplots(1, 2)

        # Scatter points for file 2
        if self.xs_2 is not None and self.ys_2 is not None and self.zs_2 is not None:
            xs2_mm = self.xs_2.flatten() * 1000
            ys2_mm = self.ys_2.flatten() * 1000
            zs2_mm = self.zs_2.flatten() * 1000

            mask_y2 = np.abs(ys2_mm - y_val_mm) < 5
            scatter_y_x2 = xs2_mm[mask_y2]
            scatter_y_z2 = zs2_mm[mask_y2]
            scatter_y_idx2 = np.where(mask_y2)[0]

            mask_x2 = np.abs(xs2_mm - x_val_mm) < 5
            scatter_x_y2 = ys2_mm[mask_x2]
            scatter_x_z2 = zs2_mm[mask_x2]
            scatter_x_idx2 = np.where(mask_x2)[0]
        else:
            scatter_y_x2 = scatter_y_z2 = scatter_y_idx2 = np.array([])
            scatter_x_y2 = scatter_x_z2 = scatter_x_idx2 = np.array([])

        img_plot2_0 = axes2[0].imshow(img_y2.T, cmap='gray', origin='lower', aspect='auto',
                                    extent=[x_mm2[0], x_mm2[-1], z_mm2[0], z_mm2[-1]])
        axes2[0].scatter(scatter_y_x2, scatter_y_z2, c='r', marker=marker_style, label='Scatter')
        axes2[0].set_title(f'Secondary File Y={y_val_mm:.2f} mm')
        axes2[0].set_xlabel('X (mm)')
        axes2[0].set_ylabel('Z (mm)')
        axes2[0].legend()
        axes2[0].set_aspect('equal', adjustable='box')

        img_plot2_1 = axes2[1].imshow(img_x2.T, cmap='gray', origin='lower', aspect='auto',
                                    extent=[y_mm2[0], y_mm2[-1], z_mm2[0], z_mm2[-1]])
        axes2[1].scatter(scatter_x_y2, scatter_x_z2, c='r', marker=marker_style, label='Scatter')
        axes2[1].set_title(f'Secondary File X={x_val_mm:.2f} mm')
        axes2[1].set_xlabel('Y (mm)')
        axes2[1].legend()
        axes2[1].set_aspect('equal', adjustable='box')

        self._apply_shared_options(axes2, z_mm2, scatter_y_x2, scatter_y_z2, scatter_y_idx2,
                                    scatter_x_y2, scatter_x_z2, scatter_x_idx2,
                                    img_plot2_1, fig2)
        fig2.subplots_adjust(wspace=0.05)
        fig2.tight_layout()

        # ================================================================
        # Embed canvas in the corresponding frames
        # ================================================================
        self._embed_canvas(canvas1, self.ui.Frame3)
        self._embed_canvas(canvas2, self.ui.Frame4)


    def _apply_shared_options(self, axes, z_mm,
                            scatter_y_x, scatter_y_z, scatter_y_idx,
                            scatter_x_y, scatter_x_z, scatter_x_idx,
                            img_plot_colorbar, fig):
        """Applies the shared options (secondax, direction, scttlabel, colorbar)."""

        if self.ui.secondax_cd.isChecked():
            axes[1].set_ylabel('Z (mm)')
            axes[1].tick_params(axis='y', which='both', labelleft=True, left=True)
        else:
            axes[1].set_ylabel("")
            axes[1].set_yticks([])
            axes[1].tick_params(axis='y', which='both', labelleft=False, left=False)

        if self.ui.direction_cd.isChecked():
            axes[0].set_ylim(z_mm[0], z_mm[-1])
            axes[1].set_ylim(z_mm[0], z_mm[-1])
        else:
            axes[0].set_ylim(z_mm[-1], z_mm[0])
            axes[1].set_ylim(z_mm[-1], z_mm[0])

        if self.ui.scttlabel_cd.isChecked():
            for idx, (x, z) in zip(scatter_y_idx, zip(scatter_y_x, scatter_y_z)):
                axes[0].text(x, z, f"No. {idx}", color='yellow', fontsize=8, ha='center', va='bottom')
            for idx, (y, z) in zip(scatter_x_idx, zip(scatter_x_y, scatter_x_z)):
                axes[1].text(y, z, f"No. {idx}", color='yellow', fontsize=8, ha='center', va='bottom')

        if self.ui.colorbar_cd.isChecked():
            fig.colorbar(img_plot_colorbar, ax=axes[1], orientation='vertical', label='Amplitude (dB)')


    def _embed_canvas(self, canvas, frame):
        """Clears the frame and inserts the matplotlib canvas."""
        layout = frame.layout()

        if layout is None:
            layout = QVBoxLayout(frame)
            frame.setLayout(layout)
        else:
            # Clear previous widgets except the toolbar if it exists
            for i in reversed(range(layout.count())):
                widget = layout.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)

        layout.addWidget(canvas)

        # Navigation toolbar: one per frame, stored as a unique attribute
        toolbar_attr = f'_toolbar_{frame.objectName()}'
        toolbar = NavigationToolbar2QT(canvas, frame)
        setattr(self, toolbar_attr, toolbar)
        layout.addWidget(toolbar)

if __name__ == "__main__":
    app = QApplication([])
    window = MyWidget()
    window.show()
    sys.exit(app.exec_())
