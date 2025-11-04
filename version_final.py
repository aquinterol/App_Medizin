import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QDialog, QMessageBox, QTableWidgetItem
from mayavi.core.ui.api import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from tvtk.pyface.scene_editor import SceneEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from mayavi import mlab
from ui_pw import Ui_Widget
from ui_sw import Ui_Dialog  
from ui_fwhm import Ui_Form  
from ui_popup import Ui_Form as Ui_ScatterDialog
import scipy.io
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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

        # Estado del botón
        self.scatter_activado = False

        # Conectar los botones, eventos y slider
        self.ui.openFile.clicked.connect(self.open_file)
        self.ui.comboBox.setEnabled(False)
        self.ui.comboBox.currentIndexChanged.connect(self.plot_volume)
        self.ui.horizontalSlider.setEnabled(False)
        self.ui.horizontalSlider.valueChanged.connect(self.update_slice_position)
        self.ui.Figure2D.clicked.connect(self.open_popup_dialog)
       
        
        # Connect the units combobox
        self.ui.comboBox_2.currentIndexChanged.connect(self.update_units)

        # Crear la visualización Mayavi para el layout principal
        self.visualization = VisualizationWidget()
        self.visualization_control = self.visualization.edit_traits(parent=self, kind='subpanel').control
        self.ui.viLayout.addWidget(self.visualization_control)

        # Crear widgets de visualización para las pestañas
        self.visualization_xy = self._create_visualization(self.ui.tabWidget.widget(0))
        self.visualization_yz = self._create_visualization(self.ui.tabWidget.widget(1))
        self.visualization_xz = self._create_visualization(self.ui.tabWidget.widget(2))

        # Variable para almacenar los datos
        self.data = None  # Store converted data
        self.data_shape = None
        self.plane_xy = None
        self.plane_yz = None
        self.plane_xz = None
        self.current_tab_index = 0
        
        # Variables para almacenar las coordenadas x, y, z
        self.x = None
        self.y = None
        self.z = None
        self.lambda_value = 1.0  # Valor predeterminado

        # Conectar el cambio de pestaña
        self.ui.tabWidget.currentChanged.connect(self.tab_changed)    

    def _create_visualization(self, parent_widget):
        vis_widget = VisualizationWidget()
        layout = QVBoxLayout()
        parent_widget.setLayout(layout)
        layout.addWidget(vis_widget.edit_traits(parent=self, kind='subpanel').control)
        return vis_widget

    def get_param_value(self, mat_data, field_name):
        try:
            value = mat_data['param'][field_name][0][0][0]
            
            # --- CORRECCIÓN DE TIPO ---
            # Asegurarse de que el valor sea un escalar de Python
            if isinstance(value, np.ndarray):
                value = value.item() # Extrae el escalar de un array (ej. np.array([1.5]) -> 1.5)
            # --- FIN CORRECCIÓN DE TIPO ---

            return value
        except (KeyError, IndexError):
            return "Parameter wasn't found."    

    def tab_changed(self, index):
        self.current_tab_index = index
        if self.data is not None:
            self.update_slider_range()
            # Actualizar la posición del slider para reflejar el plano actual
            self.update_slice_position(self.ui.horizontalSlider.value())

    def update_slider_range(self):
        if self.data is None:
            return

        # Ajustar el rango del slider según la dimensión actual
        if self.current_tab_index == 0:  # XY (control en Z)
            max_val = self.data.shape[2] - 1
        elif self.current_tab_index == 1:  # YZ (control en X)
            max_val = self.data.shape[0] - 1
        else:  # XZ (control en Y)
            max_val = self.data.shape[1] - 1

        # Preservar la posición relativa cuando se cambia de pestaña
        current_value = self.ui.horizontalSlider.value()
        old_max = self.ui.horizontalSlider.maximum()
        
        # Si el slider ya tiene un rango, calcular la posición relativa
        if old_max > 0:
            relative_position = current_value / old_max
            new_value = int(relative_position * max_val)
        else:
            # De lo contrario, usar el punto medio
            new_value = max_val // 2
        
        # Configurar el rango y valor del slider
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

            # Usar los valores de x, y, z reales para los ejes si están disponibles
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
                # Fallback a los índices si no hay coordenadas reales
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
        
        # Si no se proporcionan unidades, usar las seleccionadas actualmente
        if units is None:
            units = 'mm' if self.ui.comboBox_2.currentText() == "Milimeters" else r'$\lambda$'
            
        # Calcular el factor de escala basado en las unidades
        if units == 'mm':
            scale_factor = 1000 
        else:
            scale_factor = 1 / self.lambda_value 
            
        # Obtener el tamaño total del eje actual y la posición
        if self.current_tab_index == 0:  # XY (control en Z)
            axis_name = 'Z'
            total_size = self.data.shape[2]
            current_pos = position
            
            # Si tenemos valores reales de z, usar esos en lugar del índice
            if self.z is not None and position < len(self.z):
                real_pos = self.z[position].astype(float) * scale_factor
                if isinstance(real_pos, np.ndarray):
                 real_pos = real_pos.item()  # Convert to scalar if it's a single value array
                plane_text = f"XY Plane at Z = {real_pos:.2f} {units}"
            else:
                plane_text = f"XY Plane at Z = {current_pos} {units}"
                
        elif self.current_tab_index == 1:  # YZ (control en X)
            axis_name = 'X'
            total_size = self.data.shape[0]
            current_pos = position
            
            # Si tenemos valores reales de x, usar esos en lugar del índice
            if self.x is not None and position < len(self.x):
                real_pos = self.x[position].astype(float) * scale_factor
                if isinstance(real_pos, np.ndarray):
                 real_pos = real_pos.item()  # Convert to scalar if it's a single value array
                plane_text = f"YZ Plane at X = {real_pos:.2f} {units}"
            else:
                plane_text = f"YZ Plane at X = {current_pos} {units}"
                
        else:  # XZ (control en Y)
            axis_name = 'Y'
            total_size = self.data.shape[1]
            current_pos = position
            
            # Si tenemos valores reales de y, usar esos en lugar del índice
            if self.y is not None and position < len(self.y):
                real_pos = self.y[position].astype(float) * scale_factor
                if isinstance(real_pos, np.ndarray):
                 real_pos = real_pos.item()  # Convert to scalar if it's a single value array
                plane_text = f"XZ Plane at Y = {real_pos:.2f} {units}"
            else:
                plane_text = f"XZ Plane at Y = {current_pos} {units}"

        # Actualizar la etiqueta con el eje y la posición actual
        self.ui.label.setText(plane_text)
        
        # Actualizar también el textInfo
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

        # Obtener las unidades actuales
        selected_unit = 'mm' if self.ui.comboBox_2.currentText() == "Milimeters" else r'$\lambda$'
        self.update_position_label(position, selected_unit)

        try:
            # Asegurarse de que los planos existan antes de actualizarlos
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
            # Si no se proporcionan parámetros, usar valores predeterminados
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
            
            # Configurar planos de corte en las posiciones actuales del slider o por defecto en el medio
            slice_x = self.ui.horizontalSlider.value() if self.current_tab_index == 1 else self.data.shape[0] // 2
            slice_y = self.ui.horizontalSlider.value() if self.current_tab_index == 2 else self.data.shape[1] // 2
            slice_z = self.ui.horizontalSlider.value() if self.current_tab_index == 0 else self.data.shape[2] // 2
            
            # Asegurarse de que los índices de corte estén dentro de los límites
            slice_x = max(0, min(slice_x, self.data.shape[0] - 1))
            slice_y = max(0, min(slice_y, self.data.shape[1] - 1))
            slice_z = max(0, min(slice_z, self.data.shape[2] - 1))
            
            # Determinar los rangos para los ejes
            if self.x is not None and self.y is not None and self.z is not None:
                x_min, x_max = self.x[0], self.x[-1]
                y_min, y_max = self.y[0], self.y[-1]
                z_min, z_max = self.z[0], self.z[-1]
                
                # Aplicar factor de escala si es necesario
                x_min *= scale_factor
                x_max *= scale_factor
                y_min *= scale_factor
                y_max *= scale_factor
                z_min *= scale_factor
                z_max *= scale_factor
            else:
                # Fallback a los índices si no hay coordenadas reales
                x_min, x_max = 0, self.data.shape[0] * scale_factor
                y_min, y_max = 0, self.data.shape[1] * scale_factor
                z_min, z_max = 0, self.data.shape[2] * scale_factor
            
            # Crear un array NumPy para los rangos
            ranges = np.array([x_min, x_max, y_min, y_max, z_min, z_max]).flatten()

            # XY Visualization (Plano XY)
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

            # YZ Visualization (Plano YZ)
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

            # XZ Visualization (Plano XZ)
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

            # Actualizar etiqueta de posición con las unidades correctas
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
                
                # Cargar las coordenadas x, y, z si están disponibles
                self.x = mat_data.get('x', None)
                self.y = mat_data.get('y', None)
                self.z = mat_data.get('z', None)

                # Cargar las coordenadas x, y, z si están disponibles
                self.xs = mat_data.get('xs', None)
                self.ys = mat_data.get('ys', None)
                self.zs = mat_data.get('zs', None)

                
                # Obtener lambda_value del archivo
                self.lambda_value = self.get_param_value(mat_data, 'lambda')
                if isinstance(self.lambda_value, str) or self.lambda_value == 0 or self.lambda_value is None:
                    self.lambda_value = 1.0  # Valor predeterminado si no se encuentra
                
                # Convertir a arrays unidimensionales si es necesario
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
                    
                    # Configurar el slider y actualizar visualizaciones
                    self.update_slider_range()
                    self.plot_volume()
                    self.update_tab_visualizations()
                    
                    # Obtener unidades actuales para la visualización
                    selected_unit = 'mm' if self.ui.comboBox_2.currentText() == "Milimeters" else r'$\lambda$'
                    
                    # Añadir información sobre las coordenadas
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
        """Grafica en el layout principal"""
        if self.data is None:
            self.ui.textInfo.setText("No data loaded.")
            return

        try:
            # Limpiar la escena de visualización principal
            if not self.scatter_activado:
             mlab.clf(figure=self.visualization.scene.mayavi_scene)
            
            # Configurar el fondo y crear el campo escalar
            self.visualization.scene.background = (0.2, 0.2, 0.2)
            src = mlab.pipeline.scalar_field(self.data, figure=self.visualization.scene.mayavi_scene)
            
            # Seleccionar el tipo de visualización basado en el comboBox
            choice = self.ui.comboBox.currentText()
            if choice == "Isosurface":
                 mlab.contour3d(self.data, contours=8, opacity=0.5)
            elif choice == "Volume rendering":
                mlab.pipeline.volume(mlab.pipeline.scalar_field(self.data, vmin=0, vmax=0.8))

            
            # Determinar las etiquetas y rangos basados en valores reales si están disponibles
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

            # Configurar colorbar y ajustar la vista
            mlab.colorbar(orientation='vertical', nb_labels=5)
            self.visualization.scene.camera.zoom(1.5)
            self.visualization.scene.render()    

            if self.scatter_activado:
                print("Graficando volumen con Scatter activado...")
                
                # Mapear las coordenadas reales (xs, ys, zs) a índices de píxeles
                if self.x is not None and self.y is not None and self.z is not None:
                    # Calcular los índices de píxeles correspondientes a las coordenadas reales
                    x_indices = np.interp(self.xs, (self.x.min(), self.x.max()), (0, self.data.shape[0] - 1))
                    y_indices = np.interp(self.ys, (self.y.min(), self.y.max()), (0, self.data.shape[1] - 1))
                    z_indices = np.interp(self.zs, (self.z.min(), self.z.max()), (0, self.data.shape[2] - 1))
                else:
                    # Si no hay coordenadas reales, usar los índices directamente
                    x_indices = self.xs
                    y_indices = self.ys
                    z_indices = self.zs

                # Agregar los puntos a la escena
                mlab.points3d(
                    x_indices, y_indices, z_indices, 
                    scale_factor=15.0,  # Ajusta este valor para cambiar el tamaño de los puntos
                    color=(1, 0, 0),  # Color rojo
                    figure=self.visualization.scene.mayavi_scene,  # Escena de Mayavi existente 
                )
                # Ocultar los ejes de la escena
                self.scatter_activado = False 

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error plotting volume: {e}")

    def open_scatter_dialog(self):
        try:
            reply = QMessageBox.question(
                self,
                "Datos para Scatter",
                "Do you want to upload new data for scatter plot? (Selecting 'No' will use current data)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Select file", "", 
                    "Archivos CSV (*.csv);;Archivos de texto (*.txt);;Todos los archivos (*)"
                )
                if file_path:
                    import numpy as np
                    arr = np.loadtxt(file_path, delimiter=',')
                    if arr.shape[1] != 3:
                        QMessageBox.critical(self, "Error", "File must have tree columns (xs, ys, zs).")
                        return
                    # Asigna directamente a los atributos de la clase
                    self.xs, self.ys, self.zs = arr[:,0], arr[:,1], arr[:,2]
                else:
                    return
            
            if self.data is None:
                QMessageBox.warning(self, "Error", "No data loaded. Please open a .mat file first.")
                return

            self.scatter_activado = not self.scatter_activado
            self.plot_volume()
            
            # Pasar self.lambda_value al constructor de ScatterDialog
            self.scatter_dialog = ScatterDialog(
                self.xs, self.ys, self.zs, 
                self.x, self.y, self.z, 
                self.data, 
                self.lambda_value # <-- Valor lambda añadido
            )
            
            self.scatter_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo abrir el gráfico de dispersión: {str(e)}")

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

class ScatterDialog(QDialog):
    # Añadir lambda_value al constructor
    def __init__(self, xs, ys, zs, x=None, y=None, z=None, data= None, lambda_value=1.0):
        super().__init__()
        
        # Configurar la interfaz del diálogo
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        
        # Conectar el CheckBox de normalización
        if hasattr(self.ui, 'checkBox'):
            self.ui.checkBox.stateChanged.connect(self.on_normalize_changed)
        else:
            print("ADVERTENCIA: No se encontró 'self.ui.checkBox'.")

        # Conectar el CheckBox de unidades (checkunits)
        if hasattr(self.ui, 'checkunits'):
            self.ui.checkunits.stateChanged.connect(self.on_units_changed)
        else:
            print("ADVERTENCIA: No se encontró 'self.ui.checkunits'.")
        
        # Almacenar lambda_value
        self.lambda_value = lambda_value
        
        # --- CORRECCIÓN DE TIPO ---
        # Asegurarse de que lambda_value sea un escalar
        if isinstance(self.lambda_value, np.ndarray):
            self.lambda_value = self.lambda_value.item()
        # --- FIN CORRECCIÓN DE TIPO ---

        if self.lambda_value == 0: # Evitar división por cero
            self.lambda_value = 1.0

        # Almacenar los valores de coordenadas
        self.xs = xs
        self.ys = ys 
        self.zs = zs
       
        # Almacenar los rangos de coordenadas originales (en metros)
        self.x = x
        self.y = y
        self.z = z

        # Almacenar la data general para el grafico de la amplitud
        self.data = data 
        self.mi_lineedit = self.ui.InputIndex
        self.valor_guardado = 0 

        # Configurar estado inicial de los controles
        self.ui.combbsimu.setEnabled(False)
        self.ui.InputIndex.setEnabled(False)
        self.ui.buttonTable.setEnabled(False)
        self.ui.buttonGraph.setEnabled(False)
        self.ui.buttonTable.clicked.connect(self.open_table_dialog)
       
        # Configurar los lienzos de Matplotlib ANTES de conectar señales
        self.setup_matplotlib_canvases()

        # Conectar el signal de cambio de selección
        self.ui.combbprincipal.currentIndexChanged.connect(self.on_combobox_changed)
        self.ui.InputIndex.textChanged.connect(self.on_input_index_changed)
        self.ui.InputIndey.textChanged.connect(self.on_input_index_changed)
        self.ui.InputIndez.textChanged.connect(self.on_input_index_changed)

        # Connect the Graph button click to the update_manual_graphs function
        self.ui.buttonGraph.clicked.connect(self.update_manual_graphs)

        # Store the connection to safely disconnect later
        self.simu_connection = None

        # Populate simulation points
        self.populate_simulation_points()

        # Inicialización del zoom DEBE ir ANTES de llamar a on_combobox_changed
        self.zoom_window = 4.0 # El zoom siempre se define en mm
        if hasattr(self.ui, 'InputZoom'): 
            self.ui.InputZoom.editingFinished.connect(self.update_zoom_window)
        else:
            print("ADVERTENCIA: No se encontró 'self.ui.InputZoom'. Usando zoom fijo de 4.0")

        # Llamar al método inicialmente para configurar el estado correcto
        self.on_combobox_changed(self.ui.combbprincipal.currentIndex()) 

    def on_normalize_changed(self):
        """
        Se llama cuando el estado de self.ui.checkBox cambia.
        Vuelve a dibujar los gráficos actuales con la nueva configuración de normalización.
        """
        choice = self.ui.combbprincipal.currentText()
        
        if choice == "Simulation peaks ":
            if hasattr(self, 'simu_connection') and self.simu_connection:
                current_index = self.ui.combbsimu.currentIndex()
                if current_index >= 0:
                    self.simu_connection(current_index)
                    
        elif choice == "Manual input ":
            self.update_manual_graphs()
            
    def on_units_changed(self):
        """
        Se llama cuando el estado de self.ui.checkunits cambia.
        Vuelve a dibujar los gráficos actuales con la nueva configuración de unidades.
        """
        # El FWHM en la tabla también debe actualizarse
        if self.ui.combbprincipal.currentText() == "Simulation peaks ":
            if hasattr(self, 'simu_connection') and self.simu_connection:
                current_index = self.ui.combbsimu.currentIndex()
                if current_index >= 0:
                    self.simu_connection(current_index)
                    
        elif self.ui.combbprincipal.currentText() == "Manual input ":
            self.update_manual_graphs()
    
    def _get_plot_coords_and_labels(self):
        """
        Devuelve (x_coords, y_coords, z_coords, xlabel, ylabel, zlabel)
        basado en el estado de self.ui.checkunits.
        Los ejes de coordenadas (self.x, self.y, self.z) están en METROS.
        lambda_value está en METROS.
        """
        if hasattr(self.ui, 'checkunits') and self.ui.checkunits.isChecked():
            # --- Unidades LAMBDA ---
            # (self.x / self.lambda_value) nos da la coordenada en lambda
            x_coords = self.x.flatten() / self.lambda_value
            y_coords = self.y.flatten() / self.lambda_value
            z_coords = self.z.flatten() / self.lambda_value
            
            xlabel = r'X ($\lambda$)'
            ylabel = r'Y ($\lambda$)'
            zlabel = r'Z ($\lambda$)'
        else:
            # --- Unidades MM (Default) ---
            x_coords = self.x.flatten() * 1000.0
            y_coords = self.y.flatten() * 1000.0
            z_coords = self.z.flatten() * 1000.0
            xlabel = 'X (mm)'
            ylabel = 'Y (mm)'
            zlabel = 'Z (mm)'
            
        return x_coords, y_coords, z_coords, xlabel, ylabel, zlabel

    def populate_simulation_points(self):
        self.ui.combbsimu.clear()

        # Desanidar xs si es un array de NumPy con una sola fila
        if isinstance(self.xs, np.ndarray):
            self.xs = self.xs.flatten().tolist()
        elif isinstance(self.xs, list) and len(self.xs) == 1 and isinstance(self.xs[0], (list, np.ndarray)):
            self.xs = list(self.xs[0])  # Extraer lista interna si está anidada
   
        # Verificar si xs tiene elementos
        if self.xs and len(self.xs) > 0:
            for index in range(len(self.xs)):
                # print(f"Iteración {index}: {self.xs[index]}")  # Depuración
                point_text = f"Point {index}"
                self.ui.combbsimu.addItem(point_text)

    def on_input_index_changed(self):
        """Actualiza el valor cuando el usuario cambia el texto en InputIndex."""
        try:
            # El input manual siempre se asume en MILÍMETROS
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

    
    def find_fwhm_points(self, profile, axis):
        """
        Devuelve:
          - extremo izquierdo (mm)
          - extremo derecho (mm)
          - FWHM (mm)
          - nivel half-power (dB)
        NOTA: 'axis' DEBE estar en metros. La función devuelve todo en mm.
        """
        profile = np.array(profile)
        axis = np.array(axis).flatten() # Eje en metros
        max_idx = np.argmax(profile)
        max_value = profile[max_idx]
        half_power_db = max_value - 6

        # Buscar punto a la izquierda
        left_idx = max_idx
        while left_idx > 0 and profile[left_idx - 1] > half_power_db:
            left_idx -= 1
        # Buscar punto a la derecha
        right_idx = max_idx
        while right_idx < len(profile) - 1 and profile[right_idx + 1] > half_power_db:
            right_idx += 1

        def interpolate_point(idx1, idx2):
            if idx1 < 0 or idx1 >= len(axis) or idx2 < 0 or idx2 >= len(axis):
                return axis[max_idx] # Fallback
            if idx1 >= len(profile) or idx2 >= len(profile):
                 return axis[max_idx] # Fallback
                 
            x1_m, y1 = axis[idx1], profile[idx1] # x1 está en metros
            x2_m, y2 = axis[idx2], profile[idx2] # x2 está en metros
            if y1 == y2:
                return x1_m
            # Interpola para encontrar la coordenada en METROS
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
        
        # Devuelve todo en milímetros
        return left_x_m * 1000.0, right_x_m * 1000.0, fwhm_m * 1000.0, half_power_db

    def calculate_all_fwhms(self):
        # Calcula FWHM para todos los puntos de simulación
        fwhm_list = []
        
        # Determinar el factor de escala y la unidad
        if hasattr(self.ui, 'checkunits') and self.ui.checkunits.isChecked():
            # de mm a lambda. (self.lambda_value * 1000.0) es lambda en mm
            scale_factor = 1.0 / (self.lambda_value * 1000.0) 
        else:
            scale_factor = 1.0 # de mm a mm
            
        for idx in range(len(self.xs_pixels)):
            x_profile = self.data[:, self.ys_pixels[idx], self.zs_pixels[idx]]
            y_profile = self.data[self.xs_pixels[idx], :, self.zs_pixels[idx]]
            z_profile = self.data[self.xs_pixels[idx], self.ys_pixels[idx], :]
            
            # Aplicar normalización si es necesario ANTES de calcular FWHM
            if hasattr(self.ui, 'checkBox') and self.ui.checkBox.isChecked():
                if x_profile.size > 0: x_profile = x_profile - np.max(x_profile)
                if y_profile.size > 0: y_profile = y_profile - np.max(y_profile)
                if z_profile.size > 0: z_profile = z_profile - np.max(z_profile)

            # find_fwhm_points devuelve FWHM en mm
            _, _, fwhm_x_mm, _ = self.find_fwhm_points(x_profile, self.x)
            _, _, fwhm_y_mm, _ = self.find_fwhm_points(y_profile, self.y)
            _, _, fwhm_z_mm, _ = self.find_fwhm_points(z_profile, self.z)
            
            # Escalar FWHM según la unidad seleccionada
            fwhm_x = fwhm_x_mm * scale_factor
            fwhm_y = fwhm_y_mm * scale_factor
            fwhm_z = fwhm_z_mm * scale_factor
            
            fwhm_list.append((idx, fwhm_x, fwhm_y, fwhm_z))
            
        return fwhm_list     
            
    def setup_matplotlib_canvases(self):
        """Configurar lienzos de Matplotlib para X, Y, Z con NavigationToolbar"""
        # Para el frame X
        self.figure_x = Figure(figsize=(5, 4), dpi=100)
        self.canvas_x = FigureCanvas(self.figure_x)
        self.toolbar_x = NavigationToolbar2QT(self.canvas_x, self.ui.FrameX)
        layout_x = QVBoxLayout(self.ui.FrameX)
        layout_x.addWidget(self.toolbar_x)
        layout_x.addWidget(self.canvas_x)

        # Para el frame Y
        self.figure_y = Figure(figsize=(5, 4), dpi=100)
        self.canvas_y = FigureCanvas(self.figure_y)
        self.toolbar_y = NavigationToolbar2QT(self.canvas_y, self.ui.FrameY)
        layout_y = QVBoxLayout(self.ui.FrameY)
        layout_y.addWidget(self.toolbar_y)
        layout_y.addWidget(self.canvas_y)

        # Para el frame Z
        self.figure_z = Figure(figsize=(5, 4), dpi=100)
        self.canvas_z = FigureCanvas(self.figure_z)
        self.toolbar_z = NavigationToolbar2QT(self.canvas_z, self.ui.FrameZ)
        layout_z = QVBoxLayout(self.ui.FrameZ)
        layout_z.addWidget(self.toolbar_z)
        layout_z.addWidget(self.canvas_z)

        # Configuración de layouts
        self.figure_x.set_tight_layout(True)
        self.figure_y.set_tight_layout(True)
        self.figure_z.set_tight_layout(True)
        
    def update_zoom_window(self):
        # Asegurarse de que InputZoom existe antes de usarlo
        if not hasattr(self.ui, 'InputZoom'):
            return
            
        texto = self.ui.InputZoom.text().strip()
        try:
            valor = float(texto)
            if valor <= 0:
                raise ValueError
            self.zoom_window = valor # Zoom window siempre se guarda en mm
        except ValueError:
            QMessageBox.warning(self, "Wrong number", "The value must be positive.")
            self.ui.InputZoom.setText(str(self.zoom_window))
            return

        # Actualiza el gráfico según la pestaña activa
        if self.ui.combbprincipal.currentText() == "Simulation peaks " and self.ui.combbsimu.isEnabled():
            idx = self.ui.combbsimu.currentIndex()
            # Vuelve a graficar el perfil seleccionado
            if hasattr(self, 'simu_connection'):
                self.simu_connection(idx)  

    def on_combobox_changed(self, index):
        # Obtener el texto seleccionado
        choice = self.ui.combbprincipal.currentText()

        # Limpiar figuras anteriores
        self.figure_x.clear()
        self.figure_y.clear()
        self.figure_z.clear()

        # Deshabilitar todos los controles primero
        self.ui.combbsimu.setEnabled(False)
        self.ui.InputIndex.setEnabled(False)
        self.ui.InputIndey.setEnabled(False)
        self.ui.InputIndez.setEnabled(False)
        
        # Deshabilitar InputZoom si existe
        if hasattr(self.ui, 'InputZoom'):
            self.ui.InputZoom.setEnabled(False)


        # Habilitar controles basados en la selección
        if choice == "Simulation peaks ":
            self.ui.combbsimu.setEnabled(True)
            self.ui.buttonTable.setEnabled(True)
            self.ui.buttonGraph.setEnabled(False)
            
            # Habilitar InputZoom si existe
            if hasattr(self.ui, 'InputZoom'):
                self.ui.InputZoom.setEnabled(True)
            
            # Mapear coordenadas si se proporcionan rangos originales
            if self.x is not None and self.y is not None and self.z is not None:
                # Asegurar que xs, ys, zs sean arrays planos (1D)
                if self.xs is None:
                    QMessageBox.warning(self, "Error", "Simulation peaks (xs) not found in .mat file.")
                    return
                self.xs = np.array(self.xs).flatten()
                self.ys = np.array(self.ys).flatten()
                self.zs = np.array(self.zs).flatten()
                
                # Calcular los índices de píxeles correspondientes a las coordenadas reales
                # (self.x, y, z están en metros)
                self.xs_pixels = np.interp(self.xs, (self.x.min(), self.x.max()), (0, self.x.shape[0] - 1))
                self.ys_pixels = np.interp(self.ys, (self.y.min(), self.y.max()), (0, self.y.shape[0] - 1))
                self.zs_pixels = np.interp(self.zs, (self.z.min(), self.z.max()), (0, self.z.shape[0] - 1))
                
                self.xs_pixels = np.round(self.xs_pixels).astype(int)
                self.ys_pixels = np.round(self.ys_pixels).astype(int)
                self.zs_pixels = np.round(self.zs_pixels).astype(int)

                def update_profile_plots(point_index):
                    # Graficar perfiles en X, Y, Z
                    x_profile = self.data[:, self.ys_pixels[point_index], self.zs_pixels[point_index]]
                    y_profile = self.data[self.xs_pixels[point_index], :, self.zs_pixels[point_index]]
                    z_profile = self.data[self.xs_pixels[point_index], self.ys_pixels[point_index], :]

                    # Normalizar si el checkbox está activado
                    if hasattr(self.ui, 'checkBox') and self.ui.checkBox.isChecked():
                        if x_profile.size > 0:
                            x_profile = x_profile - np.max(x_profile)
                        if y_profile.size > 0:
                            y_profile = y_profile - np.max(y_profile)
                        if z_profile.size > 0:
                            z_profile = z_profile - np.max(z_profile)

                    # Encontrar los picos máximos
                    x_max_idx = np.argmax(x_profile)
                    y_max_idx = np.argmax(y_profile)
                    z_max_idx = np.argmax(z_profile)

                    # Limpiar figuras anteriores
                    self.figure_x.clear()
                    self.figure_y.clear()
                    self.figure_z.clear()

                    # Obtener coordenadas y etiquetas según el checkbox de unidades
                    x_coords, y_coords, z_coords, xlabel, ylabel, zlabel = self._get_plot_coords_and_labels()
                    
                    # Calcular FWHM (siempre se calcula en mm por la función, usando ejes en metros)
                    left_x_mm, right_x_mm, fwhm_x_mm, half_power_x = self.find_fwhm_points(x_profile, self.x)
                    left_y_mm, right_y_mm, fwhm_y_mm, half_power_y = self.find_fwhm_points(y_profile, self.y)
                    left_z_mm, right_z_mm, fwhm_z_mm, half_power_z = self.find_fwhm_points(z_profile, self.z)

                    # Determinar el factor de escala y la unidad del título
                    if hasattr(self.ui, 'checkunits') and self.ui.checkunits.isChecked():
                        scale_factor = 1.0 / (self.lambda_value * 1000.0) # de mm a lambda
                        unit_label = r'$\lambda$'
                    else:
                        scale_factor = 1.0 # de mm a mm
                        unit_label = 'mm'


                    # --- Gráfico de perfil X ---
                    ax_x = self.figure_x.add_subplot(111)
                    ax_x.plot(x_coords, x_profile) # x_coords ya está en mm o lambda
                    
                    # Escalar puntos
                    peak_x_scaled = x_coords[x_max_idx]
                    left_x_scaled = left_x_mm * scale_factor
                    right_x_scaled = right_x_mm * scale_factor
                    fwhm_x_scaled = fwhm_x_mm * scale_factor
                    
                    ax_x.plot(peak_x_scaled, x_profile[x_max_idx], 'ro')
                    ax_x.plot([left_x_scaled, right_x_scaled], [half_power_x, half_power_x], 'm--')
                    ax_x.plot([left_x_scaled], [half_power_x], 'mv')
                    ax_x.plot([right_x_scaled], [half_power_x], 'mv')
                    
                    x_center_scaled = x_coords[x_max_idx]
                    
                    # El zoom window (self.zoom_window) está en mm. Debo escalarlo.
                    zoom_window_scaled = self.zoom_window * scale_factor
                    
                    ax_x.set_xlim(x_center_scaled - zoom_window_scaled, x_center_scaled + zoom_window_scaled)
                    ax_x.set_title(f'X Profile (FWHM = {fwhm_x_scaled:.3f} {unit_label})')
                    ax_x.set_xlabel(xlabel)
                    ax_x.set_ylabel('Amplitude (dB)')
                    ax_x.grid(True)
                    self.canvas_x.draw()

                    # --- Gráfico de perfil Y ---
                    ax_y = self.figure_y.add_subplot(111)
                    ax_y.plot(y_coords, y_profile)

                    # Escalar puntos
                    peak_y_scaled = y_coords[y_max_idx]
                    left_y_scaled = left_y_mm * scale_factor
                    right_y_scaled = right_y_mm * scale_factor
                    fwhm_y_scaled = fwhm_y_mm * scale_factor
                    
                    ax_y.plot(peak_y_scaled, y_profile[y_max_idx], 'ro')
                    ax_y.plot([left_y_scaled, right_y_scaled], [half_power_y, half_power_y], 'm--')
                    ax_y.plot([left_y_scaled], [half_power_y], 'mv')
                    ax_y.plot([right_y_scaled], [half_power_y], 'mv')
                    
                    y_center_scaled = y_coords[y_max_idx]
                    zoom_window_scaled = self.zoom_window * scale_factor # Re-calculado por claridad
                    
                    ax_y.set_xlim(y_center_scaled - zoom_window_scaled, y_center_scaled + zoom_window_scaled)
                    ax_y.set_title(f'Y Profile (FWHM = {fwhm_y_scaled:.3f} {unit_label})')
                    ax_y.set_xlabel(ylabel)
                    ax_y.set_ylabel('Amplitude (dB)')
                    ax_y.grid(True)
                    self.canvas_y.draw()
                    
                    # --- Gráfico de perfil Z ---
                    ax_z = self.figure_z.add_subplot(111)
                    ax_z.plot(z_coords, z_profile)

                    # Escalar puntos
                    peak_z_scaled = z_coords[z_max_idx]
                    left_z_scaled = left_z_mm * scale_factor
                    right_z_scaled = right_z_mm * scale_factor
                    fwhm_z_scaled = fwhm_z_mm * scale_factor

                    ax_z.plot(peak_z_scaled, z_profile[z_max_idx], 'ro')
                    ax_z.plot([left_z_scaled, right_z_scaled], [half_power_z, half_power_z], 'm--')
                    ax_z.plot([left_z_scaled], [half_power_z], 'mv')
                    ax_z.plot([right_z_scaled], [half_power_z], 'mv')
                    
                    z_center_scaled = z_coords[z_max_idx]
                    zoom_window_scaled = self.zoom_window * scale_factor # Re-calculado por claridad
                    
                    ax_z.set_xlim(z_center_scaled - zoom_window_scaled, z_center_scaled + zoom_window_scaled)
                    ax_z.set_title(f'Z Profile (FWHM = {fwhm_z_scaled:.3f} {unit_label})')
                    ax_z.set_xlabel(zlabel)
                    ax_z.set_ylabel('Amplitude (dB)')
                    ax_z.grid(True)
                    self.canvas_z.draw()

                # Disconnect previous connection if it exists
                if self.simu_connection is not None:
                    try:
                        self.ui.combbsimu.currentIndexChanged.disconnect(self.simu_connection)
                    except TypeError:
                        pass  # No connection exists or already disconnected

                # Connect new signal and store the connection
                self.simu_connection = lambda index: update_profile_plots(index)
                self.ui.combbsimu.currentIndexChanged.connect(self.simu_connection)
                
                # Llamar una vez para graficar el primer punto
                if self.ui.combbsimu.count() > 0:
                    self.simu_connection(0)

            else:
                QMessageBox.critical(self, "Error", "Coordinate ranges not available (self.x, self.y, or self.z is None)")

        elif choice == "Manual input ":
            self.ui.InputIndex.setEnabled(True)
            self.ui.InputIndey.setEnabled(True)
            self.ui.InputIndez.setEnabled(True)
            self.ui.buttonGraph.setEnabled(True)
            self.ui.buttonTable.setEnabled(False)
            # InputZoom permanece deshabilitado (configurado al inicio de la función)
    
            try:
                # Intentar convertir el texto actual a float (siempre en mm)
                current_text_x = self.ui.InputIndex.text().strip()
                self.valor_guardado_x = float(current_text_x) if current_text_x else 0.0
                
                current_text_y = self.ui.InputIndey.text().strip()
                self.valor_guardado_y = float(current_text_y) if current_text_y else 0.0
                
                current_text_z = self.ui.InputIndez.text().strip()
                self.valor_guardado_z = float(current_text_z) if current_text_z else 0.0
                
            except ValueError:
                # Si la conversión falla, establecer un valor predeterminado
                self.valor_guardado_x = 0.0
                self.valor_guardado_y = 0.0
                self.valor_guardado_z = 0.0
                self.ui.InputIndex.setText('0')
                self.ui.InputIndey.setText('0')
                self.ui.InputIndez.setText('0')
            
            # Call the update_manual_graphs method to display initial graphs
            if self.data is not None and self.x is not None:
                self.update_manual_graphs()
        
        elif choice == "Find peaks ":
            # self.find_peaks_graph() # Esta función no está definida, la comento
            print("Find peaks no implementado")
            pass

    def update_manual_graphs(self):
        """Update graphs based on the current manual input value"""
        try:
            # Los valores guardados están en milímetros
            x_mm = self.valor_guardado_x
            y_mm = self.valor_guardado_y
            z_mm = self.valor_guardado_z

            # Verificar que el índice esté dentro de los límites de la matriz
            if self.data is not None and self.x is not None and self.y is not None and self.z is not None:
                # Obtener los rangos válidos en milímetros
                x_range_mm = (self.x.flatten().min() * 1000, self.x.flatten().max() * 1000)
                y_range_mm = (self.y.flatten().min() * 1000, self.y.flatten().max() * 1000)
                z_range_mm = (self.z.flatten().min() * 1000, self.z.flatten().max() * 1000)
                
                # Verificar si algún valor está fuera de rango
                error_messages = []
                if not (x_range_mm[0] <= x_mm <= x_range_mm[1]):
                    error_messages.append(f"X value ({x_mm:.2f} mm) is out of range [{x_range_mm[0]:.2f}, {x_range_mm[1]:.2f}] mm")
                if not (y_range_mm[0] <= y_mm <= y_range_mm[1]):
                    error_messages.append(f"Y value ({y_mm:.2f} mm) is out of range [{y_range_mm[0]:.2f}, {y_range_mm[1]:.2f}] mm")
                if not (z_range_mm[0] <= z_mm <= z_range_mm[1]):
                    error_messages.append(f"Z value ({z_mm:.2f} mm) is out of range [{z_range_mm[0]:.2f}, {z_range_mm[1]:.2f}] mm")
                
                # Si hay errores, mostrar mensaje y salir
                if error_messages:
                    error_text = "Input values out of range:\n" + "\n".join(error_messages)
                    QMessageBox.warning(self, "Out of Range Error", error_text)
                    return

                # Si todos los valores están en rango, continuar con la interpolación
                # Convertir de mm a metros para interpolar con self.x,y,z
                x_m = x_mm / 1000.0
                y_m = y_mm / 1000.0
                z_m = z_mm / 1000.0

                # Interpolar para obtener los índices
                x_index = int(np.interp(x_m, 
                                    self.x.flatten(), 
                                    np.arange(self.data.shape[0])))
                y_index = int(np.interp(y_m, 
                                    self.y.flatten(), 
                                    np.arange(self.data.shape[1])))
                z_index = int(np.interp(z_m, 
                                    self.z.flatten(), 
                                    np.arange(self.data.shape[2])))

                # Extraer perfiles de datos
                x_profile = self.data[:, y_index, z_index]  # Perfil a lo largo del eje X
                y_profile = self.data[x_index, :, z_index]  # Perfil a lo largo del eje Y
                z_profile = self.data[x_index, y_index, :]  # Perfil a lo largo del eje Z

                # Normalizar si el checkbox está activado
                if hasattr(self.ui, 'checkBox') and self.ui.checkBox.isChecked():
                    if x_profile.size > 0:
                        x_profile = x_profile - np.max(x_profile)
                    if y_profile.size > 0:
                        y_profile = y_profile - np.max(y_profile)
                    if z_profile.size > 0:
                        z_profile = z_profile - np.max(z_profile)

                # Obtener coordenadas y etiquetas según el checkbox de unidades
                x_coords, y_coords, z_coords, xlabel, ylabel, zlabel = self._get_plot_coords_and_labels()
                unit_label = r'$\lambda$' if (hasattr(self.ui, 'checkunits') and self.ui.checkunits.isChecked()) else 'mm'

                def find_peaks_and_values(profile):
                    max_value = np.max(profile)
                    min_value = np.min(profile)
                    signal_range = max_value - min_value
                    
                    if signal_range == 0:
                        return np.array([np.argmax(profile)]), [profile[np.argmax(profile)]]

                    adaptive_prominence = signal_range * 0.1
                    height_threshold = min_value + signal_range * 0.1

                    peaks, properties = find_peaks(profile,
                                                prominence=adaptive_prominence,
                                                height=height_threshold,
                                                distance=10)
                    if len(peaks) == 0:
                        adaptive_prominence = signal_range * 0.05
                        height_threshold = min_value + signal_range * 0.05
                        peaks, properties = find_peaks(profile,
                                                    prominence=adaptive_prominence,
                                                    height=height_threshold,
                                                    distance=5)
                    if len(peaks) == 0:
                        peaks = np.array([np.argmax(profile)])

                    peak_values = profile[peaks]
                    sorted_indices = np.argsort(peak_values)[::-1]
                    peaks = peaks[sorted_indices]
                    peaks = peaks[:5]
                    return peaks, [profile[peak] for peak in peaks]

                # Limpiar gráficos anteriores
                self.figure_x.clear()
                self.figure_y.clear()
                self.figure_z.clear()

                # --- GUARDAR para callback ---
                self._peak_data = {}

                # Graficar perfil X con todos los picos
                ax_x = self.figure_x.add_subplot(111)
                ax_x.plot(x_coords, x_profile)
                peaks_x, values_x = find_peaks_and_values(x_profile)
                
                # x_coords ya está en la unidad correcta (mm o lambda)
                scaled_peak_coords_x = x_coords[peaks_x]
                
                red_dots_x = ax_x.plot(scaled_peak_coords_x, values_x, 'ro', picker=5)[0]
                for i in range(len(peaks_x)):
                    ax_x.annotate(f'{values_x[i]:.1f}dB', 
                                (scaled_peak_coords_x[i], values_x[i]),
                                xytext=(0, 10), 
                                textcoords='offset points',
                                ha='center',
                                fontsize=8)
                ax_x.set_title(f'X Profile at Y={y_mm:.2f}mm, Z={z_mm:.2f}mm') # El título siempre muestra la coordenada en mm
                ax_x.set_xlabel(xlabel) # El eje X cambia
                ax_x.set_ylabel('Amplitude (dB)')
                ax_x.grid(True)
                self.canvas_x.draw()
                
                # Guardar info para callback (ya está escalada)
                self._peak_data['x'] = {
                    'coords': scaled_peak_coords_x,
                    'values': values_x
                }

                # Graficar perfil Y con todos los picos
                ax_y = self.figure_y.add_subplot(111)
                ax_y.plot(y_coords, y_profile)
                peaks_y, values_y = find_peaks_and_values(y_profile)
                
                scaled_peak_coords_y = y_coords[peaks_y]
                
                red_dots_y = ax_y.plot(scaled_peak_coords_y, values_y, 'ro', picker=5)[0]
                for i in range(len(peaks_y)):
                    ax_y.annotate(f'{values_y[i]:.1f}dB', 
                                (scaled_peak_coords_y[i], values_y[i]),
                                xytext=(0, 10), 
                                textcoords='offset points',
                                ha='center',
                                fontsize=8)
                ax_y.set_title(f'Y Profile at X={x_mm:.2f}mm, Z={z_mm:.2f}mm')
                ax_y.set_xlabel(ylabel)
                ax_y.set_ylabel('Amplitude (dB)')
                ax_y.grid(True)
                self.canvas_y.draw()
                self._peak_data['y'] = {
                    'coords': scaled_peak_coords_y,
                    'values': values_y
                }

                # Graficar perfil Z con todos los picos
                ax_z = self.figure_z.add_subplot(111)
                ax_z.plot(z_coords, z_profile)
                peaks_z, values_z = find_peaks_and_values(z_profile)
                
                scaled_peak_coords_z = z_coords[peaks_z]

                red_dots_z = ax_z.plot(scaled_peak_coords_z, values_z, 'ro', picker=5)[0]
                for i in range(len(peaks_z)):
                    ax_z.annotate(f'{values_z[i]:.1f}dB', 
                                (scaled_peak_coords_z[i], values_z[i]),
                                xytext=(0, 10), 
                                textcoords='offset points',
                                ha='center',
                                fontsize=8)
                ax_z.set_title(f'Z Profile at X={x_mm:.2f}mm, Y={y_mm:.2f}mm')
                ax_z.set_xlabel(zlabel)
                ax_z.set_ylabel('Amplitude (dB)')
                ax_z.grid(True)
                self.canvas_z.draw()
                self._peak_data['z'] = {
                    'coords': scaled_peak_coords_z,
                    'values': values_z
                }

                # --- ACCIÓN DE CLIC PARA PUNTOS ROJOS ---
                def on_pick_x(event):
                    if event.artist != red_dots_x:
                        return
                    ind = event.ind[0]
                    x_val = self._peak_data['x']['coords'][ind]
                    y_val = self._peak_data['x']['values'][ind]
                    QMessageBox.information(self, "Valor del punto",
                        f"Perfil X\nX = {x_val:.2f} {unit_label}\nAmplitud = {y_val:.2f} dB")

                def on_pick_y(event):
                    if event.artist != red_dots_y:
                        return
                    ind = event.ind[0]
                    y_val = self._peak_data['y']['coords'][ind]
                    amp_val = self._peak_data['y']['values'][ind]
                    QMessageBox.information(self, "Valor del punto",
                        f"Perfil Y\nY = {y_val:.2f} {unit_label}\nAmplitud = {amp_val:.2f} dB")

                def on_pick_z(event):
                    if event.artist != red_dots_z:
                        return
                    ind = event.ind[0]
                    z_val = self._peak_data['z']['coords'][ind]
                    amp_val = self._peak_data['z']['values'][ind]
                    QMessageBox.information(self, "Valor del punto",
                        f"Perfil Z\nZ = {z_val:.2f} {unit_label}\nAmplitud = {amp_val:.2f} dB")

                # Quitar conexiones previas para evitar mensajes duplicados
                try:
                    self.canvas_x.mpl_disconnect(self._xpick_cid)
                except AttributeError:
                    pass
                try:
                    self.canvas_y.mpl_disconnect(self._ypick_cid)
                except AttributeError:
                    pass
                try:
                    self.canvas_z.mpl_disconnect(self._zpick_cid)
                except AttributeError:
                    pass
                self._xpick_cid = self.canvas_x.mpl_connect('pick_event', on_pick_x)
                self._ypick_cid = self.canvas_y.mpl_connect('pick_event', on_pick_y)
                self._zpick_cid = self.canvas_z.mpl_connect('pick_event', on_pick_z)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error generating profiles: {str(e)}")

    def open_table_dialog(self):
        try:
            if self.xs_pixels is None: # Comprobación
                 QMessageBox.warning(self, "Error", "No simulation points loaded.")
                 return
            fwhm_list = self.calculate_all_fwhms()
            self.dlg = DialogWindow(self) # Asume que la UI (Ui_Form) tiene las cabeceras por defecto
            
            # Actualizar cabeceras de la tabla dinámicamente
            if hasattr(self.ui, 'checkunits') and self.ui.checkunits.isChecked():
                unit_label = r'($\lambda$)'
            else:
                unit_label = '(mm)'
            
            self.dlg.ui.tableWidget.horizontalHeaderItem(1).setText(f"FWHM X {unit_label}")
            self.dlg.ui.tableWidget.horizontalHeaderItem(2).setText(f"FWHM Y {unit_label}")
            self.dlg.ui.tableWidget.horizontalHeaderItem(3).setText(f"FWHM Z {unit_label}")
            
            self.dlg.load_fwhm_table(fwhm_list)
            self.dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo abrir la tabla FWHM: {str(e)}")

class DialogWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)  
        self.ui.tableWidget.verticalHeader().setVisible(False)
        self.ui.butexport.clicked.connect(self.save_table)

    def load_fwhm_table(self, fwhm_list):
        """
        fwhm_list: lista de tuplas (punto, fwhm_x, fwhm_y, fwhm_z)
        Los valores FWHM ya están escalados (mm o lambda)
        """
        self.ui.tableWidget.setRowCount(len(fwhm_list))
        for row, (pt, fx, fy, fz) in enumerate(fwhm_list):
            self.ui.tableWidget.setItem(row, 0, QTableWidgetItem(str(pt)))
            self.ui.tableWidget.setItem(row, 1, QTableWidgetItem(f"{fx:.3f}"))
            self.ui.tableWidget.setItem(row, 2, QTableWidgetItem(f"{fy:.3f}"))
            self.ui.tableWidget.setItem(row, 3, QTableWidgetItem(f"{fz:.3f}"))    

    
    def save_table(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save table", "", "CSV Files (*.csv)")
        if path:
            with open(path, 'w', encoding='utf-8', newline='') as f: # Añadido newline='' para CSV
                import csv
                writer = csv.writer(f)
                
                # Escribir encabezado
                headers = [self.ui.tableWidget.horizontalHeaderItem(i).text() for i in range(self.ui.tableWidget.columnCount())]
                writer.writerow(headers)
                
                # Escribir filas
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
        self.x = x # en metros
        self.y = y # en metros
        self.z = z # en metros
        self.xs = xs
        self.ys = ys
        self.zs = zs           

        self.ui.graph_scatter.clicked.connect(self.show_planes_together)

    def get_interpolated_plane(self, data, coords, coord_mm, axis):
        # Interpolación lineal de plano de imagen
        coord_m = coord_mm / 1000.0 # Convertir mm de entrada a metros
        idx = np.searchsorted(coords, coord_m) # Coordenadas (self.x/y/z) están en metros
        if idx == 0:
            idx0, idx1 = 0, 1
        elif idx >= len(coords):
            idx0, idx1 = len(coords) - 2, len(coords) - 1
        else:
            idx0, idx1 = idx - 1, idx
        
        if idx1 >= len(coords): # Control de borde
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
            raise ValueError('Eje inválido')
        img_interp = (1 - alpha) * img0 + alpha * img1
        return img_interp

    def show_planes_together(self):
        try:
            x_val_str = self.ui.X_value_s.text()
            y_val_str = self.ui.Y_Value_s.text()
            if not x_val_str or not y_val_str:
                QMessageBox.warning(self, "Error", "Por favor ingresa valores para X e Y.")
                return
            x_val_mm = float(x_val_str)
            y_val_mm = float(y_val_str)
        except ValueError:
            QMessageBox.warning(self, "Error", "Por favor ingresa números válidos para X e Y.")
            return
        
        if self.data is None or self.x is None or self.y is None or self.z is None:
            QMessageBox.warning(self, "Error", "Datos no cargados.")
            return

        opcion = self.ui.boxxoro.currentText()
        if opcion == "Circle":
            marker_style = 'o'
        elif opcion == "Cross":
            marker_style = 'x'
        else:
            marker_style = 'o'  # Por defecto

        # --- Interpolación para plano Y = y_val_mm ---
        img_y = self.get_interpolated_plane(
            self.data, self.y.flatten(), y_val_mm, 'y'
        )
        x_mm = self.x.flatten() * 1000
        z_mm = self.z.flatten() * 1000


        # --- Interpolación para plano X = x_val_mm ---
        img_x = self.get_interpolated_plane(
            self.data, self.x.flatten(), x_val_mm, 'x'
        )
        y_mm = self.y.flatten() * 1000
        # z_mm ya está definido

        # --- Mostrar solo scatter cerca del plano ---
        xs_mm = self.xs.flatten() * 1000 if self.xs is not None else np.array([])
        ys_mm = self.ys.flatten() * 1000 if self.ys is not None else np.array([])
        zs_mm = self.zs.flatten() * 1000 if self.zs is not None else np.array([])

        tolerance_mm = 0.5  # tolerancia en milímetros (es la resolucion que estamos usando para proyectar el scatter)

        # Para el plano Y: solo los puntos con ys_mm cerca de y_val_mm
        mask_y = np.abs(ys_mm - y_val_mm) < tolerance_mm
        scatter_y_x = xs_mm[mask_y]
        scatter_y_z = zs_mm[mask_y]
        scatter_y_idx = np.where(mask_y)[0]

        # Para el plano X: solo los puntos con xs_mm cerca de x_val_mm
        mask_x = np.abs(xs_mm - x_val_mm) < tolerance_mm
        scatter_x_y = ys_mm[mask_x]
        scatter_x_z = zs_mm[mask_x]
        scatter_x_idx = np.where(mask_x)[0]

            # Crear una nueva figura y canvas
        fig = Figure(figsize=(10, 5))
        canvas = FigureCanvas(fig)
        
        # Crear los subplots
        axes = fig.subplots(1, 2)

        # Plano Y = y_val_mm
        img_plot_0 = axes[0].imshow(img_y.T, cmap='gray', origin='lower', aspect='auto', 
                                    extent=[x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]])
        axes[0].scatter(scatter_y_x, scatter_y_z, c='r', marker= marker_style, label='Scatter')
        axes[0].set_title(f'Y={y_val_mm:.2f} mm')
        axes[0].set_xlabel('X (mm)')
        axes[0].set_ylabel('Z (mm)')
        axes[0].legend()
        axes[0].set_aspect('equal', adjustable='box')

        # Plano X = x_val_mm
        img_plot_1 =axes[1].imshow(img_x.T, cmap='gray', origin='lower', aspect='auto', 
                                   extent=[y_mm[0], y_mm[-1], z_mm[0], z_mm[-1]])
        axes[1].scatter(scatter_x_y, scatter_x_z, c='r', marker= marker_style, label='Scatter')
        axes[1].set_title(f'X={x_val_mm:.2f} mm')
        axes[1].set_xlabel('Y (mm)')
        axes[1].set_ylabel('Z (mm)')
        axes[1].legend()
        axes[1].set_aspect('equal', adjustable='box')

        if self.ui.secondax.isChecked():
            # Mostrar los valores normales del eje Y
            axes[1].set_ylabel('Z (mm)')
            axes[1].tick_params(axis='y', which='both', labelleft=True, left=True)
        else:
            # No mostrar nada en el eje Y
            axes[1].set_ylabel("")
            axes[1].set_yticks([])  # Elimina los ticks
            axes[1].tick_params(axis='y', which='both', labelleft=False, left=False)  # Elimina las líneas y las etiquetas

        if self.ui.direction.isChecked():
            axes[0].set_ylim(z_mm[0], z_mm[-1])
            axes[1].set_ylim(z_mm[0], z_mm[-1])
        else:
            axes[0].set_ylim(z_mm[-1], z_mm[0])
            axes[1].set_ylim(z_mm[-1], z_mm[0])    
        

        if self.ui.scttlabel.isChecked():
            # Para el primer scatter
            for idx, (x, z) in zip(scatter_y_idx, zip(scatter_y_x, scatter_y_z)):
                axes[0].text(x, z, f"No. {idx}", color='yellow', fontsize=8, ha='center', va='bottom') # Corregido a idx
            # Para el segundo scatter
            for idx, (y, z) in zip(scatter_x_idx, zip(scatter_x_y, scatter_x_z)):
                axes[1].text(y, z, f"No. {idx}", color='yellow', fontsize=8, ha='center', va='bottom') # Corregido a idx

        if self.ui.colorbar.isChecked():
            fig.colorbar(img_plot_1, ax=axes[1], orientation='vertical', label='Amplitude (dB)')        

        fig.subplots_adjust(wspace=0.05)
        fig.tight_layout()


        # Limpiar el layout anterior
        if hasattr(self, 'frame_layout'):
            for i in reversed(range(self.frame_layout.count())): 
                widget = self.frame_layout.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
        else:
            # Crear un layout vertical para el frame si no existe
            self.frame_layout = QVBoxLayout(self.ui.frame)
            self.ui.frame.setLayout(self.frame_layout)

        # Agregar el canvas al frame
        self.frame_layout.addWidget(canvas)

        # Opcional: Agregar una barra de herramientas de navegación
        if not hasattr(self, 'toolbar'):
            self.toolbar = NavigationToolbar2QT(canvas, self.ui.frame)
            self.frame_layout.addWidget(self.toolbar)
        else:
            # Si ya existe, solo la agregamos (esto podría ser un error si se duplica)
            # Mejor limpiar y añadir
            self.frame_layout.addWidget(self.toolbar)


if __name__ == "__main__":
    app = QApplication([])
    window = MyWidget()
    window.show()
    sys.exit(app.exec_())


#VERSION FINAL