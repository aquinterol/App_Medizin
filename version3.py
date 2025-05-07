import sys
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QTextBrowser, QDialog, QMessageBox
from mayavi.core.ui.api import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from tvtk.pyface.scene_editor import SceneEditor
from traits.api import HasTraits, Instance
from traitsui.api import View, Item
from mayavi import mlab
from ui_pw import Ui_Widget
from ui_sw import Ui_Dialog  
from PyQt5.QtCore import Qt 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import find_peaks

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
            return mat_data['param'][field_name][0][0][0]# Muestra: array([[['algún_valor']]], dtype=object), muestra algún_valor
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
            xlabel = 'X (λ)'
            ylabel = 'Y (λ)'
            zlabel = 'Z (λ)'

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
            units = 'mm' if selected_unit == "Milimeters" else 'λ'
            self.update_position_label(current_pos, units)

        except Exception as e:
            self.ui.textInfo.append(f"Error updating axis labels: {e}")

    def update_position_label(self, position, units=None):
        if self.data is None:
            return
        
        # Si no se proporcionan unidades, usar las seleccionadas actualmente
        if units is None:
            units = 'mm' if self.ui.comboBox_2.currentText() == "Milimeters" else 'λ'
            
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
        selected_unit = 'mm' if self.ui.comboBox_2.currentText() == "Milimeters" else 'λ'
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
                scale_factor = 1000 if selected_unit == "Milimeters" else (1 / self.lambda_value if self.lambda_value != 0 else 1)
                xlabel = 'X (mm)' if selected_unit == "Milimeters" else 'X (λ)'
                ylabel = 'Y (mm)' if selected_unit == "Milimeters" else 'Y (λ)'
                zlabel = 'Z (mm)' if selected_unit == "Milimeters" else 'Z (λ)'
            
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
                if scale_factor != 1.0:
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
            units = 'mm' if 'mm' in xlabel else 'λ'
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
                if isinstance(self.lambda_value, str) or self.lambda_value == 0:
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
                    selected_unit = 'mm' if self.ui.comboBox_2.currentText() == "Milimeters" else 'λ'
                    
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
            xlabel = 'X (mm)' if selected_unit == "Milimeters" else 'X (λ)'
            ylabel = 'Y (mm)' if selected_unit == "Milimeters" else 'Y (λ)'
            zlabel = 'Z (mm)' if selected_unit == "Milimeters" else 'Z (λ)'
            
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
            self.scatter_activado = not self.scatter_activado
            self.plot_volume()
            self.scatter_dialog = ScatterDialog(self.xs,self.ys,self.zs,self.x,self.y,self.z, self.data)
            self.scatter_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo abrir el gráfico de dispersión: {str(e)}")

class ScatterDialog(QDialog):
    def __init__(self, xs, ys, zs, x=None, y=None, z=None, data= None):
        super().__init__()
        
        # Configurar la interfaz del diálogo
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # Almacenar los valores de coordenadas
        self.xs = xs
        self.ys = ys 
        self.zs = zs
       
        # Almacenar los rangos de coordenadas originales
        self.x = x
        self.y = y
        self.z = z

        # Almacenar la data general para el grafico de la amplitud
        self.data = data 
        self.mi_lineedit = self.ui.InputIndex
        self.valor_guardado = 0 

        # Variables para el pan (arrastrar) de gráficos
        self._pan_active = False
        self._pan_start = None
        self._current_ax = None

        # Configurar estado inicial de los controles
        self.ui.combbsimu.setEnabled(False)
        self.ui.InputIndex.setEnabled(False)

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

        # Llamar al método inicialmente para configurar el estado correcto
        self.on_combobox_changed(self.ui.combbprincipal.currentIndex()) 

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
                print(f"Iteración {index}: {self.xs[index]}")  # Depuración
                point_text = f"Point {index}"
                self.ui.combbsimu.addItem(point_text)

    def on_input_index_changed(self):
        """Actualiza el valor cuando el usuario cambia el texto en InputIndex."""
        try:
            self.valor_guardado_x = int(self.ui.InputIndex.text().strip())  # Convierte a entero
            print(f"Nuevo valor guardado: {self.valor_guardado_x}")  # Debug
            self.valor_guardado_y = int(self.ui.InputIndey.text().strip())  # Convierte a entero
            print(f"Nuevo valor guardado: {self.valor_guardado_y}")  # Debug
            self.valor_guardado_z = int(self.ui.InputIndez.text().strip())  # Convierte a entero
            print(f"Nuevo valor guardado: {self.valor_guardado_z}")  # Debug
        except ValueError:
            self.valor_guardado_x = 0  # Si hay error, poner un valor predeterminado
            self.valor_guardado_y = 0  # Si hay error, poner un valor predeterminado
            self.valor_guardado_z = 0  # Si hay error, poner un valor predeterminado
            
    def setup_matplotlib_canvases(self):
        """Configurar lienzos de Matplotlib para X, Y, Z"""
        # Para el frame X
        self.figure_x = Figure(figsize=(5, 4), dpi=100)
        self.canvas_x = FigureCanvas(self.figure_x)
        layout_x = QVBoxLayout(self.ui.FrameX)
        layout_x.addWidget(self.canvas_x)

        # Para el frame Y
        self.figure_y = Figure(figsize=(5, 4), dpi=100)
        self.canvas_y = FigureCanvas(self.figure_y)
        layout_y = QVBoxLayout(self.ui.FrameY)
        layout_y.addWidget(self.canvas_y)

        # Para el frame Z
        self.figure_z = Figure(figsize=(5, 4), dpi=100)
        self.canvas_z = FigureCanvas(self.figure_z)
        layout_z = QVBoxLayout(self.ui.FrameZ)
        layout_z.addWidget(self.canvas_z)   

        # Habilitar interactividad
        self.figure_x.set_tight_layout(True)
        self.figure_y.set_tight_layout(True)
        self.figure_z.set_tight_layout(True)
        
        # Conectar eventos para pan (arrastrar)
        self.canvas_x.mpl_connect('button_press_event', self._on_press)
        self.canvas_x.mpl_connect('button_release_event', self._on_release)
        self.canvas_x.mpl_connect('motion_notify_event', self._on_motion)
        
        self.canvas_y.mpl_connect('button_press_event', self._on_press)
        self.canvas_y.mpl_connect('button_release_event', self._on_release)
        self.canvas_y.mpl_connect('motion_notify_event', self._on_motion)
        
        self.canvas_z.mpl_connect('button_press_event', self._on_press)
        self.canvas_z.mpl_connect('button_release_event', self._on_release)
        self.canvas_z.mpl_connect('motion_notify_event', self._on_motion)

    def _on_press(self, event):
        """Maneja el evento de presionar el botón del mouse para iniciar el arrastre."""
        if event.button == 1 and event.inaxes:  # Botón izquierdo del mouse
            self._pan_active = True
            self._pan_start = (event.xdata, event.ydata)
            self._current_ax = event.inaxes
            self.canvas_x.setCursor(Qt.ClosedHandCursor)
            self.canvas_y.setCursor(Qt.ClosedHandCursor)
            self.canvas_z.setCursor(Qt.ClosedHandCursor)

    def _on_release(self, event):
        """Maneja el evento de soltar el botón del mouse para terminar el arrastre."""
        self._pan_active = False
        self._pan_start = None
        self._current_ax = None
        self.canvas_x.setCursor(Qt.ArrowCursor)
        self.canvas_y.setCursor(Qt.ArrowCursor)
        self.canvas_z.setCursor(Qt.ArrowCursor)

    def _on_motion(self, event):
        """Maneja el evento de movimiento del mouse para realizar el arrastre."""
        if self._pan_active and self._pan_start and event.inaxes == self._current_ax:
            dx = event.xdata - self._pan_start[0]
            dy = event.ydata - self._pan_start[1]
            
            # Obtener los límites actuales
            x_min, x_max = self._current_ax.get_xlim()
            y_min, y_max = self._current_ax.get_ylim()
            
            # Mover los límites en dirección opuesta al movimiento del mouse
            self._current_ax.set_xlim(x_min - dx, x_max - dx)
            self._current_ax.set_ylim(y_min - dy, y_max - dy)
            
            # Actualizar la visualización
            self._current_ax.figure.canvas.draw_idle()
            
            # Actualizar la posición inicial para el próximo movimiento
            self._pan_start = (event.xdata, event.ydata)

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

        # Habilitar controles basados en la selección
        if choice == "Simulation peaks ":
            self.ui.combbsimu.setEnabled(True)
            
            # Mapear coordenadas si se proporcionan rangos originales
            if self.x is not None and self.y is not None and self.z is not None:
                # Asegurar que xs, ys, zs sean arrays planos (1D)
                self.xs = np.array(self.xs).flatten()
                self.ys = np.array(self.ys).flatten()
                self.zs = np.array(self.zs).flatten()
                
                # Calcular los índices de píxeles correspondientes a las coordenadas reales
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

                    # Limpiar figuras anteriores
                    self.figure_x.clear()
                    self.figure_y.clear()
                    self.figure_z.clear()

                    # Gráfico de perfil X
                    ax_x = self.figure_x.add_subplot(111)
                    ax_x.plot(self.x.flatten(), x_profile)
                    ax_x.set_title('X Profile')
                    ax_x.set_xlabel('X')
                    ax_x.set_ylabel('Amplitude')
                    ax_x.figure.canvas.mpl_connect('scroll_event', lambda event: self._zoom_factory(event, ax_x))
                    self.canvas_x.draw()

                    # Gráfico de perfil Y
                    ax_y = self.figure_y.add_subplot(111)
                    ax_y.plot(self.y.flatten(), y_profile)
                    ax_y.set_title('Y Profile')
                    ax_y.set_xlabel('Y')
                    ax_y.set_ylabel('Amplitude')
                    ax_y.figure.canvas.mpl_connect('scroll_event', lambda event: self._zoom_factory(event, ax_y))
                    self.canvas_y.draw()

                    # Gráfico de perfil Z
                    ax_z = self.figure_z.add_subplot(111)
                    ax_z.plot(self.z.flatten(), z_profile)
                    ax_z.set_title('Z Profile')
                    ax_z.set_xlabel('Z')
                    ax_z.set_ylabel('Amplitude')
                    ax_z.figure.canvas.mpl_connect('scroll_event', lambda event: self._zoom_factory(event, ax_z))
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

            else:
                QMessageBox.critical(self, "Error", "Coordinate ranges not available")

        elif choice == "Manual input ":
            self.ui.InputIndex.setEnabled(True)
            self.ui.InputIndey.setEnabled(True)
            self.ui.InputIndez.setEnabled(True)
    
            try:
                # Intentar convertir el texto actual a un entero
                current_text_x = self.ui.InputIndex.text().strip()
                self.valor_guardado_x = int(current_text_x) if current_text_x else 0
                print(f"Valor guardado en manual input: {self.valor_guardado_x}")
                current_text_y = self.ui.InputIndey.text().strip()
                self.valor_guardado_y = int(current_text_y) if current_text_y else 0
                print(f"Valor guardado en manual input: {self.valor_guardado_y}")
                current_text_z = self.ui.InputIndez.text().strip()
                self.valor_guardado_z = int(current_text_z) if current_text_z else 0
                print(f"Valor guardado en manual input: {self.valor_guardado_z}")
            except ValueError:
                # Si la conversión falla, establecer un valor predeterminado
                self.valor_guardado_x = 0
                self.valor_guardado_y = 0
                self.valor_guardado_z = 0
                self.ui.InputIndex.setText('0')
                self.ui.InputIndey.setText('0')
                self.ui.InputIndez.setText('0')
            
            # Call the update_manual_graphs method to display initial graphs
            if hasattr(self, 'xs_pixels'):
                self.update_manual_graphs()
        
        elif choice == "Find peaks ":
            self.find_peaks_graph()

    def update_manual_graphs(self):
        """Update graphs based on the current manual input value"""

        if not hasattr(self, 'xs_pixels') and self.x is not None and self.y is not None and self.z is not None:
            self.xs = np.array(self.xs).flatten()
            self.ys = np.array(self.ys).flatten()
            self.zs = np.array(self.zs).flatten()

            self.xs_pixels = np.interp(self.xs, (self.x.min(), self.x.max()), (0, self.x.shape[0] - 1)).round().astype(int)
            self.ys_pixels = np.interp(self.ys, (self.y.min(), self.y.max()), (0, self.y.shape[0] - 1)).round().astype(int)
            self.zs_pixels = np.interp(self.zs, (self.z.min(), self.z.max()), (0, self.z.shape[0] - 1)).round().astype(int)

        try:
            # Obtener el valor del índice ingresado por el usuario
            index_x = self.valor_guardado_x
            index_y = self.valor_guardado_y
            index_z = self.valor_guardado_z
            
            # Verificar que el índice esté dentro de los límites de la matriz
            if self.data is not None and 0 <= index_x < min(self.data.shape) and 0 <= index_y < min(self.data.shape) and 0 <= index_z < min(self.data.shape):
                x_index = min(index_x, self.data.shape[0]-1)
                y_index = min(index_y, self.data.shape[1]-1)
                z_index = min(index_z, self.data.shape[2]-1)
                
                # Extraer perfiles de datos
                x_profile = self.data[:, y_index, z_index]  # Perfil a lo largo del eje X
                y_profile = self.data[x_index, :, z_index]  # Perfil a lo largo del eje Y
                z_profile = self.data[x_index, y_index, :]  # Perfil a lo largo del eje Z
                
                # Limpiar gráficos anteriores
                self.figure_x.clear()
                self.figure_y.clear()
                self.figure_z.clear()
                
                # Crear arrays para los ejes si están disponibles, o usar ubicaciones creadas por defecto 
                x_coords = self.x.flatten() if self.x is not None else np.arange(self.data.shape[0])
                y_coords = self.y.flatten() if self.y is not None else np.arange(self.data.shape[1])
                z_coords = self.z.flatten() if self.z is not None else np.arange(self.data.shape[2])
                
                # Graficar perfil X
                ax_x = self.figure_x.add_subplot(111)
                ax_x.plot(x_coords, x_profile)
                ax_x.set_title(f'X Profile at Y={y_index}, Z={z_index}')
                ax_x.set_xlabel('X')
                ax_x.set_ylabel('Amplitude')
                ax_x.figure.canvas.mpl_connect('scroll_event', lambda event: self._zoom_factory(event, ax_x))
                self.canvas_x.draw()

                # Graficar perfil Y
                ax_y = self.figure_y.add_subplot(111)
                ax_y.plot(y_coords, y_profile)
                ax_y.set_title(f'Y Profile at X={x_index}, Z={z_index}')
                ax_y.set_xlabel('Y')
                ax_y.set_ylabel('Amplitude')
                ax_y.figure.canvas.mpl_connect('scroll_event', lambda event: self._zoom_factory(event, ax_y))
                self.canvas_y.draw()

                # Graficar perfil Z
                ax_z = self.figure_z.add_subplot(111)
                ax_z.plot(z_coords, z_profile)
                ax_z.set_title(f'Z Profile at X={x_index}, Y={y_index}')
                ax_z.set_xlabel('Z')
                ax_z.set_ylabel('Amplitude')
                ax_z.figure.canvas.mpl_connect('scroll_event', lambda event: self._zoom_factory(event, ax_z))
                self.canvas_z.draw()
            else:
                QMessageBox.warning(self, "Error", f"One of the index is out of range")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error generating profiles: {str(e)}")
                    
    def find_peaks_graph(self):
        """Find and display peaks in the data along each axis"""
        try:
            # Limpiar figuras anteriores
            self.figure_x.clear()
            self.figure_y.clear()
            self.figure_z.clear()
            
            # Usar el punto medio para los perfiles
            mid_y = self.data.shape[1] // 2
            mid_z = self.data.shape[2] // 2
            mid_x = self.data.shape[0] // 2
            
            # Extraer perfiles a lo largo de cada eje
            x_profile = self.data[:, mid_y, mid_z]  # Perfil a lo largo del eje X
            y_profile = self.data[mid_x, :, mid_z]  # Perfil a lo largo del eje Y
            z_profile = self.data[mid_x, mid_y, :]  # Perfil a lo largo del eje Z

            print(f"Data shape: {self.data.shape}")
            print(f"Y profile shape: {y_profile.shape}")
            
            # Crear arrays para los ejes
            x_coords = self.x.flatten() if self.x is not None else np.arange(self.data.shape[0])
            y_coords = self.y.flatten() if self.y is not None else np.arange(self.data.shape[1])
            z_coords = self.z.flatten() if self.z is not None else np.arange(self.data.shape[2])
            
            # Encontrar picos en cada perfil con parámetros ajustados para detectar menos picos
            # Aumentar los valores de prominence y distance para reducir el número de picos detectados
            peaks_x, _ = find_peaks(x_profile, prominence=0.3, distance=10)  # Valores ajustados
            peaks_y, _ = find_peaks(y_profile, prominence=0.3, distance=10)  # Cambiado de threshold a prominence
            peaks_z, _ = find_peaks(z_profile, prominence=0.3, distance=10)  # Valores ajustados
            
            print(f"Found {len(peaks_x)} peaks in X profile")
            print(f"Found {len(peaks_y)} peaks in Y profile")
            print(f"Found {len(peaks_z)} peaks in Z profile")
            
            # Graficar perfil X con picos
            ax_x = self.figure_x.add_subplot(111)
            ax_x.plot(x_coords, x_profile)
            
            # Asegurar que los picos tienen las mismas dimensiones que las coordenadas
            valid_x_peaks = [p for p in peaks_x if p < len(x_coords)]
            if valid_x_peaks:
                ax_x.scatter([x_coords[p] for p in valid_x_peaks], [x_profile[p] for p in valid_x_peaks], 
                            color='r', marker='x', s=50, picker=True, pickradius=5)
                
            ax_x.set_title(f'X Profile with Peaks (Y={mid_y}, Z={mid_z})')
            ax_x.set_xlabel('X')
            ax_x.set_ylabel('Amplitude')
            
            # Graficar perfil Y con picos
            ax_y = self.figure_y.add_subplot(111)
            ax_y.plot(y_coords, y_profile)
            
            # Asegurar que los picos tienen las mismas dimensiones que las coordenadas
            valid_y_peaks = [p for p in peaks_y if p < len(y_coords)]
            if valid_y_peaks:
                ax_y.scatter([y_coords[p] for p in valid_y_peaks], [y_profile[p] for p in valid_y_peaks], 
                            color='r', marker='x', s=50, picker=True, pickradius=5)
                
            ax_y.set_title(f'Y Profile with Peaks (X={mid_x}, Z={mid_z})')
            ax_y.set_xlabel('Y')
            ax_y.set_ylabel('Amplitude')
            
            # Graficar perfil Z con picos
            ax_z = self.figure_z.add_subplot(111)
            ax_z.plot(z_coords, z_profile)
            
            # Asegurar que los picos tienen las mismas dimensiones que las coordenadas
            valid_z_peaks = [p for p in peaks_z if p < len(z_coords)]
            if valid_z_peaks:
                ax_z.scatter([z_coords[p] for p in valid_z_peaks], [z_profile[p] for p in valid_z_peaks], 
                            color='r', marker='x', s=50, picker=True, pickradius=5)
                
            ax_z.set_title(f'Z Profile with Peaks (X={mid_x}, Y={mid_y})')
            ax_z.set_xlabel('Z')
            ax_z.set_ylabel('Amplitude')
            
            # Conectar eventos de picker para mostrar valores al hacer clic
            # Almacenar los datos de cada gráfico para usar en los callbacks
            self.peak_data_x = (x_coords, x_profile, valid_x_peaks, 'X')
            self.peak_data_y = (y_coords, y_profile, valid_y_peaks, 'Y')
            self.peak_data_z = (z_coords, z_profile, valid_z_peaks, 'Z')

            self.canvas_x.mpl_connect('pick_event', self._on_pick_x)
            self.canvas_y.mpl_connect('pick_event', self._on_pick_y)
            self.canvas_z.mpl_connect('pick_event', self._on_pick_z)
            
            # Conectar eventos de zoom
            self.figure_x.canvas.mpl_connect('scroll_event', lambda event: self._zoom_factory(event, ax_x))
            self.figure_y.canvas.mpl_connect('scroll_event', lambda event: self._zoom_factory(event, ax_y))
            self.figure_z.canvas.mpl_connect('scroll_event', lambda event: self._zoom_factory(event, ax_z))
            
            # Redibuja los canvases
            self.canvas_x.draw()
            self.canvas_y.draw()
            self.canvas_z.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error finding peaks: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging

    def _on_pick_x(self, event):
        """Handler específico para picos en el eje X"""
        coords, profile, valid_peaks, axis_name = self.peak_data_x
        self._find_closest_peak(event, coords, profile, valid_peaks, axis_name)

    def _on_pick_y(self, event):
        """Handler específico para picos en el eje Y"""
        coords, profile, valid_peaks, axis_name = self.peak_data_y
        self._find_closest_peak(event, coords, profile, valid_peaks, axis_name)

    def _on_pick_z(self, event):
        """Handler específico para picos en el eje Z"""
        coords, profile, valid_peaks, axis_name = self.peak_data_z
        self._find_closest_peak(event, coords, profile, valid_peaks, axis_name)

    def _find_closest_peak(self, event, coords, profile, valid_peaks, axis_name):
        """Encuentra el pico más cercano al punto donde se hizo clic"""
        try:
            # Obtener la posición del clic en coordenadas de datos
            mouse_x, mouse_y = event.mouseevent.xdata, event.mouseevent.ydata
            if mouse_x is None or mouse_y is None:
                return
            
            # Si no hay picos válidos, salir
            if not valid_peaks:
                return
            
            # Encontrar el pico más cercano al punto donde se hizo clic
            closest_peak = None
            min_distance = float('inf')
            
            for peak_idx in valid_peaks:
                peak_x = coords[peak_idx]
                peak_y = profile[peak_idx]
                
                # Calcular distancia euclidiana entre el clic y el pico
                distance = ((peak_x - mouse_x) ** 2 + (peak_y - mouse_y) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_peak = peak_idx
            
            if closest_peak is not None:
                # Obtener información del pico más cercano
                coord_value = coords[closest_peak]
                amplitude_value = profile[closest_peak]
                
                # Obtener las coordenadas físicas para este punto
                physical_info = ""
                
                if axis_name == 'X' and self.x is not None:
                    physical_info = f"Physical X: {coord_value:.6f}"
                elif axis_name == 'Y' and self.y is not None:
                    physical_info = f"Physical Y: {coord_value:.6f}"
                elif axis_name == 'Z' and self.z is not None:
                    physical_info = f"Physical Z: {coord_value:.6f}"
                else:
                    physical_info = f"Array index: {closest_peak}"
                
                # Mostrar la información en un diálogo
                QMessageBox.information(
                    self, 
                    f"{axis_name} Peak Information",
                    f"{physical_info}\nAmplitude: {amplitude_value:.6f}"
                )
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error displaying peak information: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full error for debugging

    def _zoom_factory(self, event, ax):
        """Función para manejar el zoom con la rueda del ratón"""
        base_scale = 1.2  # Factor de escala
        
        # Obtener la posición actual del cursor
        x_data, y_data = event.xdata, event.ydata
        
        # Si el cursor está fuera de los ejes, no hacer nada
        if x_data is None or y_data is None:
            return
        
        # Obtener los límites actuales
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # Calcular la posición relativa del cursor
        x_rel = (x_data - x_min) / (x_max - x_min)
        y_rel = (y_data - y_min) / (y_max - y_min)
        
        # Escalar según la dirección del desplazamiento de la rueda
        if event.button == 'up':  # Zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':  # Zoom out
            scale_factor = base_scale
        else:
            scale_factor = 1.0
        
        # Calcular nuevos límites
        new_width = (x_max - x_min) * scale_factor
        new_height = (y_max - y_min) * scale_factor
        
        # Establecer nuevos límites manteniendo el cursor en la misma posición relativa
        ax.set_xlim([x_data - x_rel * new_width, x_data + (1 - x_rel) * new_width])
        ax.set_ylim([y_data - y_rel * new_height, y_data + (1 - y_rel) * new_height])
        
        # Redibujar el canvas
        ax.figure.canvas.draw_idle()
    

if __name__ == "__main__":
    app = QApplication([])
    window = MyWidget()
    window.show()
    sys.exit(app.exec_())
