# Dentro de tu clase ScatterDialog:

# 1. Conecta directamente el cambio de índice del ComboBox al método real:
self.ui.combbsimu.currentIndexChanged.connect(self.update_profile_plots)

# 2. Define la función de actualización (debe aceptar el índice como argumento):
def update_profile_plots(self, point_index):
    # ... Aquí va todo tu código de graficado para el punto seleccionado ...
    # Usa self.zoom_window como siempre para el xlim/ylim/zlim
    pass

# 3. En tu función de actualización del zoom, simplemente llama a update_profile_plots con el índice actual:
def update_zoom_window(self):
    texto = self.ui.InputZoom.text().strip()
    try:
        valor = float(texto)
        if valor <= 0:
            raise ValueError
        self.zoom_window = valor
    except ValueError:
        QMessageBox.warning(self, "Valor inválido", "El valor debe ser un número positivo.")
        self.ui.InputZoom.setText(str(self.zoom_window))
        return

    # Llama a update_profile_plots para el punto actualmente seleccionado
    if self.ui.combbprincipal.currentText() == "Simulation peaks " and self.ui.combbsimu.isEnabled():
        idx = self.ui.combbsimu.currentIndex()
        self.update_profile_plots(idx)