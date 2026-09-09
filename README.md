Medicine Chair — RUB

Author: Ana Quintero

-------------------------------------------------------------------------------------------------
Overview

This application was developed for the Chair of Medicine at the Ruhr-Universität Bochum. It reads experimental/scientific data, generates 3D plots, produces per-plane 2D scatter views with FWHM (Full Width at Half Maximum) calculations, and allows comparison between different data files.

The graphical interface was designed in Qt Creator, and the application logic is implemented in Python.

--------------------------------------------------------------------------------------------------
Project Structure

- Final_version.py	Main application script — contains all the application logic and connects it to the GUI.

- principalwindow.ui	Qt Creator UI file for the main application window.

- scatterwindow.ui	Qt Creator UI file for the 3D/per-plane scatter plot window.

- pop_up_scatter.ui	Qt Creator UI file for the scatter plot pop-up dialog.

- fwhm.ui	Qt Creator UI file for the FWHM calculation window.
- compaData.ui	Qt Creator UI file for the data comparison window.

All .ui files were created using Qt Creator and define the visual layout of the application windows. They are loaded/compiled into the application via Final_version.py.


The .spec file is used for building the standalone executable. Do not modify.
