Most of settings can be customized within "src/horus/settings.json". 
This file will be created after first run.

###### Turntable controller hello string and initialization
        "firmware_string": {
        "init_string": {

#### Visualization and GUI settings: 
###### For the circular platform
        "machine_shape" => "Circular"
        "machine_diameter": {
        "machine_height": {

###### For square platform (do someone use such???). Not supported in ROI 
        "machine_shape" => "Rectangular"
        "machine_depth": {
        "machine_width": {
        "machine_height": {

###### Turntable 3D model for "scanning" workbench visualization
        "machine_model_path"
        "machine_model_diameter"
        "machine_model_offset_x"
        "machine_model_offset_y"
        "machine_model_offset_z"

###### Turntable border Z offset for video overlay visualization. 
Useful for better visualization if your turntable have raised border
        "machine_shape_z": {

###### Inner points markers on turntable border.
Later will be used for turntable calibration by markers.
        "platform_markers_diameter": {
        "platform_markers_z": {


###### Maximum values for ROI:
        "roi_diameter": {
            "max_value": 350, 

        "roi_height": {
            "max_value": 450, 
