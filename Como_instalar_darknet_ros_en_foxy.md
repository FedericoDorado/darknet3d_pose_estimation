# En un terminal se hace source de ROS 2 Foxy

source /opt/ros/foxy/setup.bash

# En el espacio de trabajo se clona el repositorio mencionado:

cd ros2_ws/src

git clone --recursive https://github.com/Ar-Ray-code/darknet_ros_yolov4.git

darknet_ros_yolov4/darknet_ros/rm_darknet_CMakeLists.sh

cd ~/ros2_ws

# En el archivo CMakeLists.txt de darknet ros editar:

gedit $HOME/ros2_ws/src/darknet_ros_yolov4/darknet_ros/darknet_ros/CMakeLists.txt

# En la línea 10:

set(CUDA_ENABLE OFF)

# Guardar y Salir
cd ~/ros2_ws

# Compilar
colcon build --symlink-install

