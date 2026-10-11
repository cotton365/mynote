cmake的依赖，以camera_process_node为例子  
add_executable(camera_process_node src/camera_process_node.cpp)  
ament_target_dependencies(camera_process_node  
rclcpp  
std_msgs  
cv_bridge  
OpenCV  
sensor_msgs)  
target_link_libraries(camera_process_node ${OpenCV_LIBS})  
install(TARGETS camera_process_node  
DESTINATION lib/${PROJECT_NAME}  
)  
前面的endif之后要来一个set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")明确位置  
opencv和cv_bridge的比较特殊  
find_package(cv_bridge REQUIRED)  
find_package(OpenCV REQUIRED COMPONENTS  
core  
imgproc  
imgcodecs  
)  
find要放在set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")之后
      

xml里注意opencv  
\<build_depend\>libopencv-dev\</build_depend\>