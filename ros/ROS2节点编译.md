add_executable(==camera_node== src/==camera_node==.cpp)  
ament_target_dependencies(==camera_node rclcpp opencv==)  
install(TARGETS ==camera_node==  
DESTINATION lib/${PROJECT_NAME}  
)
   

创建包  
# 1. 进入你的工作空间src目录  
cd ~/```_ws/src
 
# 2. 创建包（最常用的命令）  
ros2 pkg create pkg_name --build-type ament_cmake --dependencies rclcpp opencv4
   

Build  
colcon build --packages-select pkg_name  
--cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON  
环境  
source install/setup.bash  
另一个激活：  
source /opt/ros/humble/setup.bash
   

rm -rf build install log空间里删除包
 
json步骤  
1.cpp测试代码  
#include \<rclcpp/rclcpp.hpp\>  
int main(int argc,char**argv)  
{  
rclcpp::init(argc,argv);  
auto node = std::make_shared\<rclcpp::Node\>("test_node");  
RCLCPP_INFO(node-\>get_logger(),"Test Start");  
rclcpp::spin(node);  
rclcpp::shutdown();  
return 0;  
}  
3.build一下  
find build -name compile_commands.json  
终端在ws下查找  
路径复制到高级下的编译命令中
 - ==package.xml：====\<depend\>包名\</depend\>==
- ==CMakeLists.txt：====find_package(包名 REQUIRED)== ==+== ==ament_target_dependencies(包名)==
- ==OpenCV特殊：CMake用====OpenCV====，package.xml用====opencv==
- ==OpenCV== ==通过== ==find_package(OpenCV REQUIRED)== ==在== ==CMake== ==中处理，不需要在== ==package.xml== ==中声明为== ==ROS== ==依赖。==    
==编译失败后要==rm -rf build/ install/ log/保证清理，在ws下执行  
启动  
Ros2 run pkg_name node_name
   

opencv里  
xml：  
\<build_depend\>libopencv-dev\</build_depend\>