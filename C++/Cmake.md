cmake_minimum_required(VERSION 3.10)

project(EigenProject)
  

# 设置 C++ 标准

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

  

# 寻找系统安装的 Eigen3

find_package(Eigen3 REQUIRED)

  

# 添加可执行文件

add_executable(main src/test_eigen.cpp)

target_include_directories(main PRIVATE include)

  

# 链接/包含 Eigen3 头文件

target_link_libraries(main PRIVATE Eigen3::Eigen)