include_directories(include)  
保证系统可以找到hpp文件联系
 
add_executable(identify_node  
src/identify_node.cpp  
src/identify.cpp)  
节点上挂多个文件