cmake_minimum_required (VERSION 2.8)  
  
project (printf_caffemodel)  
  
add_executable(printf_caffemodel printf_caffemodel.cpp)  
  
include_directories ( /home/sad/caffe/include  
    /usr/local/include  
    /usr/local/cuda/include  
    /usr/include 
    /home/sad/caffe/build/src )  
  
target_link_libraries(printf_caffemodel  
    /home/sad/caffe/build/lib/libcaffe.so  
    /usr/lib/x86_64-linux-gnu/libglog.so  
    /usr/lib/x86_64-linux-gnu/libboost_system.so  
    )
