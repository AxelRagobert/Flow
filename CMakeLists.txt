project(flow)
cmake_minimum_required(VERSION 2.8)

set(PROJECT_SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")
set(GLAD_SOURCES "${CMAKE_SOURCE_DIR}/include/glad/src/glad.c")
set(GLAD_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/include/glad/include")
set(STB_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/include/stb/")
set(SHADER_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/include/learn_opengl/")
set(OPENCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include/OpenCL/")

find_package(PkgConfig REQUIRED)
find_package(OpenCL REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
set (OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL  REQUIRED)
find_package(glm REQUIRED)

include_directories(
  ${GLAD_INCLUDE_DIRS}
  ${STB_INCLUDE_DIRS}
  ${SHADER_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIR}
  )
  
file(GLOB SOURCES ${PROJECT_SOURCES} ${GLAD_SOURCES})

if(NOT MSVC)
  find_library(M_LIB m)
else() # not needed on Windows
  set(M_LIB "")
endif()

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(
  ${PROJECT_NAME}
  ${GLFW_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${M_LIB}
  ${CMAKE_DL_LIBS}
  OpenCL::OpenCL
  )