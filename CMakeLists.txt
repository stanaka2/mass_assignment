cmake_minimum_required(VERSION 3.18)
project(mass_assign LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

find_package(OpenMP)
pybind11_add_module(_mass_assign_core MODULE mass_assign_core.cpp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(_mass_assign_core PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(_mass_assign_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-march=native>
  $<$<CXX_COMPILER_ID:MSVC>:/O2>
)


# ---- install ----
install(TARGETS _mass_assign_core LIBRARY DESTINATION mass_assignment)
install(DIRECTORY python/ DESTINATION mass_assignment FILES_MATCHING PATTERN "*.py")
