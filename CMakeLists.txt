cmake_minimum_required(VERSION 3.18)
project(mass_assign LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

find_package(OpenMP)
pybind11_add_module(_binding MODULE binding.cpp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(_binding PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(_binding PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-march=native>
  $<$<CXX_COMPILER_ID:MSVC>:/O2>
)


# ---- install ----
install(TARGETS _binding LIBRARY DESTINATION mass_assignment)
install(DIRECTORY python/ DESTINATION mass_assignment FILES_MATCHING PATTERN "*.py")
