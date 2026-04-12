#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "yakl::yakl_fortran_interface" for configuration "Release"
set_property(TARGET yakl::yakl_fortran_interface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(yakl::yakl_fortran_interface PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA;Fortran"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libyakl_fortran_interface.a"
  )

list(APPEND _cmake_import_check_targets yakl::yakl_fortran_interface )
list(APPEND _cmake_import_check_files_for_yakl::yakl_fortran_interface "${_IMPORT_PREFIX}/lib/libyakl_fortran_interface.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
