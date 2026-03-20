# CMake generated Testfile for 
# Source directory: /home/ac.zzheng/benchmark/spec/miniWeather/cpp
# Build directory: /home/ac.zzheng/benchmark/spec/miniWeather/cpp/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(SERIAL_TEST "./check_output.sh" "./serial_test" "1e-9" "4.5e-5")
set_tests_properties(SERIAL_TEST PROPERTIES  _BACKTRACE_TRIPLES "/home/ac.zzheng/benchmark/spec/miniWeather/cpp/CMakeLists.txt;71;add_test;/home/ac.zzheng/benchmark/spec/miniWeather/cpp/CMakeLists.txt;0;")
add_test(MPI_TEST "./check_output.sh" "./mpi_test" "1e-9" "4.5e-5")
set_tests_properties(MPI_TEST PROPERTIES  _BACKTRACE_TRIPLES "/home/ac.zzheng/benchmark/spec/miniWeather/cpp/CMakeLists.txt;86;add_test;/home/ac.zzheng/benchmark/spec/miniWeather/cpp/CMakeLists.txt;0;")
add_test(YAKL_TEST "./check_output.sh" "./parallelfor_test" "1e-9" "4.5e-5")
set_tests_properties(YAKL_TEST PROPERTIES  _BACKTRACE_TRIPLES "/home/ac.zzheng/benchmark/spec/miniWeather/cpp/CMakeLists.txt;101;add_test;/home/ac.zzheng/benchmark/spec/miniWeather/cpp/CMakeLists.txt;0;")
add_test(YAKL_SIMD_X_TEST "./check_output.sh" "./parallelfor_simd_x_test" "1e-9" "4.5e-5")
set_tests_properties(YAKL_SIMD_X_TEST PROPERTIES  _BACKTRACE_TRIPLES "/home/ac.zzheng/benchmark/spec/miniWeather/cpp/CMakeLists.txt;116;add_test;/home/ac.zzheng/benchmark/spec/miniWeather/cpp/CMakeLists.txt;0;")
subdirs("YAKL")
