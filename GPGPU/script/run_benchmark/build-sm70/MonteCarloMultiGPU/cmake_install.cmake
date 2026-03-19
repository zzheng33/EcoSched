# Install script for directory: /home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70-source/Samples/5_Domain_Specific/MonteCarloMultiGPU

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70/MonteCarloMultiGPU/bin")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  
        # Determine the actual install directory based on the configuration
        # CMAKE_INSTALL_CONFIG_NAME is available at install time for multi-config generators
        if(DEFINED CMAKE_INSTALL_CONFIG_NAME)
            string(TOLOWER "${CMAKE_INSTALL_CONFIG_NAME}" INSTALL_BUILD_TYPE)
        else()
            set(INSTALL_BUILD_TYPE "release")
        endif()
        set(INSTALL_DIR "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70/MonteCarloMultiGPU/bin/x86_64/linux/${INSTALL_BUILD_TYPE}")
        
        # Search in the current project's binary directory for built executables
        file(GLOB_RECURSE BINARY_FILES 
             LIST_DIRECTORIES false
             "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70/MonteCarloMultiGPU/*")
        
        # Copy data files from sample's own data directory
        file(GLOB_RECURSE SAMPLE_DATA_FILES
             LIST_DIRECTORIES false
             "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70-source/Samples/5_Domain_Specific/MonteCarloMultiGPU/data/*")
        
        # Copy shared data files from Common/data directory
        # Try both paths: ../../../Common (for regular samples) and ../../../../Common (for Tegra)
        set(COMMON_DATA_FILES "")
        if(EXISTS "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70-source/Samples/5_Domain_Specific/MonteCarloMultiGPU/../../../Common/data")
            file(GLOB_RECURSE COMMON_DATA_FILES
                 LIST_DIRECTORIES false
                 "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70-source/Samples/5_Domain_Specific/MonteCarloMultiGPU/../../../Common/data/*")
        elseif(EXISTS "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70-source/Samples/5_Domain_Specific/MonteCarloMultiGPU/../../../../Common/data")
            file(GLOB_RECURSE COMMON_DATA_FILES
                 LIST_DIRECTORIES false
                 "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70-source/Samples/5_Domain_Specific/MonteCarloMultiGPU/../../../../Common/data/*")
        endif()
        
        # Copy shared library files from bin/win64 directory (Windows only)
        # These are pre-built DLLs like freeglut.dll, glew64.dll, etc.
        set(SHARED_LIB_FILES "")
        if(CMAKE_HOST_WIN32)
            # Determine build configuration at install time
            # CMAKE_INSTALL_CONFIG_NAME is set by CMake at install time for multi-config generators
            if(DEFINED CMAKE_INSTALL_CONFIG_NAME)
                string(TOLOWER "${CMAKE_INSTALL_CONFIG_NAME}" INSTALL_CONFIG_LOWER)
            else()
                # Fallback for single-config generators
                set(INSTALL_CONFIG_LOWER "release")
            endif()
            
            # Try multiple possible paths for bin/win64 directory
            set(BIN_WIN64_PATHS
                "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70-source/Samples/5_Domain_Specific/MonteCarloMultiGPU/../../../bin/win64/${INSTALL_CONFIG_LOWER}"
                "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70-source/Samples/5_Domain_Specific/MonteCarloMultiGPU/../../../../bin/win64/${INSTALL_CONFIG_LOWER}"
                "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70-source/Samples/5_Domain_Specific/MonteCarloMultiGPU/bin/win64/${INSTALL_CONFIG_LOWER}"
            )
            foreach(BIN_PATH IN LISTS BIN_WIN64_PATHS)
                if(EXISTS "${BIN_PATH}")
                    file(GLOB SHARED_LIB_FILES
                         LIST_DIRECTORIES false
                         "${BIN_PATH}/*.dll")
                    if(SHARED_LIB_FILES)
                        break()
                    endif()
                endif()
            endforeach()
        endif()
        
        # Combine all lists
        set(SAMPLE_FILES ${BINARY_FILES} ${SAMPLE_DATA_FILES} ${COMMON_DATA_FILES} ${SHARED_LIB_FILES})
        
        # Remove duplicates to avoid copying the same file multiple times
        # This preserves the order, so files from earlier sources take precedence
        list(REMOVE_DUPLICATES SAMPLE_FILES)
        
        set(INSTALLED_COUNT 0)
        
        # Filter to include executable files and specific file types
        foreach(SAMPLE_FILE IN LISTS SAMPLE_FILES)
            # Skip non-files
            if(NOT IS_DIRECTORY "${SAMPLE_FILE}")
                get_filename_component(SAMPLE_EXT "${SAMPLE_FILE}" EXT)
                get_filename_component(SAMPLE_NAME "${SAMPLE_FILE}" NAME)
                
                set(SHOULD_INSTALL FALSE)
                
                # Skip build artifacts, source files, and CMake files
                # Note: .lib (Windows import libs) and .a (static libs) are excluded - link-time only
                # .so (Linux shared libs) and .dll (Windows DLLs) are included - runtime dependencies
                # Source files (.cu, .cpp, .c, .h, etc.) are excluded - not needed at runtime
                if(NOT SAMPLE_EXT MATCHES "\\.(o|a|cmake|obj|lib|exp|ilk|pdb|cu|cpp|cxx|cc|c|h|hpp|hxx|cuh|inl)$" AND
                   NOT SAMPLE_NAME MATCHES "^(Makefile|cmake_install\\.cmake)$" AND
                   NOT "${SAMPLE_FILE}" MATCHES "/CMakeFiles/" AND
                   NOT "${SAMPLE_FILE}" MATCHES "\\\\CMakeFiles\\\\")
                    
                    # Check if file has required extension (fatbin, ptx, bc, raw, ppm) or is executable
                    if(SAMPLE_EXT MATCHES "\\.(fatbin|ptx|bc|raw|ppm)$")
                        set(SHOULD_INSTALL TRUE)
                    # Check for shared libraries: .dll (Windows) or .so (Linux)
                    elseif(SAMPLE_EXT MATCHES "\\.(dll|so)$")
                        set(SHOULD_INSTALL TRUE)
                    # On Windows, check for .exe extension
                    elseif(CMAKE_HOST_WIN32 AND SAMPLE_EXT MATCHES "\\.(exe)$")
                        set(SHOULD_INSTALL TRUE)
                    else()
                        # On Unix-like systems, check if file has executable permissions
                        if(NOT CMAKE_HOST_WIN32)
                            if(IS_SYMLINK "${SAMPLE_FILE}" OR 
                               (EXISTS "${SAMPLE_FILE}" AND NOT IS_DIRECTORY "${SAMPLE_FILE}"))
                                # Use test -x to check if file has executable permissions
                                execute_process(
                                    COMMAND test -x "${SAMPLE_FILE}"
                                    RESULT_VARIABLE IS_EXEC
                                    OUTPUT_QUIET ERROR_QUIET
                                )
                                if(IS_EXEC EQUAL 0)
                                    set(SHOULD_INSTALL TRUE)
                                endif()
                            endif()
                        endif()
                    endif()
                endif()
                
                if(SHOULD_INSTALL)
                    get_filename_component(FILE_NAME "${SAMPLE_FILE}" NAME)
                    set(DEST_FILE "${INSTALL_DIR}/${FILE_NAME}")
                    
                    # Determine file type based on extension
                    get_filename_component(FILE_EXT "${SAMPLE_FILE}" EXT)
                    set(IS_EXECUTABLE FALSE)
                    set(IS_SHARED_LIB FALSE)
                    set(IS_DATA_FILE FALSE)
                    
                    # Check for known data file extensions first
                    if(FILE_EXT MATCHES "\\.(fatbin|ptx|bc|raw|ppm)$")
                        set(IS_DATA_FILE TRUE)
                    # Check if it's a shared library
                    elseif(FILE_EXT MATCHES "\\.(dll|so)$")
                        set(IS_SHARED_LIB TRUE)
                    # Check if it's an executable
                    else()
                        # On Windows, check for .exe extension (not .dll - those are libraries)
                        if(CMAKE_HOST_WIN32)
                            if(FILE_EXT MATCHES "\\.(exe)$")
                                set(IS_EXECUTABLE TRUE)
                            endif()
                        else()
                            # On Unix-like systems, check for no extension (typical for executables)
                            # .so files are shared libraries, not executables
                            if(FILE_EXT STREQUAL "")
                                set(IS_EXECUTABLE TRUE)
                            endif()
                        endif()
                    endif()
                    
                    get_filename_component(DEST_DIR "${DEST_FILE}" DIRECTORY)
                    
                    # Check if this is a Common data file that already exists
                    # Skip copying to avoid redundant operations when multiple samples use the same files
                    set(SKIP_COPY FALSE)
                    if("${SAMPLE_FILE}" MATCHES "/Common/data/" AND EXISTS "${DEST_FILE}")
                        set(SKIP_COPY TRUE)
                    endif()
                    
                    if(NOT SKIP_COPY)
                    if(IS_DATA_FILE)
                        # Data file (.raw, .ppm, .ptx, .fatbin, .bc) - copy without execute permissions
                        message(STATUS "Installing data file: ${DEST_FILE}")
                        if(CMAKE_HOST_WIN32)
                            file(COPY "${SAMPLE_FILE}" DESTINATION "${DEST_DIR}")
                        else()
                            file(COPY "${SAMPLE_FILE}"
                                 DESTINATION "${DEST_DIR}"
                                 FILE_PERMISSIONS OWNER_READ OWNER_WRITE
                                                  GROUP_READ
                                                  WORLD_READ)
                        endif()
                        math(EXPR INSTALLED_COUNT "${INSTALLED_COUNT} + 1")
                    elseif(IS_EXECUTABLE)
                        # File is executable - copy with execute permissions (Unix) or as-is (Windows)
                        message(STATUS "Installing executable: ${DEST_FILE}")
                        if(CMAKE_HOST_WIN32)
                            file(COPY "${SAMPLE_FILE}" DESTINATION "${DEST_DIR}")
                        else()
                            file(COPY "${SAMPLE_FILE}"
                                 DESTINATION "${DEST_DIR}"
                                 FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
                                                  GROUP_READ GROUP_EXECUTE 
                                                  WORLD_READ WORLD_EXECUTE)
                        endif()
                        math(EXPR INSTALLED_COUNT "${INSTALLED_COUNT} + 1")
                    elseif(IS_SHARED_LIB)
                        # Shared library - copy with appropriate permissions
                        message(STATUS "Installing shared library: ${DEST_FILE}")
                        if(CMAKE_HOST_WIN32)
                            file(COPY "${SAMPLE_FILE}" DESTINATION "${DEST_DIR}")
                        else()
                            file(COPY "${SAMPLE_FILE}"
                                 DESTINATION "${DEST_DIR}"
                                 FILE_PERMISSIONS OWNER_READ OWNER_WRITE
                                                  GROUP_READ
                                                  WORLD_READ)
                        endif()
                        math(EXPR INSTALLED_COUNT "${INSTALLED_COUNT} + 1")
                    else()
                        # Unknown file type - copy as regular file without execute permissions
                        message(STATUS "Installing file: ${DEST_FILE}")
                        if(CMAKE_HOST_WIN32)
                            file(COPY "${SAMPLE_FILE}" DESTINATION "${DEST_DIR}")
                        else()
                            file(COPY "${SAMPLE_FILE}"
                                 DESTINATION "${DEST_DIR}"
                                 FILE_PERMISSIONS OWNER_READ OWNER_WRITE
                                                  GROUP_READ
                                                  WORLD_READ)
                        endif()
                        math(EXPR INSTALLED_COUNT "${INSTALLED_COUNT} + 1")
                    endif()
                    endif() # NOT SKIP_COPY
                endif()
            endif()
        endforeach()
        
        message(STATUS "Installation complete: ${INSTALLED_COUNT} files installed to ${INSTALL_DIR}")
    
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/ac.zzheng/power/GPGPU/script/run_benchmark/build-sm70/MonteCarloMultiGPU/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
