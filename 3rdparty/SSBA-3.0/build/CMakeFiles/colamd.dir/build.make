# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.1

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/build

# Include any dependencies generated for this target.
include CMakeFiles/colamd.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/colamd.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/colamd.dir/flags.make

CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o: CMakeFiles/colamd.dir/flags.make
CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o: ../COLAMD/Source/colamd.c
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o   -c /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/COLAMD/Source/colamd.c

CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/COLAMD/Source/colamd.c > CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.i

CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/COLAMD/Source/colamd.c -o CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.s

CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o.requires:
.PHONY : CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o.requires

CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o.provides: CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o.requires
	$(MAKE) -f CMakeFiles/colamd.dir/build.make CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o.provides.build
.PHONY : CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o.provides

CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o.provides.build: CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o

CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o: CMakeFiles/colamd.dir/flags.make
CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o: ../COLAMD/Source/colamd_global.c
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o   -c /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/COLAMD/Source/colamd_global.c

CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/COLAMD/Source/colamd_global.c > CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.i

CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/COLAMD/Source/colamd_global.c -o CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.s

CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o.requires:
.PHONY : CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o.requires

CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o.provides: CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o.requires
	$(MAKE) -f CMakeFiles/colamd.dir/build.make CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o.provides.build
.PHONY : CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o.provides

CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o.provides.build: CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o

CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o: CMakeFiles/colamd.dir/flags.make
CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o: ../SuiteSparse_config/SuiteSparse_config.c
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o   -c /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/SuiteSparse_config/SuiteSparse_config.c

CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/SuiteSparse_config/SuiteSparse_config.c > CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.i

CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/SuiteSparse_config/SuiteSparse_config.c -o CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.s

CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o.requires:
.PHONY : CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o.requires

CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o.provides: CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o.requires
	$(MAKE) -f CMakeFiles/colamd.dir/build.make CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o.provides.build
.PHONY : CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o.provides

CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o.provides.build: CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o

# Object files for target colamd
colamd_OBJECTS = \
"CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o" \
"CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o" \
"CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o"

# External object files for target colamd
colamd_EXTERNAL_OBJECTS =

libcolamd.a: CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o
libcolamd.a: CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o
libcolamd.a: CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o
libcolamd.a: CMakeFiles/colamd.dir/build.make
libcolamd.a: CMakeFiles/colamd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C static library libcolamd.a"
	$(CMAKE_COMMAND) -P CMakeFiles/colamd.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/colamd.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/colamd.dir/build: libcolamd.a
.PHONY : CMakeFiles/colamd.dir/build

CMakeFiles/colamd.dir/requires: CMakeFiles/colamd.dir/COLAMD/Source/colamd.c.o.requires
CMakeFiles/colamd.dir/requires: CMakeFiles/colamd.dir/COLAMD/Source/colamd_global.c.o.requires
CMakeFiles/colamd.dir/requires: CMakeFiles/colamd.dir/SuiteSparse_config/SuiteSparse_config.c.o.requires
.PHONY : CMakeFiles/colamd.dir/requires

CMakeFiles/colamd.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/colamd.dir/cmake_clean.cmake
.PHONY : CMakeFiles/colamd.dir/clean

CMakeFiles/colamd.dir/depend:
	cd /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0 /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0 /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/build /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/build /Users/changxiao/Documents/SfM-Toy-Library-master/3rdparty/SSBA-3.0/build/CMakeFiles/colamd.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/colamd.dir/depend

