# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ziyang/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ziyang/src/build

# Include any dependencies generated for this target.
include CMakeFiles/osm_objects.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/osm_objects.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/osm_objects.dir/flags.make

CMakeFiles/osm_objects.dir/bindings.cpp.o: CMakeFiles/osm_objects.dir/flags.make
CMakeFiles/osm_objects.dir/bindings.cpp.o: ../bindings.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ziyang/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/osm_objects.dir/bindings.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/osm_objects.dir/bindings.cpp.o -c /home/ziyang/src/bindings.cpp

CMakeFiles/osm_objects.dir/bindings.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/osm_objects.dir/bindings.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ziyang/src/bindings.cpp > CMakeFiles/osm_objects.dir/bindings.cpp.i

CMakeFiles/osm_objects.dir/bindings.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/osm_objects.dir/bindings.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ziyang/src/bindings.cpp -o CMakeFiles/osm_objects.dir/bindings.cpp.s

CMakeFiles/osm_objects.dir/bindings.cpp.o.requires:

.PHONY : CMakeFiles/osm_objects.dir/bindings.cpp.o.requires

CMakeFiles/osm_objects.dir/bindings.cpp.o.provides: CMakeFiles/osm_objects.dir/bindings.cpp.o.requires
	$(MAKE) -f CMakeFiles/osm_objects.dir/build.make CMakeFiles/osm_objects.dir/bindings.cpp.o.provides.build
.PHONY : CMakeFiles/osm_objects.dir/bindings.cpp.o.provides

CMakeFiles/osm_objects.dir/bindings.cpp.o.provides.build: CMakeFiles/osm_objects.dir/bindings.cpp.o


CMakeFiles/osm_objects.dir/objects.cpp.o: CMakeFiles/osm_objects.dir/flags.make
CMakeFiles/osm_objects.dir/objects.cpp.o: ../objects.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ziyang/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/osm_objects.dir/objects.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/osm_objects.dir/objects.cpp.o -c /home/ziyang/src/objects.cpp

CMakeFiles/osm_objects.dir/objects.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/osm_objects.dir/objects.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ziyang/src/objects.cpp > CMakeFiles/osm_objects.dir/objects.cpp.i

CMakeFiles/osm_objects.dir/objects.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/osm_objects.dir/objects.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ziyang/src/objects.cpp -o CMakeFiles/osm_objects.dir/objects.cpp.s

CMakeFiles/osm_objects.dir/objects.cpp.o.requires:

.PHONY : CMakeFiles/osm_objects.dir/objects.cpp.o.requires

CMakeFiles/osm_objects.dir/objects.cpp.o.provides: CMakeFiles/osm_objects.dir/objects.cpp.o.requires
	$(MAKE) -f CMakeFiles/osm_objects.dir/build.make CMakeFiles/osm_objects.dir/objects.cpp.o.provides.build
.PHONY : CMakeFiles/osm_objects.dir/objects.cpp.o.provides

CMakeFiles/osm_objects.dir/objects.cpp.o.provides.build: CMakeFiles/osm_objects.dir/objects.cpp.o


CMakeFiles/osm_objects.dir/coordinates.cpp.o: CMakeFiles/osm_objects.dir/flags.make
CMakeFiles/osm_objects.dir/coordinates.cpp.o: ../coordinates.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ziyang/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/osm_objects.dir/coordinates.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/osm_objects.dir/coordinates.cpp.o -c /home/ziyang/src/coordinates.cpp

CMakeFiles/osm_objects.dir/coordinates.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/osm_objects.dir/coordinates.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ziyang/src/coordinates.cpp > CMakeFiles/osm_objects.dir/coordinates.cpp.i

CMakeFiles/osm_objects.dir/coordinates.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/osm_objects.dir/coordinates.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ziyang/src/coordinates.cpp -o CMakeFiles/osm_objects.dir/coordinates.cpp.s

CMakeFiles/osm_objects.dir/coordinates.cpp.o.requires:

.PHONY : CMakeFiles/osm_objects.dir/coordinates.cpp.o.requires

CMakeFiles/osm_objects.dir/coordinates.cpp.o.provides: CMakeFiles/osm_objects.dir/coordinates.cpp.o.requires
	$(MAKE) -f CMakeFiles/osm_objects.dir/build.make CMakeFiles/osm_objects.dir/coordinates.cpp.o.provides.build
.PHONY : CMakeFiles/osm_objects.dir/coordinates.cpp.o.provides

CMakeFiles/osm_objects.dir/coordinates.cpp.o.provides.build: CMakeFiles/osm_objects.dir/coordinates.cpp.o


CMakeFiles/osm_objects.dir/utils.cpp.o: CMakeFiles/osm_objects.dir/flags.make
CMakeFiles/osm_objects.dir/utils.cpp.o: ../utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ziyang/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/osm_objects.dir/utils.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/osm_objects.dir/utils.cpp.o -c /home/ziyang/src/utils.cpp

CMakeFiles/osm_objects.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/osm_objects.dir/utils.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ziyang/src/utils.cpp > CMakeFiles/osm_objects.dir/utils.cpp.i

CMakeFiles/osm_objects.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/osm_objects.dir/utils.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ziyang/src/utils.cpp -o CMakeFiles/osm_objects.dir/utils.cpp.s

CMakeFiles/osm_objects.dir/utils.cpp.o.requires:

.PHONY : CMakeFiles/osm_objects.dir/utils.cpp.o.requires

CMakeFiles/osm_objects.dir/utils.cpp.o.provides: CMakeFiles/osm_objects.dir/utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/osm_objects.dir/build.make CMakeFiles/osm_objects.dir/utils.cpp.o.provides.build
.PHONY : CMakeFiles/osm_objects.dir/utils.cpp.o.provides

CMakeFiles/osm_objects.dir/utils.cpp.o.provides.build: CMakeFiles/osm_objects.dir/utils.cpp.o


# Object files for target osm_objects
osm_objects_OBJECTS = \
"CMakeFiles/osm_objects.dir/bindings.cpp.o" \
"CMakeFiles/osm_objects.dir/objects.cpp.o" \
"CMakeFiles/osm_objects.dir/coordinates.cpp.o" \
"CMakeFiles/osm_objects.dir/utils.cpp.o"

# External object files for target osm_objects
osm_objects_EXTERNAL_OBJECTS =

osm_objects.so: CMakeFiles/osm_objects.dir/bindings.cpp.o
osm_objects.so: CMakeFiles/osm_objects.dir/objects.cpp.o
osm_objects.so: CMakeFiles/osm_objects.dir/coordinates.cpp.o
osm_objects.so: CMakeFiles/osm_objects.dir/utils.cpp.o
osm_objects.so: CMakeFiles/osm_objects.dir/build.make
osm_objects.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
osm_objects.so: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
osm_objects.so: /usr/lib/x86_64-linux-gnu/libboost_python.so
osm_objects.so: /usr/lib/x86_64-linux-gnu/libpython2.7.so
osm_objects.so: CMakeFiles/osm_objects.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ziyang/src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX shared library osm_objects.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/osm_objects.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/osm_objects.dir/build: osm_objects.so

.PHONY : CMakeFiles/osm_objects.dir/build

CMakeFiles/osm_objects.dir/requires: CMakeFiles/osm_objects.dir/bindings.cpp.o.requires
CMakeFiles/osm_objects.dir/requires: CMakeFiles/osm_objects.dir/objects.cpp.o.requires
CMakeFiles/osm_objects.dir/requires: CMakeFiles/osm_objects.dir/coordinates.cpp.o.requires
CMakeFiles/osm_objects.dir/requires: CMakeFiles/osm_objects.dir/utils.cpp.o.requires

.PHONY : CMakeFiles/osm_objects.dir/requires

CMakeFiles/osm_objects.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/osm_objects.dir/cmake_clean.cmake
.PHONY : CMakeFiles/osm_objects.dir/clean

CMakeFiles/osm_objects.dir/depend:
	cd /home/ziyang/src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ziyang/src /home/ziyang/src /home/ziyang/src/build /home/ziyang/src/build /home/ziyang/src/build/CMakeFiles/osm_objects.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/osm_objects.dir/depend

