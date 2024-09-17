cmake_minimum_required(VERSION 3.10)
project(ParkingSpaceDetection) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories for your project
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the source files
add_executable(output 
    main.cpp
    box_extract_function.cpp
    utilities.cpp
    mioumap.cpp
    xmlgroundparsing.cpp
    # carsegment.cpp
    draw_rotated_rectangles.cpp
    pugixml.cpp
    tinyxml2.cpp
)

# Link libraries
target_link_libraries(output ${OpenCV_LIBS})
