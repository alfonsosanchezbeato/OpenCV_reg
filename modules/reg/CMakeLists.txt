cmake_minimum_required(VERSION 2.8)
project(OpenCV_reg)
find_package(OpenCV REQUIRED)

set(SOURCES
    map_test.cpp
    src/map.cpp
    src/mapaffine.cpp
    src/mapper.cpp
    src/mappergradaffine.cpp
    src/mappergradeuclid.cpp
    src/mappergradproj.cpp
    src/mappergradshift.cpp
    src/mappergradsimilar.cpp
    src/mapperpyramid.cpp
    src/mapprojec.cpp
    src/mapshift.cpp
    src/precomp.cpp)

set(HEADERS
    src/precomp.hpp
    include/opencv2/reg/map.hpp
    include/opencv2/reg/mapaffine.hpp
    include/opencv2/reg/mapper.hpp
    include/opencv2/reg/mappergradaffine.hpp
    include/opencv2/reg/mappergradeuclid.hpp
    include/opencv2/reg/mappergradproj.hpp
    include/opencv2/reg/mappergradshift.hpp
    include/opencv2/reg/mappergradsimilar.hpp
    include/opencv2/reg/mapperpyramid.hpp
    include/opencv2/reg/mapprojec.hpp
    include/opencv2/reg/mapshift.hpp)

include_directories(${OpenCV_reg_SOURCE_DIR}/include)
add_executable(map_test ${SOURCES} ${HEADERS})
target_link_libraries(map_test ${OpenCV_LIBS})
