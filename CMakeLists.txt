cmake_minimum_required(VERSION 3.16.3)
project(gravity_compensation)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(lbr_fri_msgs REQUIRED)

install(
  PROGRAMS
    src/EquationDerivation.py
    src/EquationDerivationRNEA.py
    src/ParamsEstimation.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY src/test
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
