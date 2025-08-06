cmake_minimum_required(VERSION 3.14)
project(autoware_imu_corrector)

find_package(autoware_cmake REQUIRED)
autoware_package()

# オリジナル版（既存）
ament_auto_add_library(imu_corrector SHARED
  src/imu_corrector_core.cpp
)

ament_auto_add_library(gyro_bias_estimator SHARED
  src/gyro_bias_estimator.cpp
  src/gyro_bias_estimation_module.cpp
)

# Sergio版（高機能版）
ament_auto_add_library(imu_corrector_sergio SHARED
  src/imu_corrector_core_sergio.cpp
)

ament_auto_add_library(gyro_bias_estimator_sergio SHARED
  src/gyro_bias_estimator_sergio.cpp
  src/gyro_bias_estimation_module.cpp
)

# Sakata版（現在版・シンプル）
ament_auto_add_library(imu_corrector_sakata SHARED
  src/imu_corrector_core_sakata.cpp
)

ament_auto_add_library(gyro_bias_estimator_sakata SHARED
  src/gyro_bias_estimator_sakata.cpp
  src/gyro_bias_estimation_module.cpp
)

# Node registrations - オリジナル版
rclcpp_components_register_node(imu_corrector
  PLUGIN "autoware::imu_corrector::ImuCorrector"
  EXECUTABLE imu_corrector_node
  EXECUTOR SingleThreadedExecutor
)

rclcpp_components_register_node(gyro_bias_estimator
  PLUGIN "autoware::imu_corrector::GyroBiasEstimator"
  EXECUTABLE gyro_bias_estimator_node
  EXECUTOR SingleThreadedExecutor
)

# Node registrations - Sergio版
rclcpp_components_register_node(imu_corrector_sergio
  PLUGIN "autoware::imu_corrector::ImuCorrectorSergio"
  EXECUTABLE imu_corrector_sergio_node
  EXECUTOR SingleThreadedExecutor
)

rclcpp_components_register_node(gyro_bias_estimator_sergio
  PLUGIN "autoware::imu_corrector::GyroBiasEstimatorSergio"
  EXECUTABLE gyro_bias_estimator_sergio_node
  EXECUTOR SingleThreadedExecutor
)

# Node registrations - Sakata版
rclcpp_components_register_node(imu_corrector_sakata
  PLUGIN "autoware::imu_corrector::ImuCorrectorSakata"
  EXECUTABLE imu_corrector_sakata_node
  EXECUTOR SingleThreadedExecutor
)

rclcpp_components_register_node(gyro_bias_estimator_sakata
  PLUGIN "autoware::imu_corrector::GyroBiasEstimatorSakata"
  EXECUTABLE gyro_bias_estimator_sakata_node
  EXECUTOR SingleThreadedExecutor
)

# Testing
function(add_testcase filepath)
  get_filename_component(filename ${filepath} NAME)
  string(REGEX REPLACE "\\.cpp$" "" test_name ${filename})
  ament_add_gmock(${test_name} ${filepath})
  target_link_libraries("${test_name}" gyro_bias_estimator)
  ament_target_dependencies(${test_name} ${${PROJECT_NAME}_FOUND_BUILD_DEPENDS})
endfunction()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  add_testcase(test/test_gyro_bias_estimation_module.cpp)
endif()

ament_auto_package(INSTALL_TO_SHARE
  launch
  config
)