// Copyright 2023 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gyro_bias_estimator.hpp"

#include <autoware/universe_utils/geometry/geometry.hpp>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace autoware::imu_corrector
{
GyroBiasEstimator::GyroBiasEstimator(const rclcpp::NodeOptions & options)
: rclcpp::Node("gyro_bias_validator", options),
  gyro_bias_threshold_(declare_parameter<double>("gyro_bias_threshold")),
  angular_velocity_offset_x_(declare_parameter<double>("angular_velocity_offset_x")),
  angular_velocity_offset_y_(declare_parameter<double>("angular_velocity_offset_y")),
  angular_velocity_offset_z_(declare_parameter<double>("angular_velocity_offset_z")),
  timer_callback_interval_sec_(declare_parameter<double>("timer_callback_interval_sec")),
  diagnostics_updater_interval_sec_(declare_parameter<double>("diagnostics_updater_interval_sec")),
  // straight_motion_ang_vel_upper_limit_(
  //   declare_parameter<double>("straight_motion_ang_vel_upper_limit")),
  // 新しく追加
  vehicle_stop_velocity_threshold_(declare_parameter<double>("vehicle_stop_velocity_threshold")),
  vehicle_stop_angular_velocity_threshold_(
    declare_parameter<double>("vehicle_stop_angular_velocity_threshold")),
  updater_(this),
  gyro_bias_(std::nullopt),
  is_vehicle_stopped_previous_(false),  // 新規追加：初期化
  stopped_sample_count_(0)              // 新規追加：初期化
{
  updater_.setHardwareID(get_name());
  updater_.add("gyro_bias_validator", this, &GyroBiasEstimator::update_diagnostics);
  updater_.setPeriod(diagnostics_updater_interval_sec_);

  gyro_bias_estimation_module_ = std::make_unique<GyroBiasEstimationModule>();

  imu_sub_ = create_subscription<Imu>(
    "~/input/imu_raw", rclcpp::SensorDataQoS(),
    [this](const Imu::ConstSharedPtr msg) { callback_imu(msg); });
  odom_sub_ = create_subscription<Odometry>(
    "~/input/odom", rclcpp::SensorDataQoS(),
    [this](const Odometry::ConstSharedPtr msg) { callback_odom(msg); });
  gyro_bias_pub_ = create_publisher<Vector3Stamped>("~/output/gyro_bias", rclcpp::SensorDataQoS());
  twist_with_covariance_sub_ = create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
    "/sensing/vehicle_velocity_converter/twist_with_covariance", rclcpp::QoS{10},
    std::bind(&GyroBiasEstimator::callback_twist_with_covariance, this, std::placeholders::_1));

  auto bound_timer_callback = std::bind(&GyroBiasEstimator::timer_callback, this);
  auto period_control = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(timer_callback_interval_sec_));
  timer_ = std::make_shared<rclcpp::GenericTimer<decltype(bound_timer_callback)>>(
    this->get_clock(), period_control, std::move(bound_timer_callback),
    this->get_node_base_interface()->get_context());
  this->get_node_timers_interface()->add_timer(timer_, nullptr);

  transform_listener_ = std::make_shared<autoware::universe_utils::TransformListener>(this);

  // initialize diagnostics_info_
  {
    diagnostics_info_.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    diagnostics_info_.summary_message = "Not initialized";
    diagnostics_info_.gyro_bias_x_for_imu_corrector = std::nan("");
    diagnostics_info_.gyro_bias_y_for_imu_corrector = std::nan("");
    diagnostics_info_.gyro_bias_z_for_imu_corrector = std::nan("");
    diagnostics_info_.estimated_gyro_bias_x = std::nan("");
    diagnostics_info_.estimated_gyro_bias_y = std::nan("");
    diagnostics_info_.estimated_gyro_bias_z = std::nan("");
  }
}

void GyroBiasEstimator::callback_imu(const Imu::ConstSharedPtr imu_msg_ptr)
{
  imu_frame_ = imu_msg_ptr->header.frame_id;
  geometry_msgs::msg::TransformStamped::ConstSharedPtr tf_imu2base_ptr =
    transform_listener_->getLatestTransform(imu_frame_, output_frame_);
  if (!tf_imu2base_ptr) {
    RCLCPP_ERROR(
      this->get_logger(), "Please publish TF %s to %s", output_frame_.c_str(),
      (imu_frame_).c_str());
    return;
  }

  geometry_msgs::msg::Vector3Stamped gyro;
  gyro.header.stamp = imu_msg_ptr->header.stamp;
  gyro.vector = transform_vector3(imu_msg_ptr->angular_velocity, *tf_imu2base_ptr);

  // 車両停止状態の判定
  bool is_currently_stopped = is_vehicle_currently_stopped();

  // 停車時のみデータを蓄積
  if (is_currently_stopped) {
    gyro_all_.push_back(gyro);
    stopped_sample_count_++;

    // FIFOの制限を実装（既存と同じ）
    if (gyro_all_.size() > MAX_BUFFER_SIZE) {
      gyro_all_.erase(gyro_all_.begin());
    }

    // 停車開始時のログ（状態変化時のみ）
    if (!is_vehicle_stopped_previous_) {
      RCLCPP_INFO(this->get_logger(), "Vehicle stopped - starting data collection");
      stopped_sample_count_ = 0;  // カウンタリセット
    }
  } else {
    // 停車終了時のログと情報表示
    if (is_vehicle_stopped_previous_) {
      RCLCPP_INFO(
        this->get_logger(), "Vehicle moving - collected %zu samples during stop",
        stopped_sample_count_);
    }
  }

  // 前回状態を更新
  is_vehicle_stopped_previous_ = is_currently_stopped;

  // Publish results for debugging
  if (gyro_bias_ != std::nullopt) {
    Vector3Stamped gyro_bias_msg;
    gyro_bias_msg.header.stamp = this->now();
    gyro_bias_msg.vector = gyro_bias_.value();
    gyro_bias_pub_->publish(gyro_bias_msg);
  }
}

void GyroBiasEstimator::callback_odom(const Odometry::ConstSharedPtr odom_msg_ptr)
{
  // 車両停止状態の判定
  bool is_currently_stopped = is_vehicle_currently_stopped();

  // 停車時のみポーズデータを蓄積
  if (is_currently_stopped) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = odom_msg_ptr->header;
    pose.pose = odom_msg_ptr->pose.pose;
    pose_buf_.push_back(pose);

    // FIFOの制限を実装（既存と同じ）
    if (pose_buf_.size() > MAX_BUFFER_SIZE) {
      pose_buf_.erase(pose_buf_.begin());
    }
  }
}

void GyroBiasEstimator::callback_twist_with_covariance(
  const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr msg)
{
  latest_twist_with_covariance_msg_ = msg;
  // RCLCPP_INFO(
  //   this->get_logger(),
  //   "Received twist_with_covariance: linear.x=%.6f, linear.y=%.6f, angular.z=%.6f",
  //   msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.angular.z);
}

// 新規追加：車両停止判定関数
bool GyroBiasEstimator::is_vehicle_currently_stopped()
{
  if (!latest_twist_with_covariance_msg_) {
    return false;
  }

  // 最新のtwist_with_covarianceデータが古すぎないかチェック
  const double twist_data_age =
    (this->now() - rclcpp::Time(latest_twist_with_covariance_msg_->header.stamp)).seconds();
  if (twist_data_age > 1.0) {  // 1秒以上古いデータは使わない
    return false;
  }

  // 停車判定：線形速度と角速度の両方をチェック
  const auto & twist = latest_twist_with_covariance_msg_->twist.twist;
  const double linear_velocity = std::sqrt(
    twist.linear.x * twist.linear.x + twist.linear.y * twist.linear.y +
    twist.linear.z * twist.linear.z);
  const double angular_velocity = std::abs(twist.angular.z);  // ヨー角速度のみチェック

  return (linear_velocity < vehicle_stop_velocity_threshold_) &&
         (angular_velocity < vehicle_stop_angular_velocity_threshold_);
}

void GyroBiasEstimator::timer_callback()
{
  if (pose_buf_.empty()) {
    diagnostics_info_.summary_message = "Skipped update (pose_buf is empty)";
    return;
  }

  // Copy data
  const std::vector<geometry_msgs::msg::PoseStamped> pose_buf = pose_buf_;
  const std::vector<geometry_msgs::msg::Vector3Stamped> gyro_all = gyro_all_;

  // Check time
  const rclcpp::Time t0_rclcpp_time = rclcpp::Time(pose_buf.front().header.stamp);
  const rclcpp::Time t1_rclcpp_time = rclcpp::Time(pose_buf.back().header.stamp);
  if (t1_rclcpp_time <= t0_rclcpp_time) {
    diagnostics_info_.summary_message = "Skipped update (pose_buf is not in chronological order)";
    return;
  }

  // Filter gyro data
  std::vector<geometry_msgs::msg::Vector3Stamped> gyro_filtered;
  for (const auto & gyro : gyro_all) {
    const rclcpp::Time t = rclcpp::Time(gyro.header.stamp);
    if (t0_rclcpp_time <= t && t < t1_rclcpp_time) {
      gyro_filtered.push_back(gyro);
    }
  }

  // Check gyro data size
  // Data size must be greater than or equal to 2 since the time difference will be taken later
  if (gyro_filtered.size() <= 1) {
    diagnostics_info_.summary_message = "Skipped update (gyro_filtered size is less than 2)";
    return;
  }

  // 十分なサンプル数があるかチェック（新規追加）
  if (stopped_sample_count_ < TARGET_STOPPED_SAMPLES) {
    diagnostics_info_.summary_message =
      "Collecting stopped samples: " + std::to_string(stopped_sample_count_) + "/" +
      std::to_string(TARGET_STOPPED_SAMPLES);
    return;
  }

  // 現在停車中でない場合はスキップ
  if (!is_vehicle_currently_stopped()) {
    diagnostics_info_.summary_message = "Vehicle is moving - bias estimation paused";
    return;
  }

  RCLCPP_INFO(
    this->get_logger(), "バイアス推定実行: samples=%zu, duration=%.2fs", gyro_filtered.size(),
    (t1_rclcpp_time - t0_rclcpp_time).seconds());

  // Calculate gyro bias
  gyro_bias_estimation_module_->update_bias(pose_buf, gyro_filtered);

  geometry_msgs::msg::TransformStamped::ConstSharedPtr tf_base2imu_ptr =
    transform_listener_->getLatestTransform(output_frame_, imu_frame_);
  if (!tf_base2imu_ptr) {
    RCLCPP_ERROR(
      this->get_logger(), "Please publish TF %s to %s", imu_frame_.c_str(), output_frame_.c_str());

    diagnostics_info_.summary_message = "Skipped update (tf between base and imu is not available)";
    return;
  }

  gyro_bias_ =
    transform_vector3(gyro_bias_estimation_module_->get_bias_base_link(), *tf_base2imu_ptr);

  validate_gyro_bias();
}

void GyroBiasEstimator::validate_gyro_bias()
{
  // Calculate diagnostics key-values
  diagnostics_info_.gyro_bias_x_for_imu_corrector = gyro_bias_.value().x;
  diagnostics_info_.gyro_bias_y_for_imu_corrector = gyro_bias_.value().y;
  diagnostics_info_.gyro_bias_z_for_imu_corrector = gyro_bias_.value().z;
  diagnostics_info_.estimated_gyro_bias_x = gyro_bias_.value().x - angular_velocity_offset_x_;
  diagnostics_info_.estimated_gyro_bias_y = gyro_bias_.value().y - angular_velocity_offset_y_;
  diagnostics_info_.estimated_gyro_bias_z = gyro_bias_.value().z - angular_velocity_offset_z_;

  // Validation
  const bool is_bias_small_enough =
    std::abs(diagnostics_info_.estimated_gyro_bias_x) < gyro_bias_threshold_ &&
    std::abs(diagnostics_info_.estimated_gyro_bias_y) < gyro_bias_threshold_ &&
    std::abs(diagnostics_info_.estimated_gyro_bias_z) < gyro_bias_threshold_;

  // Update diagnostics
  if (is_bias_small_enough) {
    diagnostics_info_.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    diagnostics_info_.summary_message = "Successfully updated";
  } else {
    diagnostics_info_.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    diagnostics_info_.summary_message =
      "Gyro bias may be incorrect. Please calibrate IMU and reflect the result in imu_corrector. "
      "You may also use the output of gyro_bias_estimator.";
  }
}

geometry_msgs::msg::Vector3 GyroBiasEstimator::transform_vector3(
  const geometry_msgs::msg::Vector3 & vec, const geometry_msgs::msg::TransformStamped & transform)
{
  geometry_msgs::msg::Vector3Stamped vec_stamped;
  vec_stamped.vector = vec;

  geometry_msgs::msg::Vector3Stamped vec_stamped_transformed;
  tf2::doTransform(vec_stamped, vec_stamped_transformed, transform);
  return vec_stamped_transformed.vector;
}

void GyroBiasEstimator::update_diagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  auto f = [](const double & value) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(8) << value;
    return ss.str();
  };

  stat.summary(diagnostics_info_.level, diagnostics_info_.summary_message);
  stat.add("gyro_bias_x_for_imu_corrector", f(diagnostics_info_.gyro_bias_x_for_imu_corrector));
  stat.add("gyro_bias_y_for_imu_corrector", f(diagnostics_info_.gyro_bias_y_for_imu_corrector));
  stat.add("gyro_bias_z_for_imu_corrector", f(diagnostics_info_.gyro_bias_z_for_imu_corrector));

  stat.add("estimated_gyro_bias_x", f(diagnostics_info_.estimated_gyro_bias_x));
  stat.add("estimated_gyro_bias_y", f(diagnostics_info_.estimated_gyro_bias_y));
  stat.add("estimated_gyro_bias_z", f(diagnostics_info_.estimated_gyro_bias_z));

  stat.add("gyro_bias_threshold", f(gyro_bias_threshold_));
}

}  // namespace autoware::imu_corrector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::imu_corrector::GyroBiasEstimator)