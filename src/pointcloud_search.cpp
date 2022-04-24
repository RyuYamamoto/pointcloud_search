#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pointcloud_search/kd_tree.hpp>

class PointCloudSearch : public rclcpp::Node
{
public:
  PointCloudSearch(const rclcpp::NodeOptions & node_options)
  : Node("pointcloud_search", node_options)
  {
    tree_ptr_ = std::make_shared<KDTree<pcl::PointXYZ>>();

    radius_ = this->declare_parameter<double>("radius", 3.0);

    map_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "points_map", rclcpp::QoS{1}.transient_local(),
      std::bind(&PointCloudSearch::mapCallback, this, std::placeholders::_1));
    initialpose_subscriber_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "initialpose", 1,
        std::bind(&PointCloudSearch::initialPoseCallback, this, std::placeholders::_1));
    result_points_publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("result_points", 5);
  }
  ~PointCloudSearch() = default;

  void mapCallback(const sensor_msgs::msg::PointCloud2 & map)
  {
    RCLCPP_INFO(get_logger(), "map callback");

    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(map, *map_cloud);

    tree_ptr_->setInputCloud(map_cloud);
  }

  void initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped & initialpose)
  {
    RCLCPP_INFO(get_logger(), "initialpose callback");
    pcl::PointXYZ point;
    point.x = initialpose.pose.pose.position.x;
    point.y = initialpose.pose.pose.position.y;
    point.z = 0.0;

    std::vector<Vector3> radius_points = tree_ptr_->radiusSearch(point, radius_);
    pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);

    for (auto points_vec : radius_points) {
      pcl::PointXYZ p;
      p.x = points_vec[0];
      p.y = points_vec[1];
      p.z = points_vec[2];
      result->points.emplace_back(p);
    }
    sensor_msgs::msg::PointCloud2 result_msg;
    pcl::toROSMsg(*result, result_msg);
    result_msg.header.frame_id = initialpose.header.frame_id;
    result_msg.header.stamp = rclcpp::Clock().now();
    result_points_publisher_->publish(result_msg);
  }

private:
  std::shared_ptr<KDTree<pcl::PointXYZ>> tree_ptr_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    initialpose_subscriber_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr result_points_publisher_;

  double radius_;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(PointCloudSearch)
