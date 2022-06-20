#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pointcloud_search/kd_tree.hpp>
#include <pointcloud_search/voxel_grid.hpp>

class PointCloudSearch : public rclcpp::Node
{
public:
  PointCloudSearch(const rclcpp::NodeOptions & node_options)
  : Node("pointcloud_search", node_options)
  {
    tree_ptr_ = std::make_shared<KDTree<pcl::PointXYZ>>();
    voxel_grid_covariance_ptr_ = std::make_shared<VoxelGridCovariance<pcl::PointXYZ>>();

    radius_ = this->declare_parameter<double>("radius", 3.0);

    rclcpp::QoS qos{1};
    qos.transient_local();
    qos.keep_last(1);

    map_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "points_map", rclcpp::QoS{1}.transient_local(),
      std::bind(&PointCloudSearch::mapCallback, this, std::placeholders::_1));
    initialpose_subscriber_ =
      this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "initialpose", 1,
        std::bind(&PointCloudSearch::initialPoseCallback, this, std::placeholders::_1));
    result_points_publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("result_points", 5);
    voxel_grid_points_publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>("voxel_grid_points", qos);
    nd_voxel_marker_publisher_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>("nd_voxel", qos);
    nd_voxel_result_marker_publisher_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>("nd_voxel_result", 5);
  }
  ~PointCloudSearch() = default;

  visualization_msgs::msg::Marker convertLeafToMarker(
    const Leaf leaf, const double r, const double g, const double b, const int id)
  {
    visualization_msgs::msg::Marker marker;

    marker.header.frame_id = "map";
    marker.header.stamp = rclcpp::Clock().now();
    marker.ns = "nd voxel";
    marker.id = id;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.pose.position.x = leaf.mean[0];
    marker.pose.position.y = leaf.mean[1];
    marker.pose.position.z = leaf.mean[2];
    Eigen::Quaterniond q(leaf.eigenvec);
    marker.pose.orientation.x = q.x();
    marker.pose.orientation.y = q.y();
    marker.pose.orientation.z = q.z();
    marker.pose.orientation.w = q.w();
    marker.scale.x = std::sqrt(leaf.eigenvalues(0) * 9.21034);
    marker.scale.y = std::sqrt(leaf.eigenvalues(1) * 9.21034);
    marker.scale.z = std::sqrt(leaf.eigenvalues(2) * 9.21034);
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 0.5;

    return marker;
  }
  visualization_msgs::msg::MarkerArray convertLeafToMarkerArray(
    const std::map<std::size_t, Leaf> leafs, const double r, const double g, const double b)
  {
    int id = -1;
    visualization_msgs::msg::MarkerArray marker_array;
    for (auto leaf : leafs) {
      visualization_msgs::msg::Marker marker = convertLeafToMarker(leaf.second, r, g, b, id++);
      marker_array.markers.emplace_back(marker);
    }
    return marker_array;
  }

  void mapCallback(const sensor_msgs::msg::PointCloud2 & map)
  {
    if (map_update_) {
      return;
    }
    RCLCPP_INFO(get_logger(), "map callback");

    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(map, *map_cloud);

    voxel_grid_covariance_ptr_->setLeafSize(2.0, 2.0, 2.0);
    voxel_grid_covariance_ptr_->setInputCloud(map_cloud);

    auto leaf_map = voxel_grid_covariance_ptr_->getLeafMap();
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid_points =
      voxel_grid_covariance_ptr_->getFilteredPoints();
    voxel_grid_points->header.frame_id = "map";

    visualization_msgs::msg::MarkerArray marker_array =
      convertLeafToMarkerArray(leaf_map, 0.0, 1.0, 0.0);

    sensor_msgs::msg::PointCloud2 voxel_points_msg;
    voxel_points_msg.header.frame_id = "map";
    voxel_points_msg.header.stamp = rclcpp::Clock().now();
    pcl::toROSMsg(*voxel_grid_points, voxel_points_msg);

    voxel_grid_points_publisher_->publish(voxel_points_msg);
    nd_voxel_marker_publisher_->publish(marker_array);

    tree_ptr_->setInputCloud(voxel_grid_points);

    RCLCPP_INFO(get_logger(), "create kd tree.");
    map_update_ = true;
  }

  void initialPoseCallback(const geometry_msgs::msg::PoseWithCovarianceStamped & initialpose)
  {
    RCLCPP_INFO(get_logger(), "initialpose callback");
    pcl::PointXYZ point;
    point.x = initialpose.pose.pose.position.x;
    point.y = initialpose.pose.pose.position.y;
    point.z = 0.0;

    std::vector<int> indices;
    tree_ptr_->radiusSearch(point, radius_, indices);
    RCLCPP_INFO(get_logger(), "search radius");
    pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);

    // clear voxel
    visualization_msgs::msg::MarkerArray clear_markers;
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.header.frame_id = "map";
    clear_marker.header.stamp = rclcpp::Clock().now();
    clear_marker.ns = "clear_marker";
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    clear_markers.markers.emplace_back(clear_marker);
    nd_voxel_result_marker_publisher_->publish(clear_markers);

    auto leafs = voxel_grid_covariance_ptr_->getLeafMap();
    auto leafs_index = voxel_grid_covariance_ptr_->getVoxelGridIndex();
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_points =
      voxel_grid_covariance_ptr_->getFilteredPoints();

    int id = -1;
    visualization_msgs::msg::MarkerArray marker_array;
    for (auto indice : indices) {
      pcl::PointXYZ p;
      p.x = filtered_points->points[indice].x;
      p.y = filtered_points->points[indice].y;
      p.z = filtered_points->points[indice].z;
      result->points.emplace_back(p);

      auto leaf = leafs.find(leafs_index[indice]);
      // voxel
      visualization_msgs::msg::Marker marker =
        convertLeafToMarker(leaf->second, 1.0, 0.0, 0.0, id++);
      marker_array.markers.emplace_back(marker);
    }
    sensor_msgs::msg::PointCloud2 result_msg;
    pcl::toROSMsg(*result, result_msg);
    result_msg.header.frame_id = initialpose.header.frame_id;
    result_msg.header.stamp = rclcpp::Clock().now();
    result_points_publisher_->publish(result_msg);

    nd_voxel_result_marker_publisher_->publish(marker_array);
  }

private:
  std::shared_ptr<KDTree<pcl::PointXYZ>> tree_ptr_;
  std::shared_ptr<VoxelGridCovariance<pcl::PointXYZ>> voxel_grid_covariance_ptr_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    initialpose_subscriber_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr result_points_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr voxel_grid_points_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr nd_voxel_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    nd_voxel_result_marker_publisher_;

  double radius_;
  bool map_update_{false};
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(PointCloudSearch)
