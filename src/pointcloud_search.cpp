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

  void mapCallback(const sensor_msgs::msg::PointCloud2 & map)
  {
    if(map_update_) {return;}
    RCLCPP_INFO(get_logger(), "map callback");

    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(map, *map_cloud);

    voxel_grid_covariance_ptr_->setLeafSize(2.0, 2.0, 2.0);
    voxel_grid_covariance_ptr_->setInputCloud(map_cloud);
    auto leaf_map = voxel_grid_covariance_ptr_->getLeafMap();

    int id = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid_points(new pcl::PointCloud<pcl::PointXYZ>);
    visualization_msgs::msg::MarkerArray marker_array;
    for(auto leaf : leaf_map){
      visualization_msgs::msg::Marker marker;
      pcl::PointXYZ p;
      p.x = leaf.second.mean.x();
      p.y = leaf.second.mean.y();
      p.z = leaf.second.mean.z();
      voxel_grid_points->points.emplace_back(p);

      marker.header.frame_id = "map";
      marker.header.stamp = rclcpp::Clock().now();
      marker.ns = "nd voxel";
      marker.id = id++;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.pose.position.x = p.x;
      marker.pose.position.y = p.y;
      marker.pose.position.z = p.z;
      Eigen::Quaternionf q(leaf.second.eigenvec);
      marker.pose.orientation.x = q.x();
      marker.pose.orientation.y = q.y();
      marker.pose.orientation.z = q.z();
      marker.pose.orientation.w = q.w();
      marker.scale.x = std::sqrt(leaf.second.eigenvalues(0)*9.21034);
      marker.scale.y = std::sqrt(leaf.second.eigenvalues(1)*9.21034);
      marker.scale.z = std::sqrt(leaf.second.eigenvalues(2)*9.21034);
      marker.color.r = 0.0;
      marker.color.g = 1.0;
      marker.color.b = 0.0;
      marker.color.a = 0.5;
      marker_array.markers.emplace_back(marker);
    }
    voxel_grid_points->header.frame_id = "map";

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
    std::vector<Vector3> radius_points = tree_ptr_->radiusSearch(point, radius_, indices);
    std::cout << "size: " << indices.size() << std::endl;
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

    int id=0;
    visualization_msgs::msg::MarkerArray marker_array;
    for (auto points_vec : radius_points) {
      pcl::PointXYZ p;
      p.x = points_vec[0];
      p.y = points_vec[1];
      p.z = points_vec[2];
      result->points.emplace_back(p);

      std::size_t indice=0;
      // TODO index search
      for(auto leaf : leafs) {
        if(leaf.second.mean[0] == p.x and leaf.second.mean[1] == p.y and leaf.second.mean[2] == p.voxel_centroids_leaf_indices_z) {
          indice = leaf.first;
          break;
        }
      }

      // voxel
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = "map";
      marker.header.stamp = rclcpp::Clock().now();
      marker.ns = "nd voxel result";
      marker.id = id++;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.pose.position.x = leafs[indice].mean[0];
      marker.pose.position.y = leafs[indice].mean[1];
      marker.pose.position.z = leafs[indice].mean[2];
      Eigen::Quaternionf q(leafs[indice].eigenvec);
      marker.pose.orientation.x = q.x();
      marker.pose.orientation.y = q.y();
      marker.pose.orientation.z = q.z();
      marker.pose.orientation.w = q.w();
      marker.scale.x = std::sqrt(leafs[indice].eigenvalues(0)*9.21034);
      marker.scale.y = std::sqrt(leafs[indice].eigenvalues(1)*9.21034);
      marker.scale.z = std::sqrt(leafs[indice].eigenvalues(2)*9.21034);
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
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
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr nd_voxel_result_marker_publisher_;

  double radius_;
  bool map_update_{false};
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(PointCloudSearch)
