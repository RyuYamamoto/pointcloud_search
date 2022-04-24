#ifndef _KD_TREE_HPP_
#define _KD_TREE_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/voxel_grid.h>

struct Node
{
  int axis;
  Node * right;
  Node * left;
  std::vector<double> median;
  Node()
  {
    axis = -1;
    right = nullptr;
    left = nullptr;
  }
};

typedef std::vector<double> Vector3;

template <typename PointType>
class KDTree
{
public:
  KDTree() {}
  ~KDTree() = default;

  void setInputCloud(const typename pcl::PointCloud<PointType>::Ptr cloud)
  {
    target_.clear();
    for (auto p : cloud->points) {
      Vector3 point;
      point = {p.x, p.y, p.z};
      target_.emplace_back(point);
    }

    node_ = build(0, cloud->points.size() - 1, 0);
  }

  Node * build(int l, int r, int depth)
  {
    if (r <= l) return nullptr;

    int median = (l + r) >> 1;
    int axis = depth % 3;
    std::sort(target_.begin() + l, target_.begin() + r, [&](const auto lhs, const auto rhs) {
      return lhs[axis] < rhs[axis];
    });

    Node * node = new Node();
    node->axis = axis;
    node->median = target_[median];
    node->left = build(l, median, depth + 1);
    node->right = build(median + 1, r, depth + 1);

    return node;
  }

  double calcEuclideanDistance(const Vector3 p1, const Vector3 p2)
  {
    double dist = 0.0;
    for (std::size_t idx = 0; idx < target_.begin()->size(); idx++)
      dist += ((p1[idx] - p2[idx]) * (p1[idx] - p2[idx]));
    return std::sqrt(dist);
  }

  std::vector<Vector3> radiusSearch(const PointType point, const double radius)
  {
    Vector3 query{point.x, point.y, point.z};
    std::vector<Vector3> radius_points;
    radiusSearchRecursive(query, radius, node_, radius_points);
    return radius_points;
  }
  void radiusSearchRecursive(
    const Vector3 query, const double radius, Node * node, std::vector<Vector3> & radius_points)
  {
    if (node == nullptr) return;

    const double distance = calcEuclideanDistance(node->median, query);
    if (distance < radius) {
      radius_points.push_back(node->median);
    }

    Node * next;
    if (query[node->axis] < node->median[node->axis]) {
      radiusSearchRecursive(query, radius, node->left, radius_points);
      next = node->right;
    } else {
      radiusSearchRecursive(query, radius, node->right, radius_points);
      next = node->left;
    }

    const double axis_diff = std::fabs(query[node->axis] - node->median[node->axis]);
    if (axis_diff < radius) {
      radiusSearchRecursive(query, radius, next, radius_points);
    }

    return;
  }

private:
  Node * node_;
  std::vector<Vector3> target_;
};

#endif
