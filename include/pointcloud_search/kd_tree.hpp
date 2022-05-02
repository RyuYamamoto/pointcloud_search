#ifndef _KD_TREE_HPP_
#define _KD_TREE_HPP_

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct Node
{
  int axis;
  int idx;
  Node * right;
  Node * left;
  std::vector<double> median;
  Node()
  {
    axis = -1;
    idx = -1;
    right = nullptr;
    left = nullptr;
  }
};

typedef std::vector<double> Vector3;

template <class PointType>
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
    indices_.resize(cloud->points.size());
    std::iota(indices_.begin(), indices_.end(), 0);

    node_ = build(0, cloud->points.size() - 1, 0);
  }

  Node * build(int l, int r, int depth)
  {
    if (r <= l) return nullptr;

    int median = (l + r) >> 1;
    int axis = depth % 3;
    std::sort(indices_.begin() + l, indices_.begin() + r, [&](int lhs, int rhs) {
      return target_[lhs][axis] < target_[rhs][axis];
    });

    Node * node = new Node();
    node->axis = axis;
    node->idx = indices_[median];
    node->median = target_[node->idx];
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

  void radiusSearch(const PointType point, const double radius, std::vector<int> & indices)
  {
    Vector3 query{point.x, point.y, point.z};
    radiusSearchRecursive(query, radius, node_, indices);
  }
  void radiusSearchRecursive(
    const Vector3 query, const double radius, Node * node, std::vector<int> & indices)
  {
    if (node == nullptr) return;

    const double distance = calcEuclideanDistance(node->median, query);
    if (distance < radius) {
      indices.emplace_back(node->idx);
    }

    Node * next;
    if (query[node->axis] < node->median[node->axis]) {
      radiusSearchRecursive(query, radius, node->left, indices);
      next = node->right;
    } else {
      radiusSearchRecursive(query, radius, node->right, indices);
      next = node->left;
    }

    const double axis_diff = std::fabs(query[node->axis] - node->median[node->axis]);
    if (axis_diff < radius) {
      radiusSearchRecursive(query, radius, next, indices);
    }

    return;
  }

private:
  Node * node_;
  std::vector<Vector3> target_;
  std::vector<int> indices_;
};

#endif
