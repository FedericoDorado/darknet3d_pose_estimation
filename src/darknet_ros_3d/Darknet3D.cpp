/*********************************************************************
*  Software License Agreement (BSD License)
*
*   Copyright (c) 2019, Intelligent Robotics
*   All rights reserved.
*
*   Redistribution and use in source and binary forms, with or without
*   modification, are permitted provided that the following conditions
*   are met:

*    * Redistributions of source code must retain the above copyright
*      notice, this list of conditions and the following disclaimer.
*    * Redistributions in binary form must reproduce the above
*      copyright notice, this list of conditions and the following
*      disclaimer in the documentation and/or other materials provided
*      with the distribution.
*    * Neither the name of Intelligent Robotics nor the names of its
*      contributors may be used to endorse or promote products derived
*      from this software without specific prior written permission.

*   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*   COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*   POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Francisco Martín fmrico@gmail.com */
/* Author: Fernando González fergonzaramos@yahoo.es  */

#include "darknet_ros_3d/Darknet3D.h"

#include <ros/ros.h>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>


#include <gb_visual_detection_3d_msgs/BoundingBoxes3d.h>

#include <limits>
#include <algorithm>

#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <tf_conversions/tf_eigen.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>
#include <tf2/transform_datatypes.h>
#include <tf2_msgs/TFMessage.h>


#include <geometry_msgs/Transform.h>
#include <geometry_msgs/TransformStamped.h>

#include </usr/include/opencv4/opencv2/opencv.hpp>
#include </usr/include/opencv4/opencv2/core.hpp>
#include </usr/include/opencv4/opencv2/core/eigen.hpp>
#include </usr/include/opencv4/opencv2/core/persistence.hpp>
#include </usr/include/opencv4/opencv2/core/types.hpp>
#include </usr/include/opencv4/opencv2/core/mat.hpp>
#include </usr/include/opencv4/opencv2/core/matx.hpp>
#include </usr/include/opencv4/opencv2/core/affine.hpp>

#include <darknet_ros_msgs/PCAValues.h>


Eigen::Vector3f centroid; // Vector para almacenar el centroide

Eigen::Quaternionf quaternion; // Quaternion para representar la orientación

Eigen::Affine3f transform; // Transformación afín para establecer el marco de coordenadas 3D


namespace darknet_ros_3d
{

Darknet3D::Darknet3D():
  nh_("~")
{
  initParams();

  darknet3d_pub_ = nh_.advertise<gb_visual_detection_3d_msgs::BoundingBoxes3d>(output_bbx3d_topic_, 100);
  markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/darknet_ros_3d/markers", 100);
  centroid_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/darknet_ros_3d/Centroid", 100);
  isolated_pointcloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/isolated_pointcloud", 1);
  pca_pub_ = nh_.advertise<darknet_ros_msgs::PCAValues>("/pca_results_topic", 1);


  yolo_sub_ = nh_.subscribe(input_bbx_topic_, 1, &Darknet3D::darknetCb, this);
  pointCloud_sub_ = nh_.subscribe(pointcloud_topic_, 1, &Darknet3D::pointCloudCb, this);

  last_detection_ts_ = ros::Time::now();// - ros::Duration(60.0);
}

void
Darknet3D::initParams()
{
  input_bbx_topic_ = "/darknet_ros/bounding_boxes";
  output_bbx3d_topic_ = "/darknet_ros_3d/bounding_boxes";
  pointcloud_topic_ = "/kinect2/hd/points";
  working_frame_ = "/custom_ir_frame";
  mininum_detection_thereshold_ = 0.5f;
  minimum_probability_ = 0.3f;

  nh_.param("darknet_ros_topic", input_bbx_topic_, input_bbx_topic_);
  nh_.param("output_bbx3d_topic", output_bbx3d_topic_, output_bbx3d_topic_);
  nh_.param("point_cloud_topic", pointcloud_topic_, pointcloud_topic_);
  nh_.param("working_frame", working_frame_, working_frame_);
  nh_.param("mininum_detection_thereshold", mininum_detection_thereshold_, mininum_detection_thereshold_);
  nh_.param("minimum_probability", minimum_probability_, minimum_probability_);
  nh_.param("interested_classes", interested_classes_, interested_classes_);
}

void
Darknet3D::pointCloudCb(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  point_cloud_ = *msg;
}

void
Darknet3D::darknetCb(const darknet_ros_msgs::BoundingBoxes::ConstPtr& msg)
{
  last_detection_ts_ = ros::Time::now();
  original_bboxes_ = msg->bounding_boxes;
}


// Función Calcular trasformaciones para el centroide

Eigen::Affine3f calculateTransform(const Eigen::Vector3f& centroid, const Eigen::Quaternionf& quaternion)
{
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << centroid(0), centroid(1), centroid(2);
    transform.rotate(quaternion);
    return transform;
}

void
Darknet3D::calculate_boxes(const sensor_msgs::PointCloud2& cloud_pc2,
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_pcl,
    gb_visual_detection_3d_msgs::BoundingBoxes3d* boxes)
{
  boxes->header.stamp = cloud_pc2.header.stamp;
  boxes->header.frame_id = working_frame_;

  for (auto bbx : original_bboxes_)
  {
    if ((bbx.probability < minimum_probability_) ||
        (std::find(interested_classes_.begin(), interested_classes_.end(), bbx.Class) == interested_classes_.end()))
    {
      continue;
    }

    int center_x, center_y;

    center_x = (bbx.xmax + bbx.xmin) / 2;
    center_y = (bbx.ymax + bbx.ymin) / 2;

    int pcl_index = (center_y* cloud_pc2.width) + center_x;
    pcl::PointXYZRGB center_point =  cloud_pcl->at(pcl_index);

    if (std::isnan(center_point.x))
      continue;

    float maxx, minx, maxy, miny, maxz, minz;

    maxx = maxy = maxz =  -std::numeric_limits<float>::max();
    minx = miny = minz =  std::numeric_limits<float>::max();

    for (int i = bbx.xmin; i < bbx.xmax; i++)
      for (int j = bbx.ymin; j < bbx.ymax; j++)
      {
        pcl_index = (j* cloud_pc2.width) + i;
        pcl::PointXYZRGB point =  cloud_pcl->at(pcl_index);

        if (std::isnan(point.x))
          continue;

        if (fabs(point.x - center_point.x) > mininum_detection_thereshold_)
          continue;

        maxx = std::max(point.x, maxx);
        maxy = std::max(point.y, maxy);
        maxz = std::max(point.z, maxz);
        minx = std::min(point.x, minx);
        miny = std::min(point.y, miny);
        minz = std::min(point.z, minz);
      }
   

      // Cálculo del centroide

        centroid = Eigen::Vector3f((maxx + minx) / 2, (maxy + miny) / 2, (maxz + minz) / 2);

        Eigen::Vector3f x_dir = Eigen::Vector3f::UnitX();
        Eigen::Vector3f y_dir = Eigen::Vector3f::UnitY();
        Eigen::Vector3f z_dir = Eigen::Vector3f::UnitZ();

        Eigen::Vector3f up_vector = y_dir;
        Eigen::Vector3f bbx_dir = Eigen::Vector3f(maxx - minx, maxy - miny, maxz - minz).normalized();

        quaternion.setFromTwoVectors(up_vector, bbx_dir);

        transform = calculateTransform(centroid, quaternion);

        // Se invoca la función para aislar el objeto detectado en una nube de puntos.

        publish_isolated_pointcloud(cloud_pcl, minx, maxx, miny, maxy, minz, maxz, working_frame_);
        

        //Publicar Marcadores Para el Centroide del Objeto detectado
    
        visualization_msgs::MarkerArray msg_centroid;

                // Create marker message
        visualization_msgs::Marker marker;
        marker.header.frame_id = working_frame_;
        marker.header.stamp = ros::Time::now();
        marker.ns = "centroid_marker";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = centroid(0);
        marker.pose.position.y = centroid(1);
        marker.pose.position.z = centroid(2);
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.02;
        marker.scale.y = 0.02;
        marker.scale.z = 0.02;
        marker.color.a = 1.0; // alpha channel
        marker.color.r = 1.0; // red channel
        marker.color.g = 0.0; // green channel
        marker.color.b = 0.0; // blue channel

        msg_centroid.markers.push_back(marker);

        // Publish marker message
        centroid_pub_.publish(msg_centroid);

        tf::TransformListener listener;

        tf::StampedTransform transform;
          try {
            listener.lookupTransform(working_frame_, working_frame_, ros::Time(0), transform);
          } catch (tf::TransformException ex) {
            ROS_ERROR("Could not get transform from base_link to object");
            return;
          }

    gb_visual_detection_3d_msgs::BoundingBox3d bbx_msg;
    bbx_msg.Class = bbx.Class;
    bbx_msg.probability = bbx.probability;
    bbx_msg.xmin = minx;
    bbx_msg.xmax = maxx;
    bbx_msg.ymin = miny;
    bbx_msg.ymax = maxy;
    bbx_msg.zmin = minz;
    bbx_msg.zmax = maxz;

    boxes->bounding_boxes.push_back(bbx_msg);

    }
}


void
Darknet3D::update()
{
  if ((ros::Time::now() - last_detection_ts_).toSec() > 2.0)
    return;

  if ((darknet3d_pub_.getNumSubscribers() == 0) &&
      (markers_pub_.getNumSubscribers() == 0) &&
      (centroid_pub_.getNumSubscribers() == 0))

    return;

  sensor_msgs::PointCloud2 local_pointcloud;

  try
  {
    pcl_ros::transformPointCloud(working_frame_, point_cloud_, local_pointcloud, tfListener_);
  }
  catch(tf::TransformException& ex)
  {
    ROS_ERROR_STREAM("Transform error of sensor data: " << ex.what() << ", quitting callback");
    return;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(local_pointcloud, *pcrgb);

  gb_visual_detection_3d_msgs::BoundingBoxes3d msg;

  calculate_boxes(local_pointcloud, pcrgb, &msg);

  darknet3d_pub_.publish(msg);

  publish_markers(msg);




}

// Función para aislar la nube de puntos del objeto detectado y trabajarla con PCA.
// Argumentos necesarios: nube de puntos original (cloud_pcl), las coordenadas del objeto detectado y el marco de referencia (frame_id = working_frame).

void Darknet3D::publish_isolated_pointcloud(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_pcl,
    const float xmin, const float xmax, const float ymin, const float ymax,
    const float zmin, const float zmax, const std::string& frame_id)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_cloud(new pcl::PointCloud<pcl::PointXYZRGB>); // INstancia de tipo Pcl para almacenar los datos aislados.

  // Itera dentro de la nube de puntos original y verifica los puntos que están dentro de las coordenadas del objeto detectado.
  // Si es así, se añade el punto a 'isolated_cloud'.
    

  for (const auto& point : cloud_pcl->points)
  {
    if (point.x >= xmin && point.x <= xmax &&
        point.y >= ymin && point.y <= ymax &&
        point.z >= zmin && point.z <= zmax)
    {
      isolated_cloud->push_back(point);
    }
  }

  // Creación de un nuevo mensaje 'sensor_msgs::PointCloud2' llamado 'isolated_cloud_msg para visualización en rviz.
  // CONvierte la nuve 'isolated_cloud' en este mensaje utilizando pcl::toROSMsg().

  sensor_msgs::PointCloud2 isolated_cloud_msg;
  pcl::toROSMsg(*isolated_cloud, isolated_cloud_msg);

  // Configuración del encabezado 'header' del mensaje 'isolated_cloud_msg' con el marco de referencia 'frame_id'

  isolated_cloud_msg.header.frame_id = frame_id;

  // Publicación del mensaje en el nuevo topic "/isolated_pointcloud".
  isolated_pointcloud_pub_.publish(isolated_cloud_msg);

    // Calcular el centroide utilizando PCL
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_point_cloud_ptr(isolated_cloud);
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*isolated_point_cloud_ptr, centroid);
  

  // Invocar función para aplicar PCA a la nube de puntos aislada
  Eigen::Matrix3f eigen_vectors;
  Eigen::Vector3f eigen_values;
  apply_pca_to_pointcloud(isolated_cloud, eigen_vectors, eigen_values);


}


void Darknet3D::apply_pca_to_pointcloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_cloud,
 Eigen::Matrix3f& eigen_vectors, Eigen::Vector3f& eigen_values)
{
  // Creación una matriz OpenCV a partir de la nube de puntos aislada
  cv::Mat cloud_mat(isolated_cloud->size(), 3, CV_32F);
  for (size_t i = 0; i < isolated_cloud->size(); ++i)
  {
    cloud_mat.at<float>(i, 0) = isolated_cloud->points[i].x;
    cloud_mat.at<float>(i, 1) = isolated_cloud->points[i].y;
    cloud_mat.at<float>(i, 2) = isolated_cloud->points[i].z;
  }

  // Calcular PCA utilizando OpenCV
  cv::PCA pca_analysis(cloud_mat, cv::Mat(), cv::PCA::DATA_AS_ROW);

  // Obtener los vectores propios y los valores propios
  cv::Mat eigen_vectors_mat = pca_analysis.eigenvectors.t();  // Transponer para obtener vectores propios en columnas
  cv::Mat eigen_values_mat = pca_analysis.eigenvalues;

  // Convertir los resultados de PCA a matrices Eigen
  cv::cv2eigen(eigen_vectors_mat, eigen_vectors);
  cv::cv2eigen(eigen_values_mat, eigen_values);


  // Crear el mensaje para los valores de PCA
  darknet_ros_msgs::PCAValues pca_msg;
  pca_msg.header.stamp = ros::Time::now();
  pca_msg.eigen_vectors.resize(3);
  pca_msg.eigen_values.resize(3);

  // Copiar los valores de PCA a los campos del mensaje
  for (int i = 0; i < 3; ++i)
  {
    pca_msg.eigen_vectors[i].x = eigen_vectors(0, i);
    pca_msg.eigen_vectors[i].y = eigen_vectors(1, i);
    pca_msg.eigen_vectors[i].z = eigen_vectors(2, i);
    pca_msg.eigen_values[i] = eigen_values(i);
  }

  // Publicar el mensaje
  pca_pub_.publish(pca_msg);  

  // Llamado para obtener la transformación del objeto
  publish_coordinate_frame(eigen_vectors, centroid);

}

void Darknet3D::publish_coordinate_frame(const Eigen::Matrix3f& eigen_vectors, const Eigen::Vector3f& centroid)
{
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;

  // Configura el marco de tiempo y el marco de referencia
  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = "custom_ir_frame"; // Marco de referencia de la aplicación

  // Configura el marco de referencia hijo (nombre del eje de coordenadas)
  transformStamped.child_frame_id = "object_coordinate_frame";

  // Configura la posición del marco de referencia utilizando el centroide
  transformStamped.transform.translation.x = centroid(0);
  transformStamped.transform.translation.y = centroid(1);
  transformStamped.transform.translation.z = centroid(2);

  // Calcula la orientación del marco de referencia utilizando los vectores propios
  Eigen::Quaternionf quat(eigen_vectors);
  transformStamped.transform.rotation.x = quat.x();
  transformStamped.transform.rotation.y = quat.y();
  transformStamped.transform.rotation.z = quat.z();
  transformStamped.transform.rotation.w = quat.w();

  // Publica el mensaje TransformStamped
  br.sendTransform(transformStamped);

}


void
Darknet3D::publish_markers(const gb_visual_detection_3d_msgs::BoundingBoxes3d& boxes)
{
  visualization_msgs::MarkerArray msg;

  int counter_id = 0;
  for (auto bb : boxes.bounding_boxes)
  {
    visualization_msgs::Marker bbx_marker;

    bbx_marker.header.frame_id = boxes.header.frame_id;
    bbx_marker.header.stamp = boxes.header.stamp;
    bbx_marker.ns = "darknet3d";
    bbx_marker.id = counter_id++;
    bbx_marker.type = visualization_msgs::Marker::CUBE;
    bbx_marker.action = visualization_msgs::Marker::ADD;
    bbx_marker.pose.position.x = (bb.xmax + bb.xmin) / 2.0;
    bbx_marker.pose.position.y = (bb.ymax + bb.ymin) / 2.0;
    bbx_marker.pose.position.z = (bb.zmax + bb.zmin) / 2.0;
    bbx_marker.pose.orientation.x = 0.0;
    bbx_marker.pose.orientation.y = 0.0;
    bbx_marker.pose.orientation.z = 0.0;
    bbx_marker.pose.orientation.w = 1.0;
    bbx_marker.scale.x = (bb.xmax - bb.xmin);
    bbx_marker.scale.y = (bb.ymax - bb.ymin);
    bbx_marker.scale.z = (bb.zmax - bb.zmin);
    bbx_marker.color.b = 0;
    bbx_marker.color.g = bb.probability * 255.0;
    bbx_marker.color.r = (1.0 - bb.probability) * 255.0;
    bbx_marker.color.a = 0.4;
    bbx_marker.lifetime = ros::Duration(0.5);

    msg.markers.push_back(bbx_marker);
  }

  markers_pub_.publish(msg);
}

};  // namespace darknet_ros_3d
