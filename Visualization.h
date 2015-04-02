//
//  Visualization.h
//  RCT
//
//  Created by DarkTango on 3/14/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __RCT__Visualization__
#define __RCT__Visualization__
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

void RunVisualizationThread();
void WaitForVisualizationThread();
void RunVisualizationOnly();
void RunVisualization(const std::vector<cv::Point3d>& pointcloud,
                      const std::vector<cv::Vec3b>& pointcloud_RGB = std::vector<cv::Vec3b>(),
                      const std::vector<cv::Point3d>& pointcloud1 = std::vector<cv::Point3d>(),
                      const std::vector<cv::Vec3b>& pointcloud1_RGB = std::vector<cv::Vec3b>());
void ShowClouds(const std::vector<cv::Point3d>& pointcloud,
                const std::vector<cv::Vec3b>& pointcloud_RGB = std::vector<cv::Vec3b>(),
                const std::vector<cv::Point3d>& pointcloud1 = std::vector<cv::Point3d>(),
                const std::vector<cv::Vec3b>& pointcloud1_RGB = std::vector<cv::Vec3b>());
void ShowCloud(const std::vector<cv::Point3d>& pointcloud,
               const std::vector<cv::Vec3b>& pointcloud_RGB,
               const std::string& name);

void visualizerShowCamera(const float R[9], const float t[3], float r, float g, float b);
void visualizerShowCamera(const cv::Matx33f& R, const cv::Vec3f& t, float r, float g, float b, double s, const std::string& name = "");
#endif /* defined(__RCT__Visualization__) */
