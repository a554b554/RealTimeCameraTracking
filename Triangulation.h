//
//  Triangulation.h
//  RCT
//
//  Created by DarkTango on 3/13/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __RCT__Triangulation__
#define __RCT__Triangulation__

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#ifdef __SFM__DEBUG__
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <vector>

#include "Common.h"

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,		//homogenous image point (u,v,1)
                                       cv::Matx34d P,		//camera 1 matrix
                                       cv::Point3d u1,		//homogenous image point in 2nd camera
                                       cv::Matx34d P1		//camera 2 matrix
);

#define EPSILON 0.0001
/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
cv::Mat_<double> IterativeLinearLSTriangulation(cv::Point3d u,	//homogenous image point (u,v,1)
                                                cv::Matx34d P,			//camera 1 matrix
                                                cv::Point3d u1,			//homogenous image point in 2nd camera
                                                cv::Matx34d P1			//camera 2 matrix
);

double TriangulatePoints(const std::vector<cv::KeyPoint>& pt_set1,
                         const std::vector<cv::KeyPoint>& pt_set2,
                         const cv::Mat& K,
                         const cv::Mat& Kinv,
                         const cv::Mat& distcoeff,
                         const cv::Matx34d& P,
                         const cv::Matx34d& P1,
                         std::vector<CloudPoint>& pointcloud,
                         std::vector<cv::KeyPoint>& correspImg1Pt);


#endif /* defined(__RCT__Triangulation__) */
