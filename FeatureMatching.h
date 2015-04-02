//
//  FeatureMatching.h
//  RCT
//
//  Created by DarkTango on 3/13/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __RCT__FeatureMatching__
#define __RCT__FeatureMatching__
#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "IDistance.h"

#include "Common.h"

void MatchFeatures(const cv::Mat& img_1, const cv::Mat& img_1_orig,
                   const cv::Mat& img_2, const cv::Mat& img_2_orig,
                   const std::vector<cv::KeyPoint>& imgpts1,
                   const std::vector<cv::KeyPoint>& imgpts2,
                   cv::Mat& descriptors_1,
                   cv::Mat& descriptors_2,
                   std::vector<cv::KeyPoint>& fullpts1,
                   std::vector<cv::KeyPoint>& fullpts2,
                   int stretegy = STRATEGY_USE_OPTICAL_FLOW + STRATEGY_USE_DENSE_OF + STRATEGY_USE_FEATURE_MATCH,
                   std::vector<cv::DMatch>* matches = NULL);

#endif /* defined(__RCT__FeatureMatching__) */
