//
//  SIFTFeatureMatcher.h
//  RCT
//
//  Created by DarkTango on 3/13/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __RCT__SIFTFeatureMatcher__
#define __RCT__SIFTFeatureMatcher__

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>


#include "Common.h"


/**
 Feature Matching Interface
 */
class IFeatureMatcher {
public:
    virtual void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches) = 0;
    virtual std::vector<cv::KeyPoint> GetImagePoints(int idx) = 0;
};

#pragma once

#define STRATEGY_USE_OPTICAL_FLOW		1
#define STRATEGY_USE_DENSE_OF			2
#define STRATEGY_USE_FEATURE_MATCH		4
#define STRATEGY_USE_HORIZ_DISPARITY	8

class IDistance {
public:
    virtual void OnlyMatchFeatures() = 0;
    virtual void RecoverDepthFromImages() = 0;
    virtual std::vector<cv::Point3d> getPointCloud() = 0;
    virtual const std::vector<cv::Vec3b>& getPointCloudRGB() = 0;
};

class RichFeatureMatcher : public IFeatureMatcher {
private:
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> extractor;
    
    std::vector<cv::Mat> descriptors;
    
    std::vector<cv::Mat>& imgs;
    std::vector<std::vector<cv::KeyPoint> >& imgpts;
public:
    //c'tor
    RichFeatureMatcher(std::vector<cv::Mat>& imgs,
                       std::vector<std::vector<cv::KeyPoint> >& imgpts);
    
    void MatchFeatures(int idx_i, int idx_j, std::vector<cv::DMatch>* matches = NULL);
    
    std::vector<cv::KeyPoint> GetImagePoints(int idx) { return imgpts[idx]; }
};


#endif /* defined(__RCT__SIFTFeatureMatcher__) */

