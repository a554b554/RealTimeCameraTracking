//
//  IDistance.h
//  RCT
//
//  Created by DarkTango on 3/13/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef RCT_IDistance_h
#define RCT_IDistance_h

#pragma once

#define STRATEGY_USE_OPTICAL_FLOW		1
#define STRATEGY_USE_DENSE_OF			2
#define STRATEGY_USE_FEATURE_MATCH		4
#define STRATEGY_USE_HORIZ_DISPARITY	8

class IDistance {
public:
    virtual void OnlyMatchFeatures();
    virtual void RecoverDepthFromImages();
    virtual std::vector<cv::Point3d> getPointCloud();
    virtual const std::vector<cv::Vec3b>& getPointCloudRGB();
};
#endif
