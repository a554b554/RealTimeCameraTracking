//
//  ScenePoint.h
//  RCT
//
//  Created by DarkTango on 3/15/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __RCT__ScenePoint__
#define __RCT__ScenePoint__

#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <vector>
using namespace std;
using namespace cv;
class ScenePoint{
public:
    Point3f pt;
    vector<int> img;   // the index of frame that the pt visible in
    vector<int> feature; // the index of feature in each frame
    vector<Point2f> location; // the 2D position of the point in each frame
    Mat descriptor; // the SIFT feature descriptor of this point
    Point3d RGB;
    double saliency; // the saliency of this point.

};







#endif /* defined(__RCT__ScenePoint__) */
