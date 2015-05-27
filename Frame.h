//
//  Frame.h
//  RCT
//
//  Created by DarkTango on 3/15/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __RCT__Frame__
#define __RCT__Frame__

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <map>
#include <vector>
#include <ext/hash_map>
using namespace std;
using namespace cv;
class Frame{
public:
    Frame(){};
    Mat img; // the origin image data
    Mat dogimg; //store the DoG img
    vector<KeyPoint> keypoint; // the keypoint in the frame
    vector<Point2f> pos;
    vector<Point3f> pos3d;
    Mat descriptor; // the feature descriptor for each keypoint
    double F; // focal length
    double k; // radial distortion coeffs
    vector<double> quanternions; // a WXYZ number denotes the rotation
    Point3f location ; // center of the camera
    vector<int> scenepoint; // the index of all scenepoint in this frame
    vector<int> featuredensity; // the density of each keypoint
    cv::flann::Index kdtree;
    FlannBasedMatcher* d_matcher;
    string filename;
private:
};




#endif /* defined(__RCT__Frame__) */
