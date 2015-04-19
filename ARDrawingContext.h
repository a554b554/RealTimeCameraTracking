//
//  ARDrawingContext.h
//  RCT
//
//  Created by DarkTango on 4/10/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __RCT__ARDrawingContext__
#define __RCT__ARDrawingContext__
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include "Frame.h"
#include <GLUT/GLUT.h>
class ARDrawingContext{
public:
    ARDrawingContext(const string windowname ,const Mat& intrinsic, const Mat& rvec, const Mat& tvec,const Frame& onlineframe);
    cv::Mat intrinsic;
    cv::Mat rvec;
    cv::Mat tvec;
    Frame onlineframe;
    void draw();
    void updateBackground();
private:
    void drawCameraFrame();
    void drawAugmentScene();
    void buildProjectionMatrix(int screen_width, int screen_height, GLfloat* projectionMat);
    void drawCoordinateAxis();
    void drawCubeModel();
    string ARWindowName;
    bool textureInitialized;
    unsigned int backgroundTextureID;
};
#endif /* defined(__RCT__ARDrawingContext__) */
