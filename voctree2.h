//
//  voctree2.h
//  RCT
//
//  Created by DarkTango on 3/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __RCT__voctree2__
#define __RCT__voctree2__

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/flann/flann_base.hpp>
#include <vector>
#include "ScenePoint.h"
#include "Frame.h"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

class node{
public:
    std::vector<int> descriptor_id;
    double weight;
    std::vector<node*> child;
    std::vector<int> frames;
};


class VocTree{
public:
    VocTree(const std::vector<Frame>& keyframes, const std::vector<ScenePoint>& scenepoints,const node& root):keyframes(keyframes),scenepoints(scenepoints){};
    node root;
    void dokmeans(node& _node, int branch);
    void construct(node& _node, int branch, int level);
    void init(int branch, int level);
    Mat alldescriptor;
    
    int getFrameIDbyDesID(int descriptor_id);
    void updateweight(node& _node);
    int spannedFrames(const node& _node);
    void candidateKeyframeSelection(const Frame& liveframe, std::vector<int>& candidateframe, int K);
    int featureinNodeandFrame(const node& _node, int frame_id);// the number of superior feature in keyframes[frame_id] that are clustered under _node.
    void frameinNode(const node& _node, std::vector<int>& frames);// calculate the spanned frames in _node.
    void updateloc();
    void cvtFrame(const Mat& img, Frame& fm);
    void matching(const std::vector<int>& candidateframe, Frame& onlineframe, std::vector<DMatch>& matches);
    
    void calibrate(const Frame& onlineframe, const std::vector<DMatch>& matches, Mat& rvec, Mat& tvec);
    void rendering(const Frame& onlineframe, const Mat& rvec, const Mat& tvec, Mat& outputimg);
    
private:
    std::vector<Frame> keyframes;
    std::vector<ScenePoint> scenepoints;
    std::vector<int> sizeofdescriptor;
    std::vector<int> rangeofdescriptor;
    FlannBasedMatcher matcher;
    cv::flann::Index kdtree;
    std::vector< std::vector<int> > loc; //loc[j][i] = k means descriptor i is in the kth node in level j.
};



struct box{
    int idx;
    int count;
    std::vector<int> descriptor;
    bool operator <(const box& thebox)const
    {
        return count < thebox.count;
    }
    bool operator >(const box& thebox)const
    {
        return count > thebox.count;
    }

};
#endif /* defined(__RCT__voctree2__) */
