//
//  VocabularyTree.h
//  RCT
//
//  Created by DarkTango on 3/18/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//
#pragma once
#ifndef __RCT__VocabularyTree__
#define __RCT__VocabularyTree__

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include "ScenePoint.h"
#include "Frame.h"

class treenode{
public:
    std::vector<int> scenepoint;
    std::vector<treenode*> child;
    std::vector<int> img;
    double weight;
    
    
};

class VocabularyTree{
public:
    
    
    VocabularyTree(const std::vector<ScenePoint>& globalscenepoint,const std::vector<Frame>& globalframe,const std::vector<int>& keyframes);
    treenode root;
    std::vector<ScenePoint> globalscenepoint;
    std::vector<Frame> globalframe;
    std::vector<int> keyframes;
    
    void construction(int b, int L, treenode& node); // b is the cluster number of k-mean, L is tree level
    void findCandidateFrame(const Frame& inputliveframe, std::vector<int>& outputframe, int K); // K is the number of candidate frame.
    
    
    void liveMat2Frame(const cv::Mat& inputlivemat, Frame& outputframe);
    
    void init(int b, int L);
    bool iskeyframe(int _id);
    
    
    void mainloop(); // the main loop for whole process.
    
    
    int spannedkeyframe(const treenode& node) const; //useless
    int featureinframe(const Frame& frame, const treenode& node);// the number of superior features in frame that are clustered under node.
    
private:
    void kmeans(treenode& node, int b);
    void updateweight(treenode& node);
    void updateimg(treenode& node);
    double minDistInNode(const treenode& node, const Mat& descriptor);
    
   // void updateweightfornode(treenode& node, int L);
};






#endif /* defined(__RCT__VocabularyTree__) */
