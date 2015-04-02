//
//  voctree2.cpp
//  RCT
//
//  Created by DarkTango on 3/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "voctree2.h"
#include <iostream>
#include <set>
#include <queue>
#include <opencv2/highgui/highgui.hpp>
void VocTree::construct(node& _node, int branch, int level){
    if (level == 0) {
        return;
    }
    //std::cout<<"constructing level: "<<level<<std::endl;
    dokmeans(_node, branch);
    for (int i = 0; i < _node.child.size(); i++) {
        construct(*_node.child[i], branch, level-1);
    }
    
}

void VocTree::init(int branch, int level){
    //build descriptor index vector
    sizeofdescriptor.push_back(0);
    for (int i = 0; i < keyframes.size(); i++) {
        Frame* curr = &keyframes[i];
        alldescriptor.push_back(curr->descriptor);
        sizeofdescriptor.push_back(curr->descriptor.rows);
    }
    
    int range = 0;
    for (int i = 0; i < sizeofdescriptor.size(); i++) {
        range += sizeofdescriptor[i];
        rangeofdescriptor.push_back(range);
    }
    
    //build root node
    for (int i = 0; i < alldescriptor.rows; i++) {
        root.descriptor_id.push_back(i);
    }
    int64 t0 = getTickCount();
    cout<<"constructing vocabulary tree..."<<endl;
    construct(root, branch, level);
    cout<<"construction complete!"<<endl;
    std::cout<<"time cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    cout<<"updating weight and frames in node";
    t0 = getTickCount();
    updateweight(root);
    cout<<"updating complete!"<<endl;
    std::cout<<"time cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    
    t0 = getTickCount();
    cout<<"updataloc..."<<endl;
    updateloc();
    cout<<"updating complete!"<<endl;
    std::cout<<"time cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    
    
    
    //build kdtree for search.
   /* cout<<"building kdtree..."<<endl;
    t0 = getTickCount();
    vector<Mat> mmm;
    mmm.push_back(alldescriptor);
    matcher.add(mmm);
    matcher.train();
    std::cout<<"time cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;*/
}



void VocTree::dokmeans(node& _node, int branch){
    Mat *currentDescriptor = new Mat();
    for (int i = 0 ; i < _node.descriptor_id.size(); i++) {
        currentDescriptor->push_back(alldescriptor.row(_node.descriptor_id[i]));
    }
    Mat *lables = new Mat();
    cv::TermCriteria term(CV_TERMCRIT_EPS,50,0.0001);
    if (_node.descriptor_id.size() < branch) {
        return;
    }
    cout<<"kmeaning constructing..."<<endl;
    int64 t0 = getTickCount();
    kmeans(*currentDescriptor, branch, *lables, term, 50, KMEANS_PP_CENTERS);
    cout<<"time: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    for (int i = 0; i < branch; i++) {
        node* childnode = new node();
        _node.child.push_back(childnode);
    }
    for (int i = 0; i < _node.descriptor_id.size(); i++) {
        int cluster = lables->at<int>(i,0);
        _node.child[cluster]->descriptor_id.push_back(_node.descriptor_id[i]);
    }
    delete currentDescriptor;
    delete lables;
    
}

int VocTree::getFrameIDbyDesID(int descriptor_id){
    for (int i = 0; i < rangeofdescriptor.size(); i++) {
        int lowerbound = rangeofdescriptor[i];
        int upperbound = rangeofdescriptor[i+1];
        if (lowerbound <= descriptor_id && descriptor_id < upperbound) {
            return i;
        }
    }
    return -1;
}

void VocTree::updateweight(node &_node){
    frameinNode(_node, _node.frames);
    _node.weight = log(keyframes.size()*1.0/_node.frames.size());
    for (int i = 0; i < _node.child.size(); i++) {
        updateweight(*_node.child[i]);
    }
}

int VocTree::spannedFrames(const node &_node){
    const int maxframesize = 150;
    bool flag[maxframesize];
    memset(flag, false, sizeof(bool)*maxframesize);
    for (int i = 0; i < _node.descriptor_id.size(); i++) {
        flag[getFrameIDbyDesID(_node.descriptor_id[i])] = true;
    }
    int count = 0;
    for (int i = 0; i < maxframesize; i++) {
        if (flag[i] == true) {
            count++;
        }
    }
    return count;
}

int VocTree::featureinNodeandFrame(const node &_node, int frame_id){
    int lowerbound = rangeofdescriptor[frame_id];
    int upperbound = rangeofdescriptor[frame_id+1];

    int count = 0;
    for (int i = 0; i < _node.descriptor_id.size(); i++) {
        int ind = _node.descriptor_id[i];
        if (lowerbound <= ind && ind < upperbound) {
            count++;
        }
    }
    return count;
}

const double MIN_WEIGHT = 0;
void VocTree::candidateKeyframeSelection(const Frame &liveframe, std::vector<int> &candidateframe, int K){
    vector<double> v_match(keyframes.size());
    std::fill(v_match.begin(), v_match.end(), 0);
    vector<DMatch> matches;
    int64 t0 = getTickCount();
    matcher.match(liveframe.descriptor, alldescriptor, matches);
    cout<<"time for matching: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    //t0 = getTickCount();
    for (int i = 0; i < liveframe.descriptor.rows; i++) {
        //const Mat& currDescriptor = liveframe.descriptor.row(i);
        std::queue<node*> nodequeue;
        nodequeue.push(&root);
        int level = 0;
        int64 t1 = getTickCount();
        while (!nodequeue.empty()) {
            int size = (int)nodequeue.size();
            node* minnode = nodequeue.front();
            bool founded = false;
            for (int j = 0; j < size; j++) {
                
                node* currNode = nodequeue.front();
                for (int k = 0; k < currNode->child.size(); k++) {
                    nodequeue.push(currNode->child[k]);
                }
                
                if (founded == true) {
                    nodequeue.pop();
                    continue;
                }
                
                //find most similar feature
                //std::vector<int> idx;
                //std::vector<float> neighbors,dist;
                //int Emax = INT_MAX;
                //kdtree.findNearest(currDescriptor, 1, Emax, idx, neighbors, dist);
                int index = matches[i].trainIdx;  // the most nearest feature id.
                 //find node contains that feature
            
                /*for (int i = 0; i < currNode->descriptor_id.size(); i++) {
                    if (currNode->descriptor_id[i] == index) {
                        minnode = currNode;
                        founded = true;
                        break;
                    }
                }*/
                if (j == loc[level][index]) {
                    minnode = currNode;
                    founded = true;
                }
                
                
                nodequeue.pop();
            }
            
            //update matching value.
            if (minnode->weight > MIN_WEIGHT) {
                for (int i = 0; i < minnode->frames.size(); i++) {
                    v_match[minnode->frames[i]] += featureinNodeandFrame(*minnode, minnode->frames[i])*(minnode->weight);
                }
            }
            level++;
        }
      //  cout<<"time for one loop: "<<(getTickCount() - t1)/getTickFrequency()<<endl;
    }
    cout<<"time cost for recognition"<<(getTickCount()-t0)/getTickFrequency()<<endl;
    //find most K relative frame;
    while (K--) {
        double max = 0;
        int idx = 0;
        for (int i = 0; i < v_match.size(); i++) {
            if (v_match[i] > max) {
                max = v_match[i];
                idx = i;
                v_match[i] = 0;
            }
        }
        candidateframe.push_back(idx);
    }
}

void VocTree::frameinNode(const node &_node, std::vector<int> &frames){
    frames.clear();
    std::set<int> frameid;
    for (int i = 0; i < _node.descriptor_id.size(); i++) {
        int fid = getFrameIDbyDesID(_node.descriptor_id[i]);
        frameid.insert(fid);
    }
    
    std::set<int>::iterator it;
    for (it = frameid.begin(); it != frameid.end(); it++) {
        frames.push_back(*it);
    }
}

void VocTree::updateloc(){
    std::queue<node*> nodequeue;
    nodequeue.push(&root);

    while (!nodequeue.empty()) {
        int size = (int)nodequeue.size();
        std::vector<int> vector_a(alldescriptor.rows);
        for (int j = 0; j < size; j++) {
            node* currNode = nodequeue.front();
            for (int k = 0; k < currNode->descriptor_id.size(); k++) {
                vector_a[currNode->descriptor_id[k]] = j;
            }
            for (int k = 0; k < currNode->child.size(); k++) {
                nodequeue.push(currNode->child[k]);
            }
            nodequeue.pop();
        }
        loc.push_back(vector_a);
    }
}

void VocTree::cvtFrame(const cv::Mat &img, Frame & fm){
    SiftFeatureDetector detector(500);
    SiftDescriptorExtractor extractor;
    img.copyTo(fm.img);
    detector.detect(img, fm.keypoint);
    extractor.compute(img, fm.keypoint, fm.descriptor);
}

void VocTree::matching(const std::vector<int> &candidateframe, Frame &onlineframe, std::vector<DMatch>& matches){
    std::vector<box> boxes(64);
    for (int i = 0; i < boxes.size(); i++) {
        boxes[i].idx = i;
        boxes[i].count = 0;
    }
    const int N = 70;
    //counting
    int xmax = onlineframe.img.cols;
    int ymax = onlineframe.img.rows;
    for (int i = 0; i < onlineframe.keypoint.size(); i++) {
        Point2f pt = onlineframe.keypoint[i].pt;
        int x_cor = (pt.x/xmax) * 8;
        int y_cor = (pt.y/ymax) * 8;
        int bid = x_cor + 8 * y_cor;
        boxes[bid].count++;
        boxes[bid].descriptor.push_back(i);
    }
    std::sort(boxes.begin(), boxes.end(), greater<box>());
    
    //step1: set c1(xj) and c2(xj) to 0.
    std::vector<int> c1(onlineframe.keypoint.size());
    std::vector<int> c2(onlineframe.keypoint.size());
    std::fill(c1.begin(), c1.end(), 0);
    std::fill(c2.begin(), c2.end(), 0);
    
    
    //step2: perform first-pass matching.
    for (int i = 0; i < boxes.size(); i++) {
        box* currBox = &boxes[i];

        for (int j = 0; j < currBox->descriptor.size(); j++) {

            int desID = currBox->descriptor[j];
            if (c1[desID] == 0) {//for each unmatched feature xj in Bi and c1(xj)=0
                c1[desID] = 1;
                
                vector<float> queryFeature;
                Mat Feature = onlineframe.descriptor.row(desID);
                for (int k = 0; k < Feature.cols; k++) {
                    queryFeature.push_back(Feature.at<float>(0,k));
                }
                //for each candidate frame.
                for (int k = 0; k < candidateframe.size(); k++) {
                    vector<int> indices(10);
                    vector<float> dists(10);
                    //find the 10 features from keyframe k that are most similar with xj.
                    keyframes[candidateframe[k]].kdtree.knnSearch(queryFeature, indices, dists, 10, cv::flann::SearchParams(64));
                    if (dists[0]/dists[1] < 0.7) {
                        // if satisfied with 2NN heuristic, stop the matching of Bi.
                        DMatch match;
                        match.queryIdx = desID;
                        match.trainIdx = indices[0];
                        match.distance = dists[0];
                        match.imgIdx = candidateframe[k];
                        matches.push_back(match);
                        break;
                    }
                }
            }
        }
    }
    std::vector<Mat> fundamentalmatrixes(candidateframe.size());
    
    
}