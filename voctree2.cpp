//
//  voctree2.cpp
//  RCT
//
//  Created by DarkTango on 3/28/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "voctree2.h"
#include "load.h"
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
    for (int i = 0; i < keyframes.size(); i++) {
        Frame* curr = &keyframes[i];
        alldescriptor.push_back(curr->descriptor);
        for (int j = 0; j < curr->descriptor.rows; j++) {
            frameid.push_back(i);
        }
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
    
  
    
    //build kdtree for search.
    /*cout<<"building kdtree..."<<endl;
    t0 = getTickCount();
    vector<Mat> mmm;
    mmm.push_back(alldescriptor);
    matcher.add(mmm);
    matcher.train();*/
    
}

int VocTree::framesinNode(const node &_node){
    std::set<int> _set;
    for (int i = 0; i < _node.descriptor_id.size(); i++) {
        _set.insert(frameid[_node.descriptor_id[i]]);
    }
    return (int)_set.size();
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
    //cout<<"kmeaning constructing..."<<endl;
    //int64 t0 = getTickCount();
    kmeans(*currentDescriptor, branch, *lables, term, 50, KMEANS_PP_CENTERS);
    //cout<<"time: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
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



void VocTree::updateweight(node &_node){
    _node.weight = log(keyframes.size()*1.0/framesinNode(_node));
    for (int i = 0; i < _node.child.size(); i++) {
        updateweight(*_node.child[i]);
    }
    //update mean descriptor for the node.
    _node.des.create(1, 128, CV_32F);
    //std::vector<Mat> _des(_node.descriptor_id.size());
    for (int i = 0; i < 128; i++) {
        _node.des.at<float>(0,i) = 0;
    }
    for (int i = 0; i < _node.descriptor_id.size(); i++) {
        _node.des += alldescriptor.row(_node.descriptor_id[i]);
    }
    _node.des = _node.des/(_node.descriptor_id.size()*1.0);
}


const double MIN_WEIGHT = 0;
void VocTree::candidateKeyframeSelection(const Frame &liveframe, std::vector<int> &candidateframe, int K){
    vector<double> v_match(keyframes.size());
    std::fill(v_match.begin(), v_match.end(), 0);
    vector<DMatch> matches;
    int64 t0 = getTickCount();
    kd_matcher->match(liveframe.descriptor, matches);
    //matcher.match(liveframe.descriptor, alldescriptor, matches);
    //vector<int> index;
    //vector<float> dist;
    //kdtree.knnSearch(liveframe.descriptor, index, dist, 1);
    //cout<<"time for matching: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
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
                              nodequeue.pop();
            }
            //update matching value.
            if (minnode->weight > MIN_WEIGHT) {
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

void VocTree::candidateKeyframeSelection2(const Frame& liveframe, std::vector<int>& candidateframe, int K){
    vector<double> v_match(keyframes.size());
    std::fill(v_match.begin(), v_match.end(), 0);
    std::queue<node*> nodequeue;
    nodequeue.push(&root);
    int64 t0 = getTickCount();
    for (int i = 0; i < liveframe.keypoint.size(); i++) {
        while (!nodequeue.empty()) {
            int size = (int)nodequeue.size();
            const node* minnode = nodequeue.front();
            double mindist = norm(liveframe.descriptor.row(i), minnode->des);
            for (int j = 0; j < size; j++) {
                const node* currNode = nodequeue.front();
                for (int k = 0; k < currNode->child.size(); k++) {
                    nodequeue.push(currNode->child[k]);
                }
                //find most similar feature
                double tmpdist = norm(liveframe.descriptor.row(i), currNode->des);
                if (tmpdist < mindist) {
                    minnode = currNode;
                    mindist = tmpdist;
                }
                nodequeue.pop();
            }
            //update matching value.
            if (minnode->weight > MIN_WEIGHT) {
                for (int i = 0; i < minnode->descriptor_id.size(); i++) {
                    v_match[frameid[minnode->descriptor_id[i]]] += minnode->weight;
                }
            }
        }
    }
    //find most K relative frame;
    while (K--) {
        double max = -1;
        int idx = 0;
        for (int i = 0; i < v_match.size(); i++) {
            if (v_match[i] > max) {
                max = v_match[i];
                idx = i;
            }
        }
        v_match[idx] = -1;
        candidateframe.push_back(idx);
    }
    cout<<"time for keyframe recognition: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
}


bool kptsort(const KeyPoint& a, const KeyPoint& b){
    return a.response > b.response;
}

void VocTree::cvtFrame(const cv::Mat &img, Frame& fm){
    int64 t0 = getTickCount();
    SiftFeatureDetector detector;
    SiftDescriptorExtractor extractor;
    img.copyTo(fm.img);
    detector.detect(img, fm.keypoint);
    std::sort(fm.keypoint.begin(), fm.keypoint.end(), kptsort);
    extractor.compute(img, fm.keypoint, fm.descriptor);
    cout<<"time for feature extraction: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
}

void drawmatchedfeature(const Frame& fm, const std::vector<DMatch>& matches, Mat& outputimg){
    std::vector<KeyPoint> kpt;
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i].imgIdx != -1) {
            kpt.push_back(fm.keypoint[matches[i].queryIdx]);
        }
    }
    drawKeypoints(fm.img, kpt, outputimg);
    imshow(to_string(rand()+400), outputimg);
}

float distP2L(Point2f pt, Point3f line){
    return fabs(pt.x*line.x+pt.y*line.y+line.z)/sqrt(line.x*line.x+line.y*line.y);
}

bool VocTree::twoPassMatching(const std::vector<int> &candidateframe, Frame &onlineframe, std::vector<std::vector<DMatch>>& matches){
    std::vector<Frame> key_frame;
    for (int i = 0; i < candidateframe.size(); i++) {
        key_frame.push_back(keyframes[candidateframe[i]]);
    }
    return Matching(key_frame, onlineframe, matches);
    /*const int minNumofTrack = 35;
    std::vector<box> boxes(64);
    int c_size = (int)candidateframe.size();
    for (int i = 0; i < boxes.size(); i++) {
        boxes[i].idx = i;
        boxes[i].count = 0;
    }
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
    
    while (1) {
        //step2: perform first-pass matching.
        for (int i = 0; i < boxes.size(); i++) {
            for (int j = 0; j < boxes[i].descriptor.size(); j++) {
                int desID = boxes[i].descriptor[j];
                if (c1[desID] == 0) {
                    c1[desID] = 1;
                    bool goodmatch = false;
                    for (int k = 0; k < c_size; k++) {
                        const Frame* k_frame = &keyframes[candidateframe[k]];
                        goodmatch = featurematch(*k_frame, onlineframe, desID, matches[k]);
                        if (goodmatch) {
                            break;
                        }
                    }
                    if (goodmatch) { //if xj satisfies the 2NN heuristic, stop matching of Bi.
                        break;
                    }
                }
            }
        }
        //step3: estimate fundamental
        std::vector<Mat> fundamental(c_size);
        for (int i = 0; i < c_size; i++) {
            const Frame* k_frame = &keyframes[candidateframe[i]];
            refinewithFundamental(*k_frame, onlineframe, matches[i], fundamental[i]);
        }
        if (matchsize(matches) > minNumofTrack) { //if there are already N inlier
            break;
        }
        for (int i = 0; i < c_size; i++) {
            //     showmatches(keyframes[candidateframe[i]], onlineframe, matches[i]);
        }
        drawmatchedpoint(onlineframe, matches);
        cout<<"first match: "<<matchsize(matches)<<endl;
        // waitKey(0);
        
        //step4: perform second-pass matching
        for (int i = 0; i < boxes.size(); i++) {
            for (int j = 0; j < boxes[i].descriptor.size(); j++) {
                int desID = boxes[i].descriptor[j];
                if (c1[desID] == 1 && c2[desID] == 0) {
                    c2[desID] = 1;
                    bool goodmatch = false;
                    for (int k = 0; k < c_size; k++) {
                        if (!fundamental[k].empty()) {
                            goodmatch = epipolarmatch(keyframes[candidateframe[k]], onlineframe, desID, fundamental[k], matches[k]);
                        }
                        if (goodmatch) {
                            break;
                        }
                    }
                    if (goodmatch) {
                        break;
                    }
                }
            }
        }
        for (int i = 0; i < c_size; i++) {
            //showmatches(keyframes[candidateframe[i]], onlineframe, matches[i]);
        }
        cout<<"second match: "<<matchsize(matches)<<endl;
        if (matchsize(matches) > minNumofTrack) {
            break;
        }
        bool breakflag = true;
        for (int i = 0; i < onlineframe.keypoint.size(); i++) {
            if (c1[i] == 0||c2[i] == 0) {
                breakflag = false;
            }
        }
        if (breakflag) {
            break;
        }
    }
    for (int i = 0; i < c_size; i++) {
        showmatches(keyframes[candidateframe[i]], onlineframe, matches[i]);
    }
    return matchsize(matches) > minNumofTrack;*/
}

bool VocTree::matchWithOnlinepool(Frame& onlineframe, std::vector<std::vector<DMatch>>& poolMatches){
    return Matching(onlinepool, onlineframe, poolMatches);
}

bool VocTree::Matching(const std::vector<Frame>& candidateframe, Frame& onlineframe, std::vector<std::vector<DMatch>>& matches){
    const int minNumofTrack = 35;
    std::vector<box> boxes(64);
    int c_size = (int)candidateframe.size();
    for (int i = 0; i < boxes.size(); i++) {
        boxes[i].idx = i;
        boxes[i].count = 0;
    }
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
    
    while (1) {
        //step2: perform first-pass matching.
        for (int i = 0; i < boxes.size(); i++) {
            for (int j = 0; j < boxes[i].descriptor.size(); j++) {
                int desID = boxes[i].descriptor[j];
                if (c1[desID] == 0) {
                    c1[desID] = 1;
                    bool goodmatch = false;
                    for (int k = 0; k < c_size; k++) {
                        const Frame* k_frame = &candidateframe[k];
                        goodmatch = featurematch(*k_frame, onlineframe, desID, matches[k]);
                        if (goodmatch) {
                            break;
                        }
                    }
                    if (goodmatch) { //if xj satisfies the 2NN heuristic, stop matching of Bi.
                        break;
                    }
                }
            }
        }
        //step3: estimate fundamental
        std::vector<Mat> fundamental(c_size);
        for (int i = 0; i < c_size; i++) {
            const Frame* k_frame = &candidateframe[i];
            refinewithFundamental(*k_frame, onlineframe, matches[i], fundamental[i]);
        }
        if (matchsize(matches) > minNumofTrack) { //if there are already N inlier
            break;
        }
        for (int i = 0; i < c_size; i++) {
            //     showmatches(keyframes[candidateframe[i]], onlineframe, matches[i]);
        }
        //cout<<"first match: "<<matchsize(matches)<<endl;
        // waitKey(0);
        
        //step4: perform second-pass matching
        for (int i = 0; i < boxes.size(); i++) {
            for (int j = 0; j < boxes[i].descriptor.size(); j++) {
                int desID = boxes[i].descriptor[j];
                if (c1[desID] == 1 && c2[desID] == 0) {
                    c2[desID] = 1;
                    bool goodmatch = false;
                    for (int k = 0; k < c_size; k++) {
                        if (!fundamental[k].empty()) {
                            goodmatch = epipolarmatch(candidateframe[k], onlineframe, desID, fundamental[k], matches[k]);
                        }
                        if (goodmatch) {
                            break;
                        }
                    }
                    if (goodmatch) {
                        break;
                    }
                }
            }
        }
        for (int i = 0; i < c_size; i++) {
            //showmatches(candidateframe[i], onlineframe, matches[i]);
        }
        //cout<<"second match: "<<matchsize(matches)<<endl;
        if (matchsize(matches) > minNumofTrack) {
            break;
        }
        bool breakflag = true;
        for (int i = 0; i < onlineframe.keypoint.size(); i++) {
            if (c1[i] == 0||c2[i] == 0) {
                breakflag = false;
            }
        }
        if (breakflag) {
            break;
        }
    }
    for (int i = 0; i < c_size; i++) {
       // showmatches(candidateframe[i], onlineframe, matches[i]);
    }
    cout<<"matched: "<<matchsize(matches)<<endl;
    return matchsize(matches) > minNumofTrack;
}

bool VocTree::epipolarmatch(const Frame &keyframe, const Frame &onlineframe, const int featureID, const Mat& fundamental,std::vector<DMatch> &m_match){
    std::vector<std::vector<DMatch>> tmp;
    keyframe.d_matcher->knnMatch(onlineframe.descriptor.row(featureID), tmp, 10);
    std::vector<Point2f> points(1);
    std::vector<Point3f> line(1);
    points[0] = onlineframe.keypoint[featureID].pt;
    computeCorrespondEpilines(points, 2, fundamental, line);
    std::vector<DMatch> goodmatches;
    Point3f _l = line[0];
    for (int i = 0; i < tmp[0].size(); i++) {
        Point2f pt = onlineframe.keypoint[tmp[0][i].trainIdx].pt;
        if (_l.x*pt.x+_l.y*pt.y+_l.z <= 2) {
            goodmatches.push_back(tmp[0][i]);
        }
        if (goodmatches.size() >= 2) {
            break;
        }
    }
    if (goodmatches.size() < 2) {
        return false;
    }
    DMatch bestmatch = goodmatches[0];
    DMatch bettermatch = goodmatches[1];
    if (bestmatch.distance/bettermatch.distance < 0.7 && bestmatch.distance < 100) {
        bestmatch.queryIdx = featureID;
        m_match.push_back(bestmatch);
        return true;
    }
    return false;
}

bool VocTree::featurematch(const Frame &keyframe, const Frame &onlineframe, const int featureID, std::vector<DMatch> &m_match){
    vector<vector<DMatch>> tmp;
    keyframe.d_matcher->knnMatch(onlineframe.descriptor.row(featureID), tmp, 2);
    DMatch bestmatch = tmp[0][0];
    DMatch bettermatch = tmp[0][1];
    const float maxRatio = 0.7;
    if (bestmatch.distance/bettermatch.distance < maxRatio) {
        bestmatch.queryIdx = featureID;
        m_match.push_back(bestmatch);
        return true;
    }
    return false;
}

int VocTree::matchsize(const std::vector<std::vector<DMatch> > &matches){
    std::set<int> _set;
    for (int i = 0; i < matches.size(); i++) {
        for (int j = 0; j < matches[i].size(); j++) {
            _set.insert(matches[i][j].queryIdx);
        }
    }
    return (int)_set.size();
}

bool VocTree::refinewithFundamental(const Frame &keyframe, const Frame &onlineframe, std::vector<DMatch> &m_match, Mat& fundamental){
    const int minNumberAllow = 7;
    if (m_match.size() < minNumberAllow) {
        m_match.clear();
        return false;
    }
    std::vector<Point2f> srcPt(m_match.size());
    std::vector<Point2f> dstPt(m_match.size());
    for (size_t i = 0; i < m_match.size(); i++) {
        srcPt[i] = keyframe.keypoint[m_match[i].trainIdx].pt;
        dstPt[i] = onlineframe.keypoint[m_match[i].queryIdx].pt;
    }
    std::vector<uchar> inlierMask(srcPt.size());
    fundamental = findFundamentalMat(srcPt, dstPt, FM_RANSAC, 1.0 , 0.99, inlierMask);
    std::vector<DMatch> inliers;
    for (size_t i = 0; i < inlierMask.size(); i++) {
        if (inlierMask[i]) {
            inliers.push_back(m_match[i]);
        }
    }
    m_match.swap(inliers);
    return m_match.size() > minNumberAllow;
}

void VocTree::updatematchingInfo(const std::vector<int> candidateframe, Frame&onlineframe, std::vector<std::vector<DMatch>> &matches){
    for (int i = 0; i < matches.size(); i++) {
        for (int j = 0; j < matches[i].size(); j++) {
            onlineframe.pos3d.push_back(keyframes[candidateframe[i]].pos3d[matches[i][j].trainIdx]);
            onlineframe.pos.push_back(onlineframe.keypoint[matches[i][j].queryIdx].pt);
        }
    }
    computekeypoint(onlineframe, onlineframe.pos);
    SiftDescriptorExtractor ex;
    ex.compute(onlineframe.img, onlineframe.keypoint, onlineframe.descriptor);
}




bool VocTree::calibrate(const Frame &onlineframe, const std::vector<std::vector<DMatch>> &matches,const std::vector<int>& candidateframe, const std::vector<std::vector<DMatch>>& poolmatches, cv::Mat& rvec, cv::Mat&tvec){
    int64 t0 = getTickCount();
    vector<Point3f> objpoints;
    vector<Point2f> imgpoints;

    std::set<int> _set;
    for (int i = 0; i < matches.size(); i++) {
        for (int j = 0; j < matches[i].size(); j++) {
            if (keyframes[candidateframe[i]].keypoint[matches[i][j].trainIdx].class_id != 0) {
                objpoints.push_back(keyframes[candidateframe[i]].pos3d[matches[i][j].trainIdx]);
                //imgpoints.push_back(onlineframe.pos[matches[i][j].queryIdx]);
                imgpoints.push_back(onlineframe.keypoint[matches[i][j].queryIdx].pt);
              //  _set.insert(matches[i][j].queryIdx);
            }
        }
    }
    for (int i = 0; i < poolmatches.size(); i++) {
        for (int j = 0; j < poolmatches[i].size(); j++) {
            objpoints.push_back(onlinepool[i].pos3d[poolmatches[i][j].trainIdx]);
            imgpoints.push_back(onlineframe.keypoint[poolmatches[i][j].queryIdx].pt);
            //_set.insert(matches[i][j].queryIdx);
        }
    }
    if (objpoints.size()<4) {
        return false;
    }
    drawmatchedpoint(onlineframe, matches, poolmatches);
    cout<<"pattern size: "<<_set.size()<<endl;
    //solvePnP(objpoints, imgpoints, intrinsic, distCoeffs, rvec, tvec);
    solvePnPRansac(objpoints, imgpoints, intrinsic, distCoeffs, rvec, tvec);
    //calibrateCamera(objpoints, imgpoints, onlineframe.img.size(), cameraMat, distCoeffs, rvec, tvec);
    cout<<"time for solvePNP: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    return _set.size() >= 70;
}

void VocTree::draw(const string windowname,const Frame &onlineframe, const cv::Mat &rvec, const cv::Mat &tvec, cv::Mat &outputimg){
    ARDrawingContext arctx(windowname, intrinsic, rvec, tvec, onlineframe);
    arctx.draw();
    waitKey(0);
}

Point3f muti(Point3f vecba, Point3f vecbc){
    return Point3f(vecba.y*vecbc.z-vecbc.y*vecba.z,vecba.z*vecbc.x-vecbc.z*vecba.x,vecba.x*vecbc.y-vecbc.x*vecba.y);
}
void VocTree::rendering(const Frame &onlineframe, const cv::Mat &rvec, const cv::Mat &tvec, cv::Mat &outputimg){

    int64 t0 = getTickCount();
    onlineframe.img.copyTo(outputimg);
    std::vector<Point3f> pt3d;
    
    std::vector<Point2f> origin2d;
    std::vector<Point3f> origin3d;
    for (int i = 0; i < onlineframe.pos3d.size(); i++) {
        
    }
    
    for (int i = 0; i < scenepoints.size(); i++) {
        pt3d.push_back(scenepoints[i].pt);
    }
    std::vector<Point2f> pt2d;
    if (!rvec.empty()&&!tvec.empty()) {
        projectPoints(pt3d, rvec, tvec, intrinsic, distCoeffs, pt2d);
    }
    int myradius=1;
    for (int i=0;i<pt2d.size();i++){
        if (i%1 == 0) {
            circle(outputimg,cvPoint(pt2d[i].x,pt2d[i].y),myradius,CV_RGB(255,255,0),-1,8,0);
        }
    }
   /* for (int i = 0; i < kpt2d.size(); i++) {
        for (int j = 0; j < kpt2d.size(); j++) {
            if (i != j) {
                line(outputimg, kpt2d[i], kpt2d[j], CV_RGB(255, 0, 255));
            }
        }
    }*/
    /*std::vector<Point3f> axis;
    findChessboardCorners(keyframes[0].img, Size(3,3), axis);
    std::vector<Point2f> axis2d;
    for (int i = 0; i < axis.size(); i++) {
        projectPoints(axis, rvec, tvec, intrinsic, distCoeffs, axis2d);
    }
    for (int i = 1; i <axis2d.size(); i++) {
        line(outputimg, axis2d[0], axis2d[i], CV_RGB(255*(i==1), 255*(i==2), 255*(i==3)),5);
    }*/
    
    cout<<"time for rendering: "<<(getTickCount() - t0)/getTickFrequency()<<endl;
    imshow("show", outputimg);
    //waitKey(0);
}

void VocTree::loadCameraMatrix(const string basepath){
    FileStorage fs(basepath+"/out_camera_data.xml", FileStorage::READ);
    fs["Camera_Matrix"] >> intrinsic;
    fs["Distortion_Coefficients"] >> distCoeffs;
    fs["square_Size"] >> squareSize;
    cout<<"intrinsic parameter: "<<intrinsic<<endl;
    cout<<"distortion coeffs: "<<distCoeffs<<endl;
    cout<<"square size: "<<squareSize<<endl;
    fs.release();
}

void VocTree::showmatch(const std::vector<int> &candidateframe, const string windowname, const Frame &onlineframe, const std::vector<std::vector<DMatch>>& matches){
    for (int i = 0; i < candidateframe.size(); i++) {
        Mat out;
        drawMatches(onlineframe.img, onlineframe.keypoint, keyframes[candidateframe[i]].img, keyframes[candidateframe[i]].keypoint, matches[i], out);
        imshow(windowname, out);
        waitKey(0);
    }
}

bool VocTree::ordinarymatching(const std::vector<int> &candidateframe, Frame&onlineframe, std::vector<std::vector<DMatch>>& matches){
    int64 tt = getTickCount();
    for (int i = 0; i < candidateframe.size(); i++) {
        Frame* currFrame = &keyframes[candidateframe[i]];
        vector<vector<DMatch>> m_matches;
        //currFrame->d_matcher->knnMatch(onlineframe.descriptor, currFrame->descriptor, m_matches, 2);
        currFrame->d_matcher->knnMatch(onlineframe.descriptor, m_matches, 2);
        for (int j = 0; j < m_matches.size(); j++) {
            const DMatch& bestMatch = m_matches[j][0];
            const DMatch& betterMatch = m_matches[j][1];
            
            float distRatio = bestMatch.distance/betterMatch.distance;
            if (distRatio < 0.7) {
                matches[i].push_back(bestMatch);
            }
        }
    }
   
   
    //showmatch(candidateframe, "first", onlineframe, matches);
    std::vector<Mat> F(candidateframe.size());
    for (int i = 0; i < candidateframe.size(); i++) {
        refinewithFundamental(keyframes[candidateframe[i]], onlineframe, matches[i], F[i]);
    }
    Mat tmp;
    std::set<int> _set;
    tmp = onlineframe.img.clone();
    for (int i = 0; i < matches.size(); i++) {
        for (int j = 0; j < matches[i].size(); j++) {
            circle(tmp, onlineframe.keypoint[matches[i][j].queryIdx].pt, 4, CV_RGB(0, 255, 0),-1,8,0);
            _set.insert(matches[i][j].queryIdx);
        }
    }
     cout<<"time for feature matching: "<<(getTickCount()-tt)/getTickFrequency()<<"size: "<<_set.size()<<endl;
    imshow("matched", tmp);
  //  showmatch(candidateframe, "final",onlineframe, matches);
    return _set.size()>=40;
}

bool VocTree::refineMatchesWithHomography(const std::vector<int> &candidateframe, Frame &onlineframe, std::vector<std::vector<DMatch>> &matches, std::vector<Mat>& Fundamental){
    const int minNumberAllowed = 8;
    for (int i = 0; i < candidateframe.size(); i++) {
        /*if (matches[i].size() < minNumberAllowed) {
            matches[i].clear();
            continue;
        }*/ 
        std::vector<Point2f> srcpt(matches[i].size());
        std::vector<Point2f> dstpt(matches[i].size());
        for (int j = 0; j < matches[i].size(); j++) {
            srcpt[j] = keyframes[candidateframe[i]].keypoint[matches[i][j].trainIdx].pt;
            dstpt[j] = onlineframe.keypoint[matches[i][j].queryIdx].pt;
        }
        //find homography matrix and get inliers mask.
        std::vector<uchar> inliersMask(srcpt.size());
        std::vector<Mat> homo(candidateframe.size());
        std::vector<Mat> show(candidateframe.size());
        if (srcpt.size()>=4) {
            //findHomography(srcpt, dstpt,CV_FM_RANSAC, 1,inliersMask);
            Fundamental[i] = findFundamentalMat(srcpt, dstpt, FM_RANSAC, 1.0 , 0.99, inliersMask);
           /* Mat out;
            onlineframe.img.copyTo(out);
            std::vector<Point3f> line;
            computeCorrespondEpilines(srcpt, 1, Fundamental[i], line);
            for (int q = 0; q < srcpt.size(); q++) {
                cv::line(out, cv::Point(0,-line[q].z/line[q].y), cv::Point(out.cols,-(line[i].z+line[i].x*out.cols)), CV_RGB(255, 0, 0));
            }
            imshow("epipline", out);
            waitKey(0);*/
            
            /*homo[i] = findHomography(srcpt, dstpt,CV_FM_RANSAC, 0.5,inliersMask);
            warpPerspective(onlineframe.img, show[i], homo[i], onlineframe.img.size());
            imshow("warp", show[i]);
            waitKey(0);*/
        }
        
        std::vector<DMatch> inliers;
        for (int j = 0; j < inliersMask.size(); j++) {
            if (inliersMask[j]) {
                inliers.push_back(matches[i][j]);
            }
        }
        matches[i].swap(inliers);
        cout<<"matches "<<i<<" remain "<<matches[i].size()<<" inliers."<<endl;
    }
    return true;
}

void VocTree::updateonlinepool(const std::vector<int> &candidateframe, const std::vector<std::vector<DMatch>> &matches, const std::vector<std::vector<DMatch>> &poolmatches, Frame &onlineframe){
    
    
    //convert keypoint
    std::vector<KeyPoint> tmpkpt;
    for (int i = 0; i < matches.size(); i++) {
        for (int j = 0; j < matches[i].size(); j++) {
            Point3f pos3d = keyframes[candidateframe[i]].pos3d[matches[i][j].trainIdx];
            onlineframe.pos3d.push_back(pos3d);
            tmpkpt.push_back(onlineframe.keypoint[matches[i][j].queryIdx]);
        }
    }
    for (int i = 0; i < poolmatches.size(); i++) {
        for (int j = 0; j < poolmatches[i].size(); j++) {
            Point3f pos3d = onlinepool[i].pos3d[poolmatches[i][j].trainIdx];
            onlineframe.pos3d.push_back(pos3d);
            tmpkpt.push_back(onlineframe.keypoint[poolmatches[i][j].queryIdx]);
        }
    }
    onlineframe.keypoint.swap(tmpkpt);

    //train descriptor
    SiftDescriptorExtractor ex;
    ex.compute(onlineframe.img, onlineframe.keypoint, onlineframe.descriptor);
    onlineframe.d_matcher = new FlannBasedMatcher();
    std::vector<Mat> des(1);
    des[0] = onlineframe.descriptor.clone();
    onlineframe.d_matcher->add(des);
    onlineframe.d_matcher->train();
    
    std::vector<std::vector<DMatch>> countmat;
    for (int i = 0; i < poolmatches.size(); i++) {
        countmat.push_back(poolmatches[i]);
    }
    for (int i = 0; i < matches.size(); i++) {
        countmat.push_back(matches[i]);
    }
    if (matchsize(countmat) < 50) {
        return;
    }
    
    onlinepool.push_back(onlineframe);
    while (onlinepool.size()>4) {
        onlinepool.erase(onlinepool.begin());
    }
    return;
}

int VocTree::framesize(){
    return (int)keyframes.size();
}













