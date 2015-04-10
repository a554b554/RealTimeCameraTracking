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
    
    cv::flann::KDTreeIndexParams indexParams(5);
    kdtree.build(alldescriptor, indexParams);
    
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

bool kptsort(const KeyPoint& a, const KeyPoint& b){
    return a.response > b.response;
}

void VocTree::cvtFrame(const cv::Mat &img, Frame& fm){
    SiftFeatureDetector detector(1000);
    SiftDescriptorExtractor extractor;
    img.copyTo(fm.img);
    detector.detect(img, fm.keypoint);
    std::sort(fm.keypoint.begin(), fm.keypoint.end(), kptsort);
    extractor.compute(img, fm.keypoint, fm.descriptor);
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

void VocTree::matching(const std::vector<int> &candidateframe, Frame &onlineframe, std::vector<std::vector<DMatch>>& matches){
    //sort keypoint by DoG strength
    //std::sort(onlineframe.keypoint.begin(), onlineframe.keypoint.end(), kptsort);
    std::vector<box> boxes(64);
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
    std::vector<int> matchedbox(boxes.size());
    std::fill(c1.begin(), c1.end(), 0);
    std::fill(c2.begin(), c2.end(), 0);
    std::fill(matchedbox.begin(), matchedbox.end(), 0);
    
    //step2: perform feature matching.
    std::vector<std::vector<std::vector<DMatch>>> m_result;
    for (int i = 0; i < candidateframe.size(); i++) {
        std::vector<std::vector<DMatch>> m_matches;
        FlannBasedMatcher mm;
        Frame* currFrame = &keyframes[candidateframe[i]];
        mm.knnMatch(onlineframe.descriptor, currFrame->descriptor, m_matches, 10);
        m_result.push_back(m_matches);
    }
    
    const int minNumberofMatches = 50;
    int64 tt = getTickCount();
    while(1){
        //step 3: perform the first-pass match.
        for (int i = 0; i < boxes.size(); i++) {
            for (int j = 0; j < boxes[i].descriptor.size(); j++) {
                bool flag = false;
                int desID = boxes[i].descriptor[j];
                if (c1[desID] == 0) {
                    c1[desID] = 1;
                    for (int k = 0; k < candidateframe.size(); k++) {
                        DMatch bestmatch = m_result[k][desID][0];
                        DMatch bettermatch = m_result[k][desID][1];
                        float distRatio = bestmatch.distance/bettermatch.distance;
                        if (distRatio < 0.7) {
                            matches[k].push_back(bestmatch);
                            flag = true;
                            break;
                        }
                    }
                }
                if (flag) { //if find match, stop the matching of Bi.
                    break;
                }
            }
        }
      
       
        //showmatch(candidateframe, onlineframe, matches);
        //step4: use fundamental matrix to remove outlier.
        std::vector<Mat> fundamental(candidateframe.size());
        refineMatchesWithHomography(candidateframe, onlineframe, matches, fundamental);
       //if there are already N inliers.
        int ans = 0;
        for (int i = 0; i < matches.size(); i++) {
            ans += matches[i].size();
        }
        cout<<"first match: "<<ans<<endl;
        if (ans >= minNumberofMatches) {
            break; //stop matching.
        }
        //step 5: perform second-pass matching.
        for (int i = 0; i < boxes.size(); i++) {
            for (int j = 0; j < boxes[i].descriptor.size(); j++) {
                int desID = boxes[i].descriptor[j];
                if (c1[desID] == 1) {
                    if (c2[desID] == 0) {
                        c2[desID] = 1;
                        for (int k = 0; k < candidateframe.size(); k++) {
                            if (fundamental[k].empty()) {
                                continue;
                            }
                            std::vector<Point2f> imgpt(1);
                            std::vector<Point3f> line(1);
                            imgpt[0] = onlineframe.keypoint[desID].pt;
                            computeCorrespondEpilines(imgpt, 1, fundamental[k], line);
                            //cout<<line[0]<<endl;
                            if (line[0].x == 0&&line[0].y==0) {//check if the line is valid.
                                continue;
                            }
                            for (int q = 0; q < m_result[k].size(); q++) {
                                std::vector<DMatch> candidateMatch;
                                for (int p = 0; p < m_result[k][q].size(); p++) {
                                    if (distP2L(keyframes[candidateframe[k]].keypoint[m_result[k][q][p].trainIdx].pt, line[0]) < 2) {
                                        candidateMatch.push_back(m_result[k][q][p]);
                                    }
                                    if (candidateMatch.size() >= 2) {
                                        break;
                                    }
                                }
                                if (candidateMatch.size()<2) {
                                    continue;
                                }
                                float distRatio = candidateMatch[0].distance/candidateMatch[1].distance;
                                if (distRatio < 0.7 && candidateMatch[0].distance<150) {
                                    matches[k].push_back(candidateMatch[0]);
                                }
                            }
                        }
                    }
                }
            }
        }
        ans = 0;
        for (int i = 0; i < matches.size(); i++) {
            ans += matches[i].size();
        }
        cout<<"second mathces: "<<ans<<endl;
        if (ans > minNumberofMatches) {
            break;
        }
        //if all feature has matched.
        bool breakflag = true;
        for (int i = 0; i < c1.size(); i++) {
            if (c1[i]==0||c2[i]==0) {
                breakflag = false;
            }
        }
        if (breakflag) {
            break;
        }
    }
    int ans = 0;
    for (int i = 0; i < matches.size(); i++) {
        ans += matches[i].size();
    }
    cout<<"final matches: "<<ans<<endl;
    cout<<"time for feature matching: "<<(getTickCount()-tt)/getTickFrequency()<<endl;
    showmatch(candidateframe, onlineframe, matches);
}

void VocTree::calibrate(const Frame &onlineframe, const std::vector<std::vector<DMatch>> &matches,const std::vector<int>& candidateframe, cv::Mat& rvec, cv::Mat&tvec){
    vector<Point3f> objpoints;
    vector<Point2f> imgpoints;
    for (int i = 0; i < matches.size(); i++) {
        for (int j = 0; j < matches[i].size(); j++) {
            if (keyframes[candidateframe[i]].keypoint[matches[i][j].trainIdx].class_id != 0) {
                objpoints.push_back(keyframes[candidateframe[i]].pos3d[matches[i][j].trainIdx]);
                imgpoints.push_back(onlineframe.keypoint[matches[i][j].queryIdx].pt);
            }
        }
    }
    if (objpoints.size()<4) {
        return;
    }
    //solvePnP(objpoints, imgpoints, intrinsic, distCoeffs, rvec, tvec);
    solvePnPRansac(objpoints, imgpoints, intrinsic, distCoeffs, rvec, tvec);
    //calibrateCamera(objpoints, imgpoints, onlineframe.img.size(), cameraMat, distCoeffs, rvec, tvec);
}


void VocTree::rendering(const Frame &onlineframe, const cv::Mat &rvec, const cv::Mat &tvec, cv::Mat &outputimg){
    onlineframe.img.copyTo(outputimg);
    std::vector<Point3f> pt3d;
    for (int i = 0; i < scenepoints.size(); i++) {
        pt3d.push_back(scenepoints[i].pt);
    }
    std::vector<Point2f> pt2d;
    if (!rvec.empty()&&!tvec.empty()) {
        projectPoints(pt3d, rvec, tvec, intrinsic, distCoeffs, pt2d);
    }
    int myradius=2;
    for (int i=0;i<pt2d.size();i++){
        circle(outputimg,cvPoint(pt2d[i].x,pt2d[i].y),myradius,CV_RGB(100,100,0),-1,8,0);
    }
    imshow("show", outputimg);
    waitKey(0);
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

void VocTree::showmatch(const std::vector<int> &candidateframe, const Frame &onlineframe, const std::vector<std::vector<DMatch>>& matches){
    for (int i = 0; i < candidateframe.size(); i++) {
        Mat out;
        drawMatches(onlineframe.img, onlineframe.keypoint, keyframes[candidateframe[i]].img, keyframes[candidateframe[i]].keypoint, matches[i], out);
        imshow("matches", out);
        waitKey(0);
    }
}

void VocTree::ordinarymatching(const std::vector<int> &candidateframe, Frame&onlineframe, std::vector<std::vector<DMatch>>& matches){
    int64 tt = getTickCount();
    for (int i = 0; i < candidateframe.size(); i++) {
        Frame* currFrame = &keyframes[candidateframe[i]];
        vector<vector<DMatch>> m_matches;
        currFrame->d_matcher.knnMatch(onlineframe.descriptor, currFrame->descriptor, m_matches, 2);
        for (int j = 0; j < m_matches.size(); j++) {
            const DMatch& bestMatch = m_matches[j][0];
            const DMatch& betterMatch = m_matches[j][1];
            
            float distRatio = bestMatch.distance/betterMatch.distance;
            if (distRatio < 0.7) {
                matches[i].push_back(bestMatch);
            }
        }
    }
    Mat tmp;
    tmp = onlineframe.img.clone();
    for (int i = 0; i < matches.size(); i++) {
        for (int j = 0; j < matches[i].size(); j++) {
            circle(tmp, onlineframe.keypoint[matches[i][j].queryIdx].pt, 2, CV_RGB(0, 40*i, 0),-1,8,0);
        }
    }
    imshow("matched", tmp);
    cout<<"time for feature matching: "<<(getTickCount()-tt)/getTickFrequency()<<"size: "<<matches.size()<<endl;
    //showmatch(candidateframe, onlineframe, matches);
    std::vector<Mat> F(candidateframe.size());
    refineMatchesWithHomography(candidateframe, onlineframe, matches, F);
    
    showmatch(candidateframe, onlineframe, matches);
}

bool VocTree::refineMatchesWithHomography(const std::vector<int> &candidateframe, Frame &onlineframe, std::vector<std::vector<DMatch>> &matches, std::vector<Mat>& Fundamental){
    //const int minNumberAllowed = 0;
    for (int i = 0; i < candidateframe.size(); i++) {
       /* if (matches[i].size() < minNumberAllowed) {
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
        if (srcpt.size()!=0) {
            //findHomography(srcpt, dstpt,CV_FM_RANSAC, 1,inliersMask);
            Fundamental[i] = findFundamentalMat(srcpt, dstpt, inliersMask, FM_RANSAC, 1.5);
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