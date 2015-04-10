//
//  OfflineModule.cpp
//  RCT
//
//  Created by DarkTango on 3/15/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "OfflineModule.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <set>
#include "dump.h"
static const int MAX_FRAME = 40000;


/*uchar DoG(const Mat& dogimg, Point2f position){
    uchar ans = dogimg.at<uchar>((int)position.x ,(int)position.y);
    return ans;
}*/

static int64 _t0;
static void initTimer(){
    _t0 = getTickCount();
    cout<<"timing.."<<endl;
}
static void getTimer(){
    cout<<(getTickCount()-_t0)/getTickFrequency()<<endl;
}

int density(const vector<KeyPoint>& features, const Point2f& point){
    int ans = 0;
    float leftbound = point.x - 15;
    float rightbound = point.x + 15;
    float upperbound = point.y + 15;
    float lowerbound = point.y - 15;
    //cout<<"featuresize: "<<features.size()<<endl;
    for (int i = 0; i < features.size(); i++) {
        if (features[i].pt.x >= leftbound &&
            features[i].pt.x <= rightbound &&
            features[i].pt.y >= lowerbound &&
            features[i].pt.y <= upperbound) {
            ans += 1;
        }
    }
    return ans;
}


double saliency(const ScenePoint& pt,const std::vector<Frame>& inputframe){
    const u_long THRESHOLD = 30;
    u_long factor;
    double DoGstrenth = 0;
    
    factor = std::min(THRESHOLD, pt.img.size());
    std::min(4,6);
    
    for (int i = 0; i < pt.img.size(); i++) {
        
        DoGstrenth += inputframe[pt.img[i]].dogimg.at<uchar>((int)pt.location[i].x, (int)pt.location[i].y);
        //DoG(inputframe[pt.img[i]].dogimg, pt.location[i]);
        //cout<<"current dog:"<<DoGstrenth<<endl;
    }
    DoGstrenth /= pt.img.size();
    return DoGstrenth * factor;
}

double FeatureDensity(const ScenePoint& pt, const std::vector<Frame>& inputframe){
    double ans = 0;
    for (int i = 0; i < pt.img.size(); i++) {
        ans += inputframe[pt.img[i]].featuredensity[pt.feature[i]];
        
    }
    return ans /= pt.img.size();
}



bool isScenePointInFrameSet(const ScenePoint& inputscenepoint, const std::vector<int>& frameset){
   /* for (int i = 0; i < inputscenepoint.img.size(); i++) {
        for (int j = 0; j < frameset.size(); j++) {
            if (inputscenepoint.img[i] == frameset[j]) {
                return true;
            }
        }
    }*/
    char tmp[5000];
    /*for (int i = 0; i < 5000; i++) {
        tmp[i] = 0;
    }*/
    memset(tmp, 0, sizeof(char)*5000);
    for (int i = 0; i < frameset.size(); i++) {
        tmp[frameset[i]] = '1';
    }
    
    for (int i = 0; i < inputscenepoint.img.size(); i++) {
        if (tmp[inputscenepoint.img[i]] == '1') {
            return true;
        }
    }
    return false;
}


const int yita = 3;
double CompletenessTerm(const std::vector<Frame>& inputframe, const std::vector<int>& candidateframe, const std::vector<ScenePoint>& inputscenepoint){

    double ans;
    double factor1,factor2;  // ans = 1 - (factor1/factor2)
    factor1 = 0;
    factor2 = 0;
    cout<<"computing completeness..."<<endl;
    initTimer();
    /*for (int i = 0; i < inputscenepoint.size(); i++) {
        double term = inputscenepoint[i].saliency/(yita + FeatureDensity(inputscenepoint[i], inputframe));
        std::vector<int> unionset;
        unionframe(inputscenepoint[i].img, candidateframe, unionset);
        if (!unionset.empty()) {
            factor1 += term;
        }
        factor2 += term;
    }*/
   /* int hashtable[MAX_FRAME];
    memset(hashtable, 0, sizeof(int)*MAX_FRAME);
    for (int i = 0; i < candidateframe.size(); i++) {
        for (int j = 0; j < inputframe[candidateframe[i]].scenepoint.size(); j++) {
            hashtable[inputframe[candidateframe[i]].scenepoint[j]] = 1;
        }
    }
    std::vector<int> idx;
    for (int i = 0; i < MAX_FRAME; i++) {
        if (hashtable[i] == 1) {
            idx.push_back(i);
        }
    }
    for (int i = 0; i < idx.size(); i++) {
        factor1 += inputscenepoint[idx[i]].saliency/(yita+FeatureDensity(inputscenepoint[idx[i]], inputframe));
    }*/
    
    std::set<int> _set;
    for (int i = 0; i < candidateframe.size(); i++) {
        for (int j = 0; j < inputframe[candidateframe[i]].scenepoint.size(); j++) {
            _set.insert(inputframe[candidateframe[i]].scenepoint[j]);
        }
    }
    std::set<int>::iterator it;
    for (it = _set.begin(); it != _set.end(); it++) {
        factor1 += inputscenepoint[*it].saliency/(yita+FeatureDensity(inputscenepoint[*it], inputframe));
        //cout<<"it:"<<*it<<endl;
    }
    
    for (int i = 0; i < inputscenepoint.size(); i++) {
        factor2 += inputscenepoint[i].saliency/(yita+FeatureDensity(inputscenepoint[i], inputframe));
    }
    cout<<"factor1 "<<factor1<<" factor2 "<<factor2<<endl;
    ans = 1 - (factor1 / factor2);
    cout<<"compute complete!"<<endl;
    cout<<"Completeness: "<<ans<<endl;
    getTimer();
    return ans;
}



double Redundancy(const std::vector<Frame>& inputframe, const std::vector<int>& candidateframe, const std::vector<ScenePoint>& inputscenepoint){
    int ans = 0;
    
    cout<<"computing redundancy..."<<endl;
    initTimer();
   /* int hashtable[MAX_FRAME];
    memset(hashtable, 0, sizeof(int)*MAX_FRAME);
    for (int i = 0; i < candidateframe.size(); i++) {
        for (int j = 0; j < inputframe[candidateframe[i]].scenepoint.size(); j++) {
            hashtable[inputframe[candidateframe[i]].scenepoint[j]] = 1;
        }
    }
    std::vector<int> idx;
    for (int i = 0; i < MAX_FRAME; i++) {
        if (hashtable[i] == 1) {
            idx.push_back(i);
        }
    }
    for (int i = 0; i < idx.size(); i++) {
        std::vector<int> unionset;
        unionframe(inputscenepoint[idx[i]].img, candidateframe, unionset);
        if (!unionset.empty()) {
            ans += unionset.size() - 1;
        }
    }*/
    
    std::set<int> _set;
    for (int i = 0; i < candidateframe.size(); i++) {
        for (int j = 0; j < inputframe[candidateframe[i]].scenepoint.size(); j++) {
            _set.insert(inputframe[candidateframe[i]].scenepoint[j]);
        }
 
    }
    
    std::set<int>::iterator it;
    for (it = _set.begin(); it != _set.end(); it++) {
        std::vector<int> unionset;
        unionframe(inputscenepoint[*it].img, candidateframe, unionset);
        if (!unionset.empty()) {
            ans += unionset.size() - 1;
        }
    }
    cout<<"compute complete!"<<endl;
    cout<<"Redundancy: "<<( (double)ans / inputframe.size())<<endl;
    getTimer();
    return double(ans) / inputframe.size();
}

bool frameisinframeset(const std::vector<int>& frameset, const int frameid){
    for (int i = 0; i < frameset.size(); i++) {
        if (frameset[i] == frameid) {
            return true;
        }
    }
    return false;
}


void KeyframeSelection(const std::vector<Frame>& inputframe, const std::vector<ScenePoint>& inputscenepoint
                       ,std::vector<int>& outputkeyframe){
    const double redundancyfactor = 0.005;
    while (1) {
        double origin_cost = redundancyfactor * Redundancy(inputframe, outputkeyframe, inputscenepoint)+
        CompletenessTerm(inputframe, outputkeyframe, inputscenepoint);
        cout<<"oringcost: "<<origin_cost<<endl;
        int best_id = -1;
        double difference = 0;
        for (int i = 0; i < inputframe.size(); i++) {
            if (!frameisinframeset(outputkeyframe, i)) { // if the frame is not in the outputkeyframe
                outputkeyframe.push_back(i);
                cout<<i<<endl;
                double cost = redundancyfactor * Redundancy(inputframe, outputkeyframe, inputscenepoint)+
                CompletenessTerm(inputframe, outputkeyframe, inputscenepoint);
                cout<<"currentcost: "<<origin_cost<<endl;
                if (origin_cost - cost > difference) {
                    best_id = i;
                    difference = origin_cost - cost;
                }
                outputkeyframe.pop_back();
            }
        }
        if (best_id == -1) {
            break;
        }
        else{
            outputkeyframe.push_back(best_id);
        }
        //for debug
        cout<<"size: "<<outputkeyframe.size()<<" ";
        for (int i = 0; i < outputkeyframe.size(); i++) {
            cout<<outputkeyframe[i];
        }
        cout<<endl;
    }
}



int countscnenpoint(const std::vector<ScenePoint>& scenepoint, int threshold){
    int count = 0;
    for (int i = 0; i < scenepoint.size(); i++) {
        if (scenepoint[i].img.size() >= threshold) {
            count++;
        }
    }
    return  count;
}

void fakeKeyFrameSelection(std::vector<int>& keyframes,string basepath){
    int keyid;
    fstream fobj;
    fobj.open(basepath+"/keyframe.txt");
    while (fobj>>keyid) {
        keyframes.push_back(keyid);
    }
}

