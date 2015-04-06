//
//  load.cpp
//  RCT
//
//  Created by DarkTango on 3/19/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "load.h"
#include "OfflineModule.h"
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/ocl.hpp>
static int64 __timer;
void initTimer(){
    __timer = getTickCount();
    cout<<"begin time counting..."<<endl;
}
void getTimer(){
    cout<<"time cost: "<<(getTickCount()-__timer)/getTickFrequency()<<endl;
    __timer = getTickCount();
}


double dist(Point2f a, Point2f b){
    return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
}

void meanMat(const std::vector<Mat>& inputmat, Mat& output){
    for (int i = 0; i < inputmat.size(); i++) {
        output += inputmat[i];
      //  cout<<output;
    }
    output /= inputmat.size();
    //cout<<output;
}

void calculateDescriptor(std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint){
    //calculate descriptor for each frame
    for (int i = 0; i < globalframe.size(); i++) {
        
        SiftDescriptorExtractor extractor;
        
        extractor.compute(globalframe[i].img, globalframe[i].keypoint, globalframe[i].descriptor);
    }
   // cout<<norm(globalframe[2].descriptor.row(0),globalframe[14].descriptor.row(0),NORM_L2)<<endl;
    
    
    
    
    //BFMatcher matcher(NORM_L2,true);
  
    /*
    FlannBasedMatcher matcher;
    
    vector<DMatch> matches;
    vector<KeyPoint> kpt;
    
    vector<DMatch> good;
    
    matcher.match(globalframe[2].descriptor, globalframe[50].descriptor, matches);
    
    
    Mat out,outgood;
    //drawMatches(frame.img, kpt, frame.img, kpt, matches, out);
    
    double max_dist = 0; double min_dist = 500;
    for (int i = 0;  i < globalframe[1].descriptor.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) {
            min_dist = dist;
        }
        if (dist > max_dist) {
            max_dist = dist;
        }
    }
    
    for (int i = 0;  i < matches.size(); i++) {
        if (matches[i].distance < 3*min_dist) {
            good.push_back(matches[i]);
        }
    }
    cout<<good.size();
    drawMatches(globalframe[2].img, globalframe[2].keypoint, globalframe[50].img, globalframe[50].keypoint, matches, out);
    drawMatches(globalframe[2].img, globalframe[2].keypoint, globalframe[50].img, globalframe[50].keypoint, good, outgood);
    imshow("123", out);
    imshow("good", outgood);
    waitKey(0);*/
    
    
    
    
    //set globalscnenpoint's descriptor
    for (int i = 0; i < globalscenepoint.size(); i++) {
        std::vector<Mat> des;
        des.clear();
        globalscenepoint[i].descriptor.create(1, 128, CV_32F);

        for (int j = 0; j < globalscenepoint[i].img.size(); j++) {
            
           // globalscenepoint[i].descriptor += globalframe[globalscenepoint[i].img[j]].descriptor.row(globalscenepoint[i].feature[j]);
            if (globalscenepoint[i].feature[j]==0) {
                des.push_back(globalframe[globalscenepoint[i].img[j]].descriptor.row(0));
            }
            else{
                des.push_back(globalframe[globalscenepoint[i].img[j]].descriptor.row(globalscenepoint[i].feature[j]-1));
            }
            //cout<<globalframe[globalscenepoint[i].img[j]].descriptor.row(globalscenepoint[i].feature[j]).cols<<endl;
        }
        //cout<<"des.size: "<<des.size()<<endl;

        meanMat(des, globalscenepoint[i].descriptor);
        globalscenepoint[i].descriptor.convertTo(
                                                 globalscenepoint[i].descriptor, CV_8U);
        globalscenepoint[i].descriptor.convertTo(
                                                 globalscenepoint[i].descriptor, CV_32F);
    }
    /*
    for (int i = 1 ; i < 12; i++) {
        cout<<norm(globalscenepoint[i+14].descriptor, globalscenepoint[i].descriptor,NORM_L2)<<endl;
    }*/
    
    
}


void draw(const Frame& frame, string windowname){
    Mat out;
    drawKeypoints(frame.img, frame.keypoint, out);
    imshow(windowname, out);
}


void drawnativekeypoints(const Frame& frame, string windowname){
    vector<KeyPoint> kpt;
    SiftFeatureDetector detector;
    detector.detect(frame.img, kpt);
    Mat out;
    drawKeypoints(frame.img, kpt, out);
    
    imshow(windowname, out);
    
}

void drawmatch(Frame& frame, string windowname, int type){  // type1 = bfmatch type2 = flann
    SiftFeatureDetector detector(15000);
    SiftDescriptorExtractor extractor;
    
    BFMatcher matcher(NORM_L2,true);
    FlannBasedMatcher matcherflann;
    
    
    vector<DMatch> matches;
    vector<KeyPoint> kpt;
    
    detector.detect(frame.img, kpt);
    Mat des,desnative;
    
    extractor.compute(frame.img, frame.keypoint, des);
    extractor.compute(frame.img, kpt, desnative);
    
    if (type == 1) {
        matcher.match(des, desnative, matches);
    }
    else if (type ==2){
        matcherflann.match(des, desnative, matches);
    }
    
    Mat out;
    //drawMatches(frame.img, kpt, frame.img, kpt, matches, out);
    
    drawMatches(frame.img, frame.keypoint, frame.img, kpt, matches, out);
    
    imshow(windowname, out);
    
}


void drawmatch2(Frame& frame1, Frame& frame2, string windowname){
    
    BFMatcher matcher(NORM_L2,true);
    SiftDescriptorExtractor extractor;
    
    Mat des1,des2;
    extractor.compute(frame1.img, frame1.keypoint, des1);
    extractor.compute(frame2.img, frame2.keypoint, des2);
   
    
    vector<DMatch> matches;
    
    matcher.match(des1, des2, matches);
    
    Mat out;
    //drawMatches(frame.img, kpt, frame.img, kpt, matches, out);
    
    drawMatches(frame1.img, frame1.keypoint, frame2.img, frame2.keypoint, matches, out);
    
    imshow(windowname, out);
    waitKey(0);
}



void computekeypoint(Frame& frame, const vector<Point2f>& point){
    SiftFeatureDetector detector;
    vector<KeyPoint> kpt;
    detector.detect(frame.img, kpt);
    
    //cout<<"pointsize: "<<point.size()<<endl;
    
    Mat data,querydata;
    data.create(kpt.size(), 2, CV_32F);
    querydata.create(point.size(), 2, CV_32F);
    for (int i = 0; i < data.rows; i++) {
        data.at<float>(i,0) = kpt[i].pt.x;
        data.at<float>(i,1) = kpt[i].pt.y;
    }
    
    for (int i = 0; i < querydata.rows; i++) {
        querydata.at<float>(i,0) = point[i].x;
        querydata.at<float>(i,1) = point[i].y;
    }
    
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(querydata, data, matches);
    
    for (int i = 0; i < point.size(); i++) {
        frame.keypoint.push_back(kpt[matches[i].trainIdx]);
    }
    
   
    
    
  /*  Mat k1,k2,k3;
    drawKeypoints(frame.img,kpt, k1);
    
    kpt.clear();
    for (int i = 0; i < point.size(); i++) {
        KeyPoint k;
        k.pt.x = point[i].x;
        k.pt.y = point[i].y;
        kpt.push_back(k);
    }
    drawKeypoints(frame.img, kpt, k2);
    drawKeypoints(frame.img, frame.keypoint, k3);
    imshow("all",k1);
    imshow("native", k2);
    imshow("new", k3);*/
   // waitKey(0);
    
    
    
    
    
    /*
    for (int i = 0; i < point.size(); i++) {
        double mindist = dist(point[i], kpt[0].pt);
        int index = 0;
        
        for (int j = 0; j < kpt.size(); j++) {
            double _dist = dist(point[i], kpt[j].pt);
            if (_dist < mindist) {
                mindist = _dist;
                index = j;
            }
        }
        
        frame.keypoint.push_back(kpt[index]);
        
    }*/
    
    
    
    
}

void computekeypoint(Frame& frame, map<int, Point2f> & point){
    SiftFeatureDetector detector;
    vector<KeyPoint> kpt;
    detector.detect(frame.img, kpt);
    //cout<<"pointsize: "<<point.size()<<endl;
    Mat data,querydata;
    data.create((int)kpt.size(), 2, CV_32F);
    querydata.create((int)point.size(), 2, CV_32F);
    for (int i = 0; i < data.rows; i++) {
        data.at<float>(i,0) = kpt[i].pt.x;
        data.at<float>(i,1) = kpt[i].pt.y;
    }
    int i = 0;
    map<int,Point2f>::iterator it;
    for (it = point.begin(); it != point.end(); it++) {
        querydata.at<float>(i,0) = it->second.x;
        querydata.at<float>(i,1) = it->second.y;
        i++;
    }
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(querydata, data, matches);
    i = 0;
    for (it = point.begin(); it != point.end(); it++) {
        frame.keypoint.push_back(kpt[matches[i].trainIdx]);
        frame.pos_id.push_back(it->first);
        i++;
    }

}


void calcSaliency(ScenePoint& pt,std::vector<Frame>& inputframe){
    const u_long THRESHOLD = 30;
    u_long factor;
    double DoGstrenth = 0;
    
    factor = std::min(THRESHOLD, pt.img.size());
   // std::min(4,6);
    
    for (int i = 0; i < pt.img.size(); i++) {
        //cout<<pt.location[i]<<endl;
        //cout<<"rows: "<<inputframe[pt.img[i]].img.rows<<" cols: "<<inputframe[pt.img[i]].img.cols<<endl;
        //cout<<"location: "<<pt.location[i]<<endl;
        //cout<<"feature: "<<inputframe[pt.img[i]].pos[pt.feature[i]]<<endl;
        //inputframe[pt.img[i]].pos[0] = Point(0,0);
        Point p = inputframe[pt.img[i]].pos[pt.feature[i]];
        
      
    
        DoGstrenth += (int)inputframe[pt.img[i]].dogimg.at<uchar>(p.y, p.x);
  
        
        
        
        //DoG(inputframe[pt.img[i]].dogimg, pt.location[i]);
        //cout<<"current dog:"<<DoGstrenth<<endl;
    }
    DoGstrenth /= pt.img.size();
    pt.saliency = DoGstrenth * factor;
}

void gettime(int64& t0){
    cout<<"time cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
    t0 = getTickCount();

}

void load(const string basepath, std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint){
    ifstream file_obj;
    int numberofframe;
    file_obj.open(basepath+"/data.nvm");
    if (!file_obj.is_open()) {
        cerr<<"open "<<basepath+"./data.nvm"<<" failed!"<<endl;
        exit(1);
    }
    string c; //useless
    file_obj>>c;
    //load number of frame;
    file_obj>>numberofframe;
    //init frame set
    for (int i = 0; i < numberofframe ; i++) {
        Frame *tmp = new Frame();
        globalframe.push_back(*tmp);
    }
    //load camera parameters;
    cout<<"loading camera parameters..."<<endl;
    initTimer();
    for (int i = 0; i < numberofframe; i++) {
        string fn;
        file_obj>>fn;
        globalframe[i].img = cv::imread(basepath+"/"+fn);
        if (globalframe[i].img.empty()) {
            cerr<<"read image "+fn+" failed!"<<endl;
            exit(0);
        }
        file_obj>>globalframe[i].F;
        double a;
        //quanternions
        file_obj>>a;
        globalframe[i].quanternions.push_back(a);
        file_obj>>a;
        globalframe[i].quanternions.push_back(a);
        file_obj>>a;
        globalframe[i].quanternions.push_back(a);
        file_obj>>a;
        globalframe[i].quanternions.push_back(a);
        file_obj>>globalframe[i].location.x;
        file_obj>>globalframe[i].location.y;
        file_obj>>globalframe[i].location.z;
        //radial distortion
        file_obj>>globalframe[i].k;
        file_obj>>c;
    }
    getTimer();
    //load scene point information
    cout<<"loading scene point information..."<<endl;
    initTimer();
    int numberof3dpoint;
    file_obj>>numberof3dpoint;
    //cout<<numberof3dpoint<<endl;
    int count = 0;
    for (int i = 0; i < numberof3dpoint; i++) {
        ScenePoint *tmp = new ScenePoint();
        //load location
        file_obj >> tmp -> pt.x;
        file_obj >> tmp -> pt.y;
        file_obj >> tmp -> pt.z;
        //load RGB
        file_obj >> tmp -> RGB.x;
        file_obj >> tmp -> RGB.y;
        file_obj >> tmp -> RGB.z;
        int numofmeasure;
        file_obj>>numofmeasure;
        for (int j = 0; j < numofmeasure; j++) {
            int img;
            file_obj>>img;
            tmp -> img.push_back(img);
            if (numofmeasure > MIN_TRACK) {
                globalframe[img].scenepoint.push_back(count);
            }
            int feature;
            file_obj>>feature;
            tmp ->feature.push_back(feature);
            //feature's 2D position
            // cout<<feature<<endl;
            double x,y;
            file_obj>>x>>y;
            double xsize = globalframe[img].img.cols/2;
            double ysize = globalframe[img].img.rows/2;
            x += xsize;
            y += ysize;
            Point2f p(x,y);
            tmp->location.push_back(p);
            if (numofmeasure > MIN_TRACK) {
                globalframe[img].pos[feature] = p;
                globalframe[img].pos3d[feature] = tmp->pt;
                //globalframe[img].pos.push_back(p);
                //globalframe[img].pos_id.push_back(feature);
                globalframe[img].featuresize++;
            }
        }
        if (numofmeasure > MIN_TRACK) {
            globalscenepoint.push_back(*tmp);
            count++;
        }
        delete tmp;
    }
    file_obj.close();
}

void computeAttribute(std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint){
    initTimer();
    // compute keypoint in each frame.
    cout<<"compute keypoint in each frame..."<<endl;
    for (int i = 0; i < globalframe.size(); i++) {
        computekeypoint(globalframe[i], globalframe[i].pos);
    }
    getTimer();
    //drawmatch(globalframe[12], "test");
    // compute dog mat in each frame.
    cout<<"compute dog image in each frame..."<<endl;
    for (int i = 0; i < globalframe.size(); i++) {
        Mat g1,g2;
        
        cv::GaussianBlur(globalframe[i].img, g1, Size(1,1), 0);
        cv::GaussianBlur(globalframe[i].img, g2, Size(3,3), 2.0,2.0);
        cvtColor(g1, g1, CV_BGR2GRAY);
        cvtColor(g2, g2, CV_BGR2GRAY);
        globalframe[i].dogimg = g1 - g2;
        
        
        
        /*cout<<globalframe[i].dogimg.type()<<endl;
         
         waitKey(0);*/
    }
    //  imshow("1", globalframe[6].dogimg);
    
    
    
    
    // calculate saliency for each scene point
    cout<<"calculate saliency for each scene point..."<<endl;
    for (int i = 0; i < globalscenepoint.size(); i++) {
        calcSaliency(globalscenepoint[i], globalframe);
    }
    
    
    
    //calculate feature density
    cout<<"calculate feature density for each frame..."<<endl;
    for (int i = 0; i < globalframe.size(); i++) {
        //for each frame
        for (int j = 0; j < globalframe[i].pos.size(); j++) {
            
            Point2f point = globalframe[i].pos[j];
            float leftbound = point.x - 15;
            float rightbound = point.x + 15;
            float upperbound = point.y + 15;
            float lowerbound = point.y - 15;
            //cout<<"featuresize: "<<features.size()<<endl;
            int count = 0;
            for (int k = 0; k < globalframe[i].pos.size(); k++) {
                if (globalframe[i].pos[k].x >= leftbound &&
                    globalframe[i].pos[k].x <= rightbound &&
                    globalframe[i].pos[k].y >= lowerbound &&
                    globalframe[i].pos[k].y <= upperbound) {
                    count += 1;
                }
            }
            //cout<<"density: "<<count<<"size: "<<globalframe[i].pos.size()<<endl;
            globalframe[i].featuredensity.push_back(count);
        }
    }
    
    //calculate descriptor
    cout<<"calculate descriptor... "<<endl;
    calculateDescriptor(globalframe, globalscenepoint);
    

}

string toString(int a){
    stringstream ss;
    string ans;
    ss<<a;
    ss>>ans;
    return ans;
}


int keyframeid(const std::vector<int>& key, int query_id){
    for (int i = 0; i < key.size(); i++) {
        if (key[i] == query_id) {
            return i;
        }
    }
    return -1;
}

void load2(const char* filename, std::vector<Frame>& keyframes, std::vector<ScenePoint>& scenepoints, const std::vector<int>& key){
    ifstream file_obj;
    int64 t0 = getTickCount();
    int numberofframe;
    file_obj.open(filename);
    if (!file_obj.is_open()) {
        cerr<<"open "<<filename<<" failed!"<<endl;
        exit(1);
    }
    string c; //useless
    file_obj>>c;
    //load number of frame;
    file_obj>>numberofframe;
    //cout<<numberofframe<<endl;
    
    //init frame set
    for (int i = 0; i < numberofframe ; i++) {
        if (keyframeid(key,i) != -1) {
            Frame *tmp = new Frame();
            keyframes.push_back(*tmp);
        }
    }
    //load camera parameters;
    cout<<"loading camera parameters..."<<endl;
    for (int i = 0; i < numberofframe; i++) {
        int kid = keyframeid(key,i);

        string fn;
        file_obj>>fn;
        Frame *curr = new Frame();
        if (kid != -1) {
            curr = &keyframes[kid];
            curr->img = cv::imread("./campusSFM/"+fn);
        }
        file_obj>>keyframes[i].F;
        double a;
        //quanternions
        file_obj>>a;
        curr->quanternions.push_back(a);
        file_obj>>a;
        curr->quanternions.push_back(a);
        file_obj>>a;
        curr->quanternions.push_back(a);
        file_obj>>a;
        curr->quanternions.push_back(a);
        
        // camera center
        file_obj>>curr->location.x;
        file_obj>>curr->location.y;
        file_obj>>curr->location.z;
    
        //radial distortion
        file_obj>>curr->k;
        
        file_obj>>c;

        
    }
    gettime(t0);
    
    
    //load scene point information
    cout<<"loading scene point information..."<<endl;
    int numberof3dpoint;
    file_obj>>numberof3dpoint;
    //cout<<numberof3dpoint<<endl;
    
    int count = 0;
    Frame* ttt = new Frame();
    for (int i = 0; i < numberof3dpoint; i++) {
        ScenePoint *tmp = new ScenePoint();
        
        //load location
        file_obj >> tmp -> pt.x;
        file_obj >> tmp -> pt.y;
        file_obj >> tmp -> pt.z;
        
        //load RGB
        file_obj >> tmp -> RGB.x;
        file_obj >> tmp -> RGB.y;
        file_obj >> tmp -> RGB.z;
        
        int numofmeasure;
        file_obj>>numofmeasure;
        
        for (int j = 0; j < numofmeasure; j++) {
            int img;
            file_obj>>img;
            int kid = keyframeid(key, img);
            Frame* curr = ttt;
            if (kid != -1) {
                curr = &keyframes[kid];
                tmp -> img.push_back(kid);
            }
            
            if (numofmeasure > MIN_TRACK) {
                curr->scenepoint.push_back(count);
            }
            int feature;
            file_obj>>feature;
            tmp ->feature.push_back(feature);
            double x,y;
            file_obj>>x>>y;
            double xsize = curr->img.cols/2;
            double ysize = curr->img.rows/2;
            x += xsize; y+= ysize;
            Point2f p(x,y);
            tmp->location.push_back(p);
            if (numofmeasure > MIN_TRACK) {
                //globalframe[img].pos.push_back(p);
                curr->pos[feature] = p;
                curr->pos3d[feature] = tmp->pt;
                //curr->pos.push_back(p);
                //curr->pos_id.push_back(feature);
                //curr->featuresize++;
            }
            
        }
        if (numofmeasure > MIN_TRACK) {
            scenepoints.push_back(*tmp);
            count++;
            delete tmp;
        }
        else{
            delete tmp;
        }
        
    }
    gettime(t0);
    file_obj.close();
    
}

void showallframe(const std::vector<Frame>& frameset){
    for (int i = 0; i < frameset.size(); i++) {
        imshow("show"+toString(i), frameset[i].img);
    }
    waitKey(0);
    for (int i = 0; i <frameset.size(); i++) {
        destroyWindow("show"+toString(i));
    }
}

void computeAttribute2(std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint){
    int64 t0 =getTickCount();
    
    // compute keypoint in each frame.
    cout<<"compute keypoint in each frame..."<<endl;
    for (int i = 0; i < globalframe.size(); i++) {
        computekeypoint(globalframe[i], globalframe[i].pos);
    }
    gettime(t0);
    
    //drawmatch(globalframe[12], "test");
    
    cout<<"compute descriptor..."<<endl;
    //calculate descriptor for each frame
    SiftDescriptorExtractor extractor;
    for (int i = 0; i < globalframe.size(); i++) {
        extractor.compute(globalframe[i].img, globalframe[i].keypoint, globalframe[i].descriptor);
        cv::flann::KDTreeIndexParams indexParams(5);
        globalframe[i].kdtree.build(globalframe[i].descriptor, indexParams, cvflann::FLANN_DIST_L2);
    }
    gettime(t0);
    
}

