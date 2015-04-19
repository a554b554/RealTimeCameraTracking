//
//  load.cpp
//  RCT
//
//  Created by DarkTango on 3/19/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//
#include "load.h"
static int64 __timer;
void initTimer(){
    __timer = getTickCount();
    cout<<"begin time counting..."<<endl;
}
void getTimer(){
    cout<<"time cost: "<<(getTickCount()-__timer)/getTickFrequency()<<endl;
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

inline bool isgood(Point2f p1, Point2f p2){
    if (fabs(p1.x-p2.x)<0.3&&fabs(p1.y-p2.y)<0.3) {
        return true;
    }
    return false;
}

void computekeypoint(Frame& frame, const vector<Point2f>& point){
    SiftFeatureDetector detector(2000);
    vector<KeyPoint> kpt,ktmp;
    detector.detect(frame.img, kpt);
    frame.keypoint.clear();
    //cout<<"pointsize: "<<point.size()<<endl;
    Mat data,querydata;
    data.create((int)kpt.size(), 2, CV_32F);
    querydata.create((int)point.size(), 2, CV_32F);
    for (int i = 0; i < data.rows; i++) {
        data.at<float>(i,0) = kpt[i].pt.x;
        data.at<float>(i,1) = kpt[i].pt.y;
    }
    for (int i = 0; i < querydata.rows; i++) {
        querydata.at<float>(i,0) = point[i].x;
        querydata.at<float>(i,1) = point[i].y;
    }
    BFMatcher matcher;
    //FlannBasedMatcher matcher;
    vector<DMatch> matches,gmatch;
    matcher.match(querydata, data, matches);
    
    for (int i = 0; i < point.size(); i++) {
        KeyPoint pp(point[i], 1);
        ktmp.push_back(pp);
    }
    /*drawMatches(frame.img, ktmp, frame.img, kpt, matches, data);
    imshow("before", data);
    waitKey(0);
    */
    for (int i = 0; i < point.size(); i++) {
        frame.keypoint.push_back(kpt[matches[i].trainIdx]);
        if (matches[i].distance >= 0.5 && !isgood(kpt[matches[i].trainIdx].pt , point[matches[i].queryIdx])){
            frame.keypoint[frame.keypoint.size()-1].class_id = 0;
        }
        else{
            gmatch.push_back(matches[i]);
        }
    }
    /*drawMatches(frame.img, ktmp, frame.img, kpt, gmatch, data);
    imshow("after", data);
    waitKey(0);
    */

}



/*void computekeypoint(Frame& frame, std::vector<Point2f> & point){
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
    for (int i = 0; i < point.size(); i++){
        querydata.at<float>(i,0) = point[i].x;
        querydata.at<float>(i,1) = point[i].y;
    }
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(querydata, data, matches);
    for (int i = 0; i < point.size(); i++) {
        frame.keypoint.push_back(kpt[matches[i].trainIdx]);
    }
}*/

//////////////////////////////////////////////////////////////////
void calcSaliency(ScenePoint& pt,std::vector<Frame>& inputframe){
    const u_long THRESHOLD = 30;
    u_long factor;
    double DoGstrenth = 0;
    factor = std::min(THRESHOLD, pt.img.size());
    for (int i = 0; i < pt.img.size(); i++) {
        Point p = inputframe[pt.img[i]].pos[pt.feature[i]];
        DoGstrenth += (int)inputframe[pt.img[i]].dogimg.at<uchar>(p.y, p.x);
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
    file_obj.open(basepath+"/offline/data.nvm");
    if (!file_obj.is_open()) {
        cerr<<"open "<<basepath+"/offline/data.nvm"<<" failed!"<<endl;
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
        const string imgpath = basepath + "/offline/" + fn;
        globalframe[i].img = cv::imread(imgpath);
        if (globalframe[i].img.empty()) {
            cerr<<"read image "+imgpath+" failed!"<<endl;
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
            tmp ->feature.push_back((int)globalframe[img].pos.size());
            //feature's 2D position
            double x,y;
            file_obj>>x>>y;
            double xsize = globalframe[img].img.cols/2;
            double ysize = globalframe[img].img.rows/2;
            x += xsize;
            y += ysize;
            Point2f p(x,y);
            tmp->location.push_back(p);
            if (numofmeasure > MIN_TRACK) {
                globalframe[img].pos.push_back(p);
                globalframe[img].pos3d.push_back(tmp->pt);
            }
        }
        if (numofmeasure > MIN_TRACK) {
            globalscenepoint.push_back(*tmp);
            count++;
        }
        delete tmp;
    }
    getTimer();
    file_obj.close();
}

void setupIndex(std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint){
    for (int i = 0; i < globalscenepoint.size(); i++) {
        ScenePoint* currscenepoint = &globalscenepoint[i];
        for (int j = 0; j < currscenepoint->img.size(); j++) {
            const Point2f currpt = currscenepoint->location[j];
            const Frame* currFrame = &globalframe[currscenepoint->img[j]];
            double mindist = dist(currpt, currFrame->pos[0]);
            int minidx = 0;
            for (int k = 1; k < currFrame->pos.size(); k++) {
                double tmp = dist(currpt, currFrame->pos[k]);
                if (tmp < mindist) {
                    mindist = tmp;
                    minidx = k;
                }
            }
            currscenepoint->feature[j] = minidx;
        }
    }
}

void computeAttribute(std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint){
    initTimer();
    // compute keypoint in each frame.
    cout<<"compute keypoint in each frame..."<<endl;
    for (int i = 0; i < globalframe.size(); i++) {
        computekeypoint(globalframe[i], globalframe[i].pos);
    }
    getTimer();
    
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

    cout<<"calculate saliency for each scene point..."<<endl;
    for (int i = 0; i < globalscenepoint.size(); i++) {
        calcSaliency(globalscenepoint[i], globalframe);
    }
   
    //calculate feature density
    initTimer();
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
    getTimer();
    
    //calculate descriptor
    cout<<"calculate descriptor... "<<endl;
    initTimer();
    calculateDescriptor(globalframe, globalscenepoint);
    getTimer();
    //train feature descriptor
  /*  for (int i = 0; i < globalframe.size(); i++) {
        Frame* curr = &globalframe[i];
        curr->d_matcher.clear();
        std::vector<Mat> des(1);
        des[0] = curr->descriptor.clone();
        curr->d_matcher.add(des);
        curr->d_matcher.train();
    }*/
}

void savekeyframe(const std::vector<int>& keyframe,string basepath){
    ofstream fobj;
    fobj.open(basepath+"/keyframe.txt");
    for (int i = 0; i < keyframe.size(); i++) {
        fobj<<keyframe[i]<<" ";
    }
    fobj.close();
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

void load2(const string basepath, std::vector<Frame>& keyframes, std::vector<ScenePoint>& scenepoints, const std::vector<int>& key){
    ifstream file_obj;
    int64 t0 = getTickCount();
    int numberofframe;
    string filename = basepath+"/offline/data.nvm";
    file_obj.open(filename);
    if (!file_obj.is_open()) {
        cerr<<"open "<<filename<<" failed!"<<endl;
        exit(1);
    }
    string c; //useless
    file_obj>>c;
    //load number of frame;
    file_obj>>numberofframe;
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
            curr->img = cv::imread(basepath+"/offline/"+fn);
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
            double x,y;
            file_obj>>x>>y;
            double xsize = curr->img.cols/2;
            double ysize = curr->img.rows/2;
            x += xsize; y+= ysize;
            Point2f p(x,y);
            if (kid != -1) {
                tmp ->feature.push_back((int)curr->pos.size());
                tmp->location.push_back(p);
            }
            if (numofmeasure > MIN_TRACK) {
                curr->pos.push_back(p);
                curr->pos3d.push_back(tmp->pt);
            }
            
        }
        if (numofmeasure > MIN_TRACK) {
            scenepoints.push_back(*tmp);
            count++;
        }
        delete tmp;
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
    int64 t0 = getTickCount();
    
    // compute keypoint in each frame.
    cout<<"compute keypoint in each frame..."<<endl;
    for (int i = 0; i < globalframe.size(); i++) {
        computekeypoint(globalframe[i], globalframe[i].pos);
    }
    gettime(t0);
    cout<<"compute descriptor..."<<endl;
    //calculate descriptor for each frame
    SiftDescriptorExtractor extractor;
    for (int i = 0; i < globalframe.size(); i++) {
        extractor.compute(globalframe[i].img, globalframe[i].keypoint, globalframe[i].descriptor);
    //    cv::flann::KDTreeIndexParams indexParams(5);
      //  globalframe[i].kdtree.build(globalframe[i].descriptor, indexParams, cvflann::FLANN_DIST_L2);
        //train feature descriptor
         Frame* curr = &globalframe[i];
         curr->d_matcher = new FlannBasedMatcher();
        // curr->d_matcher->clear();
         std::vector<Mat> des(1);
         des[0] = curr->descriptor.clone();
         curr->d_matcher->add(des);
         curr->d_matcher->train();
    }
    gettime(t0);
    
}

void loadonlineimglist(const string basepath, std::vector<string>& filename){
    fstream fobj;
    fobj.open(basepath+"/online/list.txt");
    string fn;
    while (fobj>>fn) {
        filename.push_back(fn);
    }
}
