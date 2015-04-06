//
//  VocabularyTree.cpp
//  RCT
//
//  Created by DarkTango on 3/18/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "VocabularyTree.h"
#include <iostream>
#include "load.h"
#include <set>
#include <queue>

double distan(const Mat& vec1, const Mat& vec2){ // return the distance between vec1 and vec2.
    return cv::norm(vec1, vec2, NORM_L2);
}


VocabularyTree::VocabularyTree(const std::vector<ScenePoint>& globalscenepoint,const std::vector<Frame>& globalframe,const std::vector<int>& keyframes){
    this->globalframe = globalframe;
    this->globalscenepoint = globalscenepoint;
    this->keyframes = keyframes;
    for (int i = 0; i < globalscenepoint.size() ; i++) {
        this->root.scenepoint.push_back(i);
    }
    
}






void VocabularyTree::construction(int b, int L, treenode &node){ //constuct the vocabulary tree
    //node.weight = log(double(keyframes.size())/spannedkeyframe(node));
    if (L == 0) {
        return;
    }
    
    cout<<"kmeans L:"<<L<<endl;
    int64 t0 = getTickCount();
    kmeans(node, b);
    gettime(t0);
    for (int i = 0; i < node.child.size(); i++) {
        construction(b, L-1, *node.child[i]);
    }
    
}


void VocabularyTree::init(int b, int L){
    int64 t0 = getTickCount();
    cout<<"constructing vocabulary tree..."<<endl;
    construction(b, L, root);
    gettime(t0);
    
    cout<<"updating img index for vocabulary tree..."<<endl;
    updateimg(root);
    gettime(t0);
    
    cout<<"updating weight for vocabulary tree..."<<endl;
    updateweight(root);
    gettime(t0);
    
}


void VocabularyTree::kmeans(treenode& node, int b){  // b is the cluster number
    std::vector<treenode*> childnode;
    for (int i = 0; i < b; i++) {
        treenode* node = new treenode();
        childnode.push_back(node);
    }
    Mat *lable = new Mat;
    Mat *combined = new Mat;
    
    for (int i = 0; i < node.scenepoint.size(); i++) {
        combined->push_back(globalscenepoint[node.scenepoint[i]].descriptor.row(0));
    }
    //cout<<*combined<<endl;
    //cout<<"____________________"<<endl;
    //cout<<globalframe[1].descriptor<<endl;
    if (node.scenepoint.size() < b) {
        return;
    }
    
    cv::TermCriteria term(CV_TERMCRIT_EPS,50,0.0001);
    
    
    int clusterFactor = b;
    
    if (clusterFactor!=0) {
        cv::kmeans(*combined, clusterFactor, *lable, term, 50, KMEANS_PP_CENTERS);
    }

  /*  cout<<lable->type()<<endl;
    for (int i = 0; i < lable->rows; i++) {
        std::cout<<lable->at<int>(i,0)<<std::endl;
    }*/
    
    delete combined;
    for (int i = 0; i < lable->rows; i++) {
        int cluster = lable->at<int>(i,0);
        childnode[cluster]->scenepoint.push_back(node.scenepoint[i]);
        
    }
    
    delete lable;
    
    for (int i = 0; i < clusterFactor ; i++) {
        node.child.push_back(childnode[i]);
    }
    
}



int VocabularyTree::spannedkeyframe(const treenode &node)const{
    int count = 0;
    std::vector<int> tb(keyframes.size(),0);
    for (int i = 0; i < node.scenepoint.size(); i++) {
        for (int j = 0; j < globalscenepoint[node.scenepoint[i]].img.size(); i++) {
            tb[globalscenepoint[node.scenepoint[i]].img[j]] = 1;
        }
    }
    
    for (int i = 0; i < keyframes.size(); i++) {
        if (tb[i] == 1) {
            count++;
        }
    }
    
    return count;
}



void VocabularyTree::updateimg(treenode& node){
    std::set<int> tmp;
    for (int i = 0; i < node.scenepoint.size(); i++) {
        for (int j = 0; j < globalscenepoint[node.scenepoint[i]].img.size(); j++) {
            if (iskeyframe(globalscenepoint[node.scenepoint[i]].img[j])) {
                tmp.insert(globalscenepoint[node.scenepoint[i]].img[j]);
            }
        }
    }
    
    std::set<int>::iterator it;
    for (it = tmp.begin(); it != tmp.end(); it++) {
        node.img.push_back(*it);
    }
    
    for (int i = 0; i < node.child.size(); i++) {
        updateimg(*node.child[i]);
    }
    
}

bool VocabularyTree::iskeyframe(int _id){
    for (int i = 0; i < keyframes.size(); i++) {
        if (keyframes[i] == _id) {
            return true;
        }
    }
    return false;
}


void VocabularyTree::updateweight(treenode& node){
    node.weight = log((float)keyframes.size()/node.img.size());
    for (int i  = 0; i < node.child.size(); i++) {
        updateweight(*node.child[i]);
    }
}





const double THRESHOLD_FOR_WEIGHT = 0;
void VocabularyTree::findCandidateFrame(const Frame &inputliveframe, std::vector<int>& outputframe, int K){//find K keyframes most related to input live frame
    
    std::vector<double> v_match;
    for (int i = 0; i < globalframe.size(); i++) {
        v_match.push_back(0);
    }
    
    for (int i = 0; i < inputliveframe.descriptor.rows; i++) {
        std::queue<treenode*> que;
        que.push(&root);
        while (!que.empty()) { // level order traversal.
            
            int size = que.size();
            
            double mindist = 100000;
            treenode* minnode = que.front();
            for (int j = 0; j < size; j++) {
                treenode* curr = que.front();
                for (int k = 0; k < curr->child.size(); k++) {
                    que.push(que.front()->child[k]);
                }
                double tmp = minDistInNode(*curr, inputliveframe.descriptor.row(i));
                if (tmp < mindist) {
                    mindist = tmp;
                    minnode = curr;
                }
                que.pop();
            }
            if (minnode->weight > THRESHOLD_FOR_WEIGHT) {
                for (int j = 0; j < minnode->img.size(); j++) {
                    if (iskeyframe(minnode->img[j])) {
                        v_match[minnode->img[j]] += featureinframe(inputliveframe, *minnode) * minnode->weight;
                    }
                }
            }
        }
    }
}

double VocabularyTree::minDistInNode(const treenode &node, const cv::Mat &descriptor){
    return 0;
}

void VocabularyTree::liveMat2Frame(const cv::Mat& inputlivemat, Frame& outputframe){
    outputframe.img = inputlivemat;
    SiftFeatureDetector detector;
    SiftDescriptorExtractor extractor;
    std::vector<KeyPoint> keypoints;
    
    
    
    detector.detect(outputframe.img, keypoints);
    outputframe.keypoint = keypoints;
    extractor.compute(outputframe.img, keypoints, outputframe.descriptor);
    
}


int VocabularyTree::featureinframe(const Frame& frame, const treenode& node){
    int ans = 0;
    std::vector<int> hash(globalscenepoint.size());
    for (int i = 0; i < frame.scenepoint.size(); i++) {
        hash[frame.scenepoint[i]] = 1;
    }
    
    for (int i = 0; i < node.scenepoint.size(); i++) {
        if (hash[node.scenepoint[i]] == 1) {
            ans++;
        }
    }
    
    return ans;
}



void knnsamplecode(){
    
    int numData = 10000;
    int numQueries = 2;
    int numDimensions = 128;
    int k = 2;
    float Mean = 0.0f;
    float Variance = 1.0f;
    // Create the data
    Mat features(numData,numDimensions,CV_32F), query(numQueries,numDimensions,CV_32F);
    randu(features, Scalar::all(Mean), Scalar::all(Variance));
    randu(query, Scalar::all(Mean), Scalar::all(Variance));
    
    // Print generated data
    cout << "Input::" << endl;
    for(int row = 0 ; row < features.rows ; row++){
        for(int col = 0 ; col < features.cols ; col++){
            //cout << features.at<float>(row,col) <<"\t";
        }
        // cout << endl;
    }
    cout << "Query::" << endl;
    for(int row = 0 ; row < query.rows ; row++){
        for(int col = 0 ; col < query.cols ; col++){
            //  cout << query.at<float>(row,col) <<"\t";
        }
        //  cout << endl;
    }
    
    // KdTree with 5 random trees
    cv::flann::KDTreeIndexParams indexParams(5);
    
    // You can also use LinearIndex
    //cv::flann::LinearIndexParams indexParams;
    
    // Create the Index
    cv::flann::Index kdtree(features, indexParams);
    
    // Perform single search for mean
    cout << "Performing single search to find closest data point to mean:" << endl;
    vector<float> singleQuery;
    vector<int> index(1);
    vector<float> dist(1);
    
    // Searching for the Mean
    for(int i = 0 ; i < numDimensions ;i++){
        singleQuery.push_back(Mean);
    }
    
    // Invoke the function
    kdtree.knnSearch(singleQuery, index, dist, 1, cv::flann::SearchParams(64));
    
    // Print single search results
    cout << "(index,dist):" << index[0] << "," << dist[0]<< endl;
    
    // Batch: Call knnSearch
    cout << "Batch search:"<< endl;
    Mat indices;//(numQueries, k, CV_32S);
    Mat dists;//(numQueries, k, CV_32F);
    
    // Invoke the function
    kdtree.knnSearch(query, indices, dists, k, cv::flann::SearchParams(64));
    
    cout << indices.rows << "\t" << indices.cols << endl;
    cout << dists.rows << "\t" << dists.cols << endl;
    
    // Print batch results
    cout << "Output::"<< endl;
    for(int row = 0 ; row < indices.rows ; row++){
        cout << "(index,dist):";
        for(int col = 0 ; col < indices.cols ; col++){
            cout << "(" << indices.at<int>(row,col) << "," << dists.at<float>(row,col) << ")" << "\t";
        }
        cout << endl;
    }

}





