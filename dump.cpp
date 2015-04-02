//
//  dump.cpp
//  RCT
//
//  Created by DarkTango on 3/27/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

/*
 NVM_V3 [optional calibration]                        # file version header
 <Model1> <Model2> ...                                # multiple reconstructed models
 <Empty Model containing the unregistered Images>     # number of camera > 0, but number of points = 0
 <0>                                                  # 0 camera to indicate the end of model section
 <Some comments describing the PLY section>
 <Number of PLY files> <List of indices of models that have associated PLY>
 
 The [optional calibration] exists only if you use "Set Fixed Calibration" Function
 FixedK fx cx fy cy
 
 Each reconstructed <model> contains the following
 <Number of cameras>   <List of cameras>
 <Number of 3D points> <List of points>
 
 The cameras and 3D points are saved in the following format
 <Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
 <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
 <Measurement> = <Image index> <Feature Index> <xy>
*/

#include "dump.h"
#include <fstream>

void unionframe(const std::vector<int>& keyframes, const std::vector<int>& imgs, std::vector<int>& unionframe){
    int tb[2000];
    for (int i = 0; i < 2000; i++) {
        tb[i] = 0;
    }
    unionframe.clear();
    for (int i = 0; i < keyframes.size(); i++) {
        tb[keyframes[i]] = 1;
    }
    for (int i = 0; i < imgs.size(); i++) {
        if (tb[imgs[i]] == 1) {
            unionframe.push_back(imgs[i]);
        }
    }
}

int imgindex(const std::vector<int>& keyframes, int query){
    for (int i = 0; i < keyframes.size(); i++) {
        if (query == keyframes[i]) {
            return i;
        }
    }
    return -1;
}



void savedata(const std::vector<Frame>& globalframe, const std::vector<ScenePoint>& globalscenepoint, const std::vector<int>& keyframes, string filename){
    ofstream fobj;
    fobj.open(filename+".dt");
    fobj<<"NVM_V3"<<endl<<endl;
    fobj<<keyframes.size()<<endl;
    for (int i = 0; i < keyframes.size(); i++) {
        const Frame& curr = globalframe[keyframes[i]];
        if (keyframes[i]<1000) {
            fobj<<"0"<<keyframes[i]<<".jpg\t";
        }
        else{
            fobj<<keyframes[i]<<".jpg\t";
        }
        fobj<<curr.F<<" "<<curr.quanternions[0]<<" "<<curr.quanternions[1]<<" "<<curr.quanternions[2]<<" "<<curr.quanternions[3]<<" "<<curr.location.x<<" "<<curr.location.y<<" "<<curr.location.z<<" "<<curr.k<<" "<<0<<endl;
        
    }
    cout<<endl;
    
    fobj<<globalscenepoint.size()<<endl;
    for (int i = 0; i < globalscenepoint.size(); i++) {
        const ScenePoint& curr = globalscenepoint[i];
        fobj<<curr.pt.x<<" "<<curr.pt.y<<" "<<curr.pt.z<<" "<<curr.RGB.x<<" "<<curr.RGB.y<<" "<<curr.RGB.z<<" ";
       
        std::vector<int> unionset;
        unionframe(keyframes, curr.img, unionset);
        fobj<<unionset.size()<<" ";
        for (int j = 0; j < unionset.size(); j++) {
            int index = imgindex(keyframes, unionset[j]);
            assert(index>=0);
            
            fobj<<index<<" "<<curr.feature[unionset[j]]<<" "<<curr.location[unionset[j]].x<<" "<<curr.location[unionset[j]].y<<" ";
            
        }
            
            
            
        
    }
}