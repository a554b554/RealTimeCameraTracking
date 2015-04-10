#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string.h>
#include <algorithm>

using namespace cv;
using namespace std;



#include "OfflineModule.h"
#include "load.h"
#include "VocabularyTree.h"
#include <vector>
#include <sstream>
#include "dump.h"
#include "voctree2.h"
#include <pcl/common/poses_from_matches.h>
#include "cameraCalibration.h"
#include <dirent.h>

vector<Mat> images;
vector<string> images_name;


enum{
    MODE_CALIBRATION,
    MODE_OFFLINE,
    MODE_ONLINE,
    MODE_DOWNSAMPLE
};

int main(int ac, char** av) {
    

    //////////////////////////////
    int mode = MODE_ONLINE;
    const string basepath = "./myindoor2";
    if (mode == MODE_CALIBRATION) {
        startcalibration(basepath);
        return 0;
    }
    else if (mode == MODE_OFFLINE){
        std::vector<Frame> inputframe;
        vector<ScenePoint> inputpoint;
        cout<<"loading..."<<endl;
        load(basepath, inputframe, inputpoint);
        computeAttribute(inputframe, inputpoint);
        std::vector<int> keyframes;
        KeyframeSelection(inputframe, inputpoint, keyframes);
        savekeyframe(keyframes,basepath);
    }
    else if (mode == MODE_ONLINE){
        // load for online module.
        std::vector<Frame> keyframes;
        std::vector<ScenePoint> scenepoints;
        std::vector<int> outputkeyframe;
        fakeKeyFrameSelection(outputkeyframe,basepath);
        load2(basepath, keyframes, scenepoints,outputkeyframe);
        //showallframe(keyframes);
        computeAttribute2(keyframes, scenepoints);
        //begin online module
        node root;
        VocTree tree(keyframes, scenepoints, root);
        tree.loadCameraMatrix(basepath);
        tree.init(10, 5);
        std::vector<string> filelist;
        loadonlineimglist(basepath, filelist);
        sort(filelist.begin(), filelist.end());
        namedWindow("show");
        for (int i = 0; i < filelist.size(); i++) {
            std::vector<int> candi;
            Mat test = imread(basepath+"/online/"+filelist[i]);
            if (test.empty()) {
                break;
            }
            imshow("a",test);
            Frame t_frame;
            tree.cvtFrame(test, t_frame);
            tree.candidateKeyframeSelection(t_frame, candi, 4);
            std::vector<std::vector<DMatch>> matches(candi.size());
            tree.matching(candi, t_frame, matches);
         //   tree.ordinarymatching(candi, t_frame, matches);
            Mat rvec,tvec,outimg;
            tree.calibrate(t_frame, matches, candi, rvec, tvec);
            tree.rendering(t_frame, rvec, tvec, outimg);
            
            waitKey(30);
        }
            /*std::vector<DMatch> match;
            int64 t0 = getTickCount();
            tree.matching(candi, t_frame, match);
            cout<<"matching cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
            t0 = getTickCount();
            Mat rvec,tvec;
            tree.calibrate(t_frame, match, rvec, tvec);
            cout<<"calibration cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
            t0 = getTickCount();
            Mat show;
            tree.rendering(t_frame, rvec, tvec, show);
            cout<<"rendering cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;*/
        
        
            /*imshow("orin", test);
             for (int i = 0; i < candi.size(); i++) {
             string name;
             name = toString(i);
             imshow(name, keyframes[candi[i]].img);
             }*/
            waitKey(0);
            //  destroyWindow("show");
        
    }
    else if (mode == MODE_DOWNSAMPLE){
        DIR *dir;
        struct dirent *ent;
        string path = basepath+"/origindata/offline";
        if ((dir = opendir(path.c_str())) != NULL) {
            while ((ent = readdir(dir)) != NULL) {
                string imgpath(ent->d_name);
                imgpath = path+"/"+imgpath;
                string filename(ent->d_name);
                Mat tmp = imread(imgpath);
                if (tmp.empty()) {
                    continue;
                }
                resize(tmp, tmp, Size(640,360));
                imwrite(basepath+"/offline/"+filename, tmp);
            }
        }
    }
    return 0;
}















