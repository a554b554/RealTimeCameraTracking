#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string.h>
#include <algorithm>

using namespace cv;
using namespace std;

#include <thread>
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
#include <fstream>

vector<Mat> images;
vector<string> images_name;


enum{
    MODE_CALIBRATION,
    MODE_OFFLINE,
    MODE_ONLINE,
    MODE_DOWNSAMPLE,
    MODE_TEST
};

int main(int ac, char** av) {
    
    
    //////////////////////////////
    int mode = MODE_ONLINE;
    const string basepath = "./desktop";
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
        Mat odR,odT;
        int ct = 0;
        for (int i = 0; i < filelist.size(); i++) {
            int64 t1 = getTickCount();
            std::vector<int> candi;
            Mat test = imread(basepath+"/online/"+filelist[i]);
            if (test.empty()) {
                break;
            }
        
            //imshow("a",test);
            Frame t_frame;
            tree.cvtFrame(test, t_frame);
            tree.candidateKeyframeSelection2(t_frame, candi, 4);
            cout<<"Frame: "<<i<<endl;
            printmatchinfo(candi);
            std::vector<std::vector<DMatch>> matches(candi.size());
            std::vector<std::vector<DMatch>> poolmatches(tree.onlinepool.size());
            tree.twoPassMatching(candi, t_frame, matches);
            tree.matchWithOnlinepool(t_frame, poolmatches);
            if(tree.matchsize(matches)<25&&tree.matchsize(poolmatches)<25){
                continue;
            }
           // isgood = tree.ordinarymatching(candi, t_frame, matches);
            Mat rvec,tvec,outimg,out2,rv2,tv2;
            tree.calibrate(t_frame, matches, candi, poolmatches, rvec, tvec);
            tree.rendering(t_frame, rvec, tvec, outimg);
            imwrite(basepath+"/output/"+toString(ct)+".jpg", outimg);
            ct++;
            tree.updateonlinepool(candi, matches, poolmatches, t_frame);
            cout<<"fps: "<<getTickFrequency()/(getTickCount() - t1)<<endl;
            waitKey(1);
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
        string path = basepath+"/origindata/";
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
                //rotateimg(tmp);
                imwrite(basepath+"/offline/"+filename, tmp);
            }
        }
    }
    else if (mode == MODE_TEST){
        fstream ff;
        ff.open("0000.sift");
        char b;
        while(1){
            ff >> b;
            cout<<b;
        }
    }
    return 0;
}















