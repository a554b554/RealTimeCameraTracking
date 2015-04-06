#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string.h>


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

vector<Mat> images;
vector<string> images_name;


enum{
    MODE_CALIBRATION,
    MODE_OFFLINE,
    MODE_ONLINE
};

int main(int ac, char** av) {
    int mode = MODE_CALIBRATION;
    const string basepath = "./mydata1";
    if (mode == MODE_CALIBRATION) {
        startcalibration(basepath);
        return 0;
    }
    else if (mode == MODE_OFFLINE){
         std::vector<Frame> inputframe;
         vector<ScenePoint> inputpoint;
         cout<<"loading..."<<endl;
         load(basepath, inputframe, inputpoint);
    }
    else if (mode == MODE_ONLINE){
        // load for online module.
        std::vector<Frame> keyframes;
        std::vector<ScenePoint> scenepoints;
        std::vector<int> outputkeyframe;
        fakeKeyFrameSelection(outputkeyframe);
        load2("./campusSFM/campusdata.nvm", keyframes, scenepoints,outputkeyframe);
        //showallframe(keyframes);
        computeAttribute2(keyframes, scenepoints);
        
        
        //begin online module
        node root;
        VocTree tree(keyframes, scenepoints, root);
        tree.init(10, 5);
        
        namedWindow("show");
        int count = 100;
        while(1){
            std::vector<int> candi;
            
            Mat test = imread("./campusSFM/0"+toString(count)+".jpg");
            imshow("a",test);
            count++;
            if (test.empty()) {
                break;
            }
            Frame t_frame;
            tree.cvtFrame(test, t_frame);
            tree.candidateKeyframeSelection(t_frame, candi, 4);
            /*candi.push_back(18);
             candi.push_back(45);
             candi.push_back(38);
             candi.push_back(40);*/
            std::vector<DMatch> match;
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
            cout<<"rendering cost: "<<(getTickCount()-t0)/getTickFrequency()<<endl;
            
            /*imshow("orin", test);
             for (int i = 0; i < candi.size(); i++) {
             string name;
             name = toString(i);
             imshow(name, keyframes[candi[i]].img);
             }*/
            waitKey(0);
            //  destroyWindow("show");
        }
    }
    return 0;
}















