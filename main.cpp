#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Common.h"
#include "FeatureMatching.h"

#include <iostream>
#include <string.h>

#include "Distance.h"
#include "MultiCameraPnP.h"
#include <dirent.h>
using namespace cv;
using namespace std;

#include "../3rdparty/SSBA-3.0/Math/v3d_linear.h"
#include "../3rdparty/SSBA-3.0/Base/v3d_vrmlio.h"
#include "../3rdparty/SSBA-3.0/Geometry/v3d_metricbundle.h"



#include "OfflineModule.h"
#include "load.h"
#include "VocabularyTree.h"
#include <vector>
#include <sstream>
#include "dump.h"
#include "voctree2.h"



vector<Mat> images;
vector<string> images_name;




int main(int ac, char** av) {
    
    /*{
    std::vector<Frame> inputframe;
    vector<ScenePoint> inputpoint;
    cout<<"loading..."<<endl;
    load("./campusSFM/campusdata.nvm", inputframe, inputpoint);
   
    
    
    for (int i = 0; i < inputframe.size(); i++) {
        cout<<"keypoint: "<<inputframe[i].keypoint.size()<<endl;
        cout<<"pos: "<<inputframe[i].pos.size()<<endl;
        cout<<"featuresize: "<<inputframe[i].featuresize<<endl;
        cout<<"scenepoint: "<<inputframe[i].scenepoint.size()<<endl;
        for (int j = 0; j < inputframe[i].pos.size(); j++) {
            cout<<inputframe[i].pos[j]<<" ";
        }
    }
    
    std::vector<int> test;
    for (int i = 0; i < inputframe.size(); i++) {
        cout<<"com: "<<i<<" "<<CompletenessTerm(inputframe, test, inputpoint)<<endl;
        test.push_back(i);
    }
    
    test.clear();
    
    for (int i = 0; i < inputframe.size(); i++) {
        cout<<"redu: "<<i<<" "<<Redundancy(inputframe, test, inputpoint)<<endl;
        test.push_back(i);
    }
    return 0;
    
   
    //imshow("d", inputframe[10].dogimg);
    //drawmatch(inputframe[30], "1", 2);
    //drawmatch2(inputframe[30], inputframe[2], "2");
    
    
    //calculateDescriptor(inputframe, inputpoint);
    //draw(inputframe[0], "1");
    //drawnativekeypoints(inputframe[0], "2");
    //waitKey(0);
    

    cout<<"loading complete!"<<endl;
    
    
    cout<<"computing attribute..."<<endl;
    computeAttribute(inputframe, inputpoint);
    cout<<"compute complete!"<<endl;
    for (int i = 0 ; i < 15; i++) {
        cout<<norm(inputframe[inputpoint[i].img[1]].descriptor.row(8), inputframe[inputpoint[i].img[0]].descriptor.row(0))<<endl;
    }
    
    
    
    
    std::vector<int> outputkeyframe;
    cout<<"keyframe selecting..."<<endl;
  //  KeyframeSelection(inputframe, inputpoint, outputkeyframe);
    fakeKeyFrameSelection(1, outputkeyframe);
    
    cout<<"selection complete!"<<endl;
    cout<<"keyframe size:"<<outputkeyframe.size();
    

    cout<<"saving data..."<<endl;
    savedata(inputframe, inputpoint, outputkeyframe, "./campusSFM/keyframedata");
    cout<<"saving complete!"<<endl;
    }*/

    /*for (int i = 0; i < outputkeyframe.size(); i++) {
        string window;
        window = toString(outputkeyframe[i]);
        imshow(window, inputframe[outputkeyframe[i]].img);
    }*/
    
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
    //tree.init(10, 5);
    
    std::vector<int> candi;
    
    Mat test = imread("./campusSFM/0402.jpg");
    Frame t_frame;
    tree.cvtFrame(test, t_frame);
    //tree.candidateKeyframeSelection(t_frame, candi, 4);
    candi.push_back(18);
    candi.push_back(45);
    candi.push_back(38);
    candi.push_back(40);
    std::vector<DMatch> match;
    tree.matching(candi, t_frame, match);
    
    imshow("orin", test);
    for (int i = 0; i < candi.size(); i++) {
        string name;
        name = toString(i);
        imshow(name, keyframes[candi[i]].img);
    }
    waitKey(0);
    
    return 0;
}















