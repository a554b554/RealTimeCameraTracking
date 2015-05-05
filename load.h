//
//  load.h
//  RCT
//
//  Created by DarkTango on 3/19/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//
#pragma once
#ifndef __RCT__load__
#define __RCT__load__

#include <iostream>
#include <vector>
#include <string>
#include "Frame.h"
#include "ScenePoint.h"
#include <ctime>
#include <fstream>

#include "OfflineModule.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/ocl.hpp>

void computekeypoint(Frame& frame, const vector<Point2f>& point);
void load(const string basepath, std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint);
void load2(const string filename, std::vector<Frame>& keyframes, std::vector<ScenePoint>& scenepoints, const std::vector<int>& key);


void computeAttribute(std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint);
void computeAttribute2(std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint);
void rotateimg(Mat& img);

void calculateDescriptor(std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint);

void savekeyframe(const std::vector<int>& keyframe,string basepath);

//for debug
void draw(const Frame& frame, string windowname);
void drawnativekeypoints(const Frame& frame, string windowname);
void drawmatch(Frame& frame, string windowname, int type);
void drawmatch2(Frame& frame1, Frame& frame2, string windowname);
void loadonlineimglist(const string basepath, std::vector<string>& filename);

string toString(int a);

void meanMat(const std::vector<Mat>& inputmat, Mat& output);
void gettime(int64& t0);
void showallframe(const std::vector<Frame>& frameset);
void showmatches(const Frame& keyframe, const Frame& onlineframe, const std::vector<DMatch>& matches);
void drawmatchedpoint(const Frame& onlineframe, const std::vector<std::vector<DMatch>>& matches1, const std::vector<std::vector<DMatch>>& matches2);
void drawmatchedpoint(const Frame& onlineframe, const std::vector<std::vector<DMatch>>& matches);

void printmatchinfo(const std::vector<int> candi);
#endif /* defined(__RCT__load__) */