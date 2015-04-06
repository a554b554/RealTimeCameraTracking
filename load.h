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


#endif /* defined(__RCT__load__) */

void load(const string basepath, std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint);
void load2(const char* filename, std::vector<Frame>& keyframes, std::vector<ScenePoint>& scenepoints, const std::vector<int>& key);


void computeAttribute(std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint);
void computeAttribute2(std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint);


void calculateDescriptor(std::vector<Frame>& globalframe, std::vector<ScenePoint>& globalscenepoint);


//for debug
void draw(const Frame& frame, string windowname);

void drawnativekeypoints(const Frame& frame, string windowname);

void drawmatch(Frame& frame, string windowname, int type);

void drawmatch2(Frame& frame1, Frame& frame2, string windowname);


string toString(int a);

void meanMat(const std::vector<Mat>& inputmat, Mat& output);
void gettime(int64& t0);
void showallframe(const std::vector<Frame>& frameset);