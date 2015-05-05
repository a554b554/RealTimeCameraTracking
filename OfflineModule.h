//
//  OfflineModule.h
//  RCT
//
//  Created by DarkTango on 3/15/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//
#pragma once
#ifndef __RCT__OfflineModule__
#define __RCT__OfflineModule__


#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <vector>
#include "Frame.h"
#include "ScenePoint.h"
#define MIN_TRACK 5
// only input frame is reference, other varible like outputkeyframe is index.

void KeyframeSelection(const std::vector<Frame>& inputframe, const std::vector<ScenePoint>& inputscenepoint
                       ,std::vector<int>& outputkeyframe);
double CompletenessTerm(const std::vector<Frame>& inputframe, const std::vector<int>& candidateframe, const std::vector<ScenePoint>& inputscenepoint);
double Redundancy(const std::vector<Frame>& inputframe, const std::vector<int>& candidateframe, const std::vector<ScenePoint>& inputscenepoint);
double FeatureDensity(const ScenePoint& pt, const std::vector<Frame>& inputframe);
//uchar DoG(const Mat& img, Point2f position);
void fakeKeyFrameSelection(std::vector<int>& keyframes, string basepath);
int countscnenpoint(const std::vector<ScenePoint>& scenepoint, int threshold);
#endif /* defined(__RCT__OfflineModule__) */
