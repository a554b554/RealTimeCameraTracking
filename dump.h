//
//  dump.h
//  RCT
//
//  Created by DarkTango on 3/27/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __RCT__dump__
#define __RCT__dump__

#include <iostream>
#include <vector>
#include "load.h"
#include "Frame.h"
#include "ScenePoint.h"




void unionframe(const std::vector<int>& keyframes, const std::vector<int>& imgs, std::vector<int>& unionframe);


void savedata(const std::vector<Frame>& globalframe, const std::vector<ScenePoint>& globalscenepoint, const std::vector<int>& keyframes, string filename);



#endif /* defined(__RCT__dump__) */
