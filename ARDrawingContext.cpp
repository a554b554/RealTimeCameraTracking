 //
//  ARDrawingContext.cpp
//  RCT
//
//  Created by DarkTango on 4/10/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "ARDrawingContext.h"
#include <OpenGL/gl.h>


void ARDrawingContextDrawCallback(void* param){
    ARDrawingContext* ctx = static_cast<ARDrawingContext*>(param);
    if (ctx) {
        ctx->draw();
    }
}

ARDrawingContext::ARDrawingContext(const string windowname ,const Mat& intrinsic, const Mat& rvec, const Mat& tvec,const Frame& onlineframe):textureInitialized(false),intrinsic(intrinsic),rvec(rvec),tvec(tvec),onlineframe(onlineframe)
{
    ARWindowName = windowname;
    cv::namedWindow(windowname, cv::WINDOW_OPENGL);
    
    //initial
    cv::setOpenGlContext(ARWindowName);
    cv::setOpenGlDrawCallback(ARWindowName, ARDrawingContextDrawCallback, this);
}

void ARDrawingContext::draw(){
    //clear entire screen
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    //render background
    drawCameraFrame();
    drawAugmentScene();
}

void ARDrawingContext::drawAugmentScene(){
    GLfloat projectionMatrix[16];
    int w = onlineframe.img.cols;
    int h = onlineframe.img.rows;
    buildProjectionMatrix(w, h, projectionMatrix);
    
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(projectionMatrix);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    if (!rvec.empty()) {
        GLfloat mode[16];
        Mat rotation;
        Rodrigues(rvec, rotation);
        mode[0] = rotation.at<double>(0,0);
        mode[1] = rotation.at<double>(1,0);
        mode[2] = rotation.at<double>(2,0);
        mode[3] = 1;
        mode[4] = rotation.at<double>(0,1);
        mode[5] = rotation.at<double>(1,1);
        mode[6] = rotation.at<double>(2,1);
        mode[7] = 1;
        mode[8] = rotation.at<double>(0,2);
        mode[9] = rotation.at<double>(1,2);
        mode[10] = rotation.at<double>(2,2);
        mode[11] = 1;
        mode[12] = tvec.at<double>(0,0);
        mode[13] = tvec.at<double>(1,0);
        mode[14] = tvec.at<double>(2,0);
        mode[15] = 1;
        glLoadMatrixf(mode);
        drawCoordinateAxis();
        drawCubeModel();
    }
}

void ARDrawingContext::buildProjectionMatrix(int screen_width, int screen_height, GLfloat* projectionMatrix)
{
    float nearPlane = 0.01f;  // Near clipping distance
    float farPlane  = 100.0f;  // Far clipping distance
    
    // Camera parameters
    float f_x = intrinsic.at<float>(0,0); // Focal length in x axis
    float f_y = intrinsic.at<float>(1,1); // Focal length in y axis (usually the same?)
    float c_x = intrinsic.at<float>(0,2); // Camera primary point x
    float c_y = intrinsic.at<float>(1,2); // Camera primary point y
    
    projectionMatrix[0] = -2.0f * f_x / screen_width;
    projectionMatrix[1] = 0.0f;
    projectionMatrix[2] = 0.0f;
    projectionMatrix[3] = 0.0f;
    
    projectionMatrix[4] = 0.0f;
    projectionMatrix[5] = 2.0f * f_y / screen_height;
    projectionMatrix[6] = 0.0f;
    projectionMatrix[7] = 0.0f;
    
    projectionMatrix[8] = 2.0f * c_x / screen_width - 1.0f;
    projectionMatrix[9] = 2.0f * c_y / screen_height - 1.0f;
    projectionMatrix[10] = -( farPlane + nearPlane) / ( farPlane - nearPlane );
    projectionMatrix[11] = -1.0f;
    
    projectionMatrix[12] = 0.0f;
    projectionMatrix[13] = 0.0f;
    projectionMatrix[14] = -2.0f * farPlane * nearPlane / ( farPlane - nearPlane );
    projectionMatrix[15] = 0.0f;
}

void ARDrawingContext::drawCoordinateAxis()
{
    static float lineX[] = {0,0,0,1,0,0};
    static float lineY[] = {0,0,0,0,1,0};
    static float lineZ[] = {0,0,0,0,0,1};
    
    glLineWidth(2);
    
    glBegin(GL_LINES);
    
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3fv(lineX);
    glVertex3fv(lineX + 3);
    
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3fv(lineY);
    glVertex3fv(lineY + 3);
    
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3fv(lineZ);
    glVertex3fv(lineZ + 3);
    
    glEnd();
}

void ARDrawingContext::drawCubeModel()
{
    static const GLfloat LightAmbient[]=  { 0.25f, 0.25f, 0.25f, 1.0f };    // Ambient Light Values
    static const GLfloat LightDiffuse[]=  { 0.1f, 0.1f, 0.1f, 1.0f };    // Diffuse Light Values
    static const GLfloat LightPosition[]= { 0.0f, 0.0f, 2.0f, 1.0f };    // Light Position
    
    glPushAttrib(GL_COLOR_BUFFER_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_LIGHTING_BIT | GL_POLYGON_BIT);
    
    glColor4f(0.2f,0.35f,0.3f,0.75f);         // Full Brightness, 50% Alpha ( NEW )
    glBlendFunc(GL_ONE,GL_ONE_MINUS_SRC_ALPHA);       // Blending Function For Translucency Based On Source Alpha
    glEnable(GL_BLEND);
    
    glShadeModel(GL_SMOOTH);
    
    glEnable(GL_LIGHTING);
    glDisable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);
    glLightfv(GL_LIGHT1, GL_POSITION, LightPosition);
    glEnable(GL_COLOR_MATERIAL);
    
    glScalef(0.25,0.25, 0.25);
    glTranslatef(0,0, 1);
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glBegin(GL_QUADS);
    // Front Face
    glNormal3f( 0.0f, 0.0f, 1.0f);    // Normal Pointing Towards Viewer
    glVertex3f(-1.0f, -1.0f,  1.0f);  // Point 1 (Front)
    glVertex3f( 1.0f, -1.0f,  1.0f);  // Point 2 (Front)
    glVertex3f( 1.0f,  1.0f,  1.0f);  // Point 3 (Front)
    glVertex3f(-1.0f,  1.0f,  1.0f);  // Point 4 (Front)
    // Back Face
    glNormal3f( 0.0f, 0.0f,-1.0f);    // Normal Pointing Away From Viewer
    glVertex3f(-1.0f, -1.0f, -1.0f);  // Point 1 (Back)
    glVertex3f(-1.0f,  1.0f, -1.0f);  // Point 2 (Back)
    glVertex3f( 1.0f,  1.0f, -1.0f);  // Point 3 (Back)
    glVertex3f( 1.0f, -1.0f, -1.0f);  // Point 4 (Back)
    // Top Face
    glNormal3f( 0.0f, 1.0f, 0.0f);    // Normal Pointing Up
    glVertex3f(-1.0f,  1.0f, -1.0f);  // Point 1 (Top)
    glVertex3f(-1.0f,  1.0f,  1.0f);  // Point 2 (Top)
    glVertex3f( 1.0f,  1.0f,  1.0f);  // Point 3 (Top)
    glVertex3f( 1.0f,  1.0f, -1.0f);  // Point 4 (Top)
    // Bottom Face
    glNormal3f( 0.0f,-1.0f, 0.0f);    // Normal Pointing Down
    glVertex3f(-1.0f, -1.0f, -1.0f);  // Point 1 (Bottom)
    glVertex3f( 1.0f, -1.0f, -1.0f);  // Point 2 (Bottom)
    glVertex3f( 1.0f, -1.0f,  1.0f);  // Point 3 (Bottom)
    glVertex3f(-1.0f, -1.0f,  1.0f);  // Point 4 (Bottom)
    // Right face
    glNormal3f( 1.0f, 0.0f, 0.0f);    // Normal Pointing Right
    glVertex3f( 1.0f, -1.0f, -1.0f);  // Point 1 (Right)
    glVertex3f( 1.0f,  1.0f, -1.0f);  // Point 2 (Right)
    glVertex3f( 1.0f,  1.0f,  1.0f);  // Point 3 (Right)
    glVertex3f( 1.0f, -1.0f,  1.0f);  // Point 4 (Right)
    // Left Face
    glNormal3f(-1.0f, 0.0f, 0.0f);    // Normal Pointing Left
    glVertex3f(-1.0f, -1.0f, -1.0f);  // Point 1 (Left)
    glVertex3f(-1.0f, -1.0f,  1.0f);  // Point 2 (Left)
    glVertex3f(-1.0f,  1.0f,  1.0f);  // Point 3 (Left)
    glVertex3f(-1.0f,  1.0f, -1.0f);  // Point 4 (Left)
    glEnd();
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor4f(0.2f,0.65f,0.3f,0.35f); // Full Brightness, 50% Alpha ( NEW )
    glBegin(GL_QUADS);
    // Front Face
    glNormal3f( 0.0f, 0.0f, 1.0f);    // Normal Pointing Towards Viewer
    glVertex3f(-1.0f, -1.0f,  1.0f);  // Point 1 (Front)
    glVertex3f( 1.0f, -1.0f,  1.0f);  // Point 2 (Front)
    glVertex3f( 1.0f,  1.0f,  1.0f);  // Point 3 (Front)
    glVertex3f(-1.0f,  1.0f,  1.0f);  // Point 4 (Front)
    // Back Face
    glNormal3f( 0.0f, 0.0f,-1.0f);    // Normal Pointing Away From Viewer
    glVertex3f(-1.0f, -1.0f, -1.0f);  // Point 1 (Back)
    glVertex3f(-1.0f,  1.0f, -1.0f);  // Point 2 (Back)
    glVertex3f( 1.0f,  1.0f, -1.0f);  // Point 3 (Back)
    glVertex3f( 1.0f, -1.0f, -1.0f);  // Point 4 (Back)
    // Top Face
    glNormal3f( 0.0f, 1.0f, 0.0f);    // Normal Pointing Up
    glVertex3f(-1.0f,  1.0f, -1.0f);  // Point 1 (Top)
    glVertex3f(-1.0f,  1.0f,  1.0f);  // Point 2 (Top)
    glVertex3f( 1.0f,  1.0f,  1.0f);  // Point 3 (Top)
    glVertex3f( 1.0f,  1.0f, -1.0f);  // Point 4 (Top)
    // Bottom Face
    glNormal3f( 0.0f,-1.0f, 0.0f);    // Normal Pointing Down
    glVertex3f(-1.0f, -1.0f, -1.0f);  // Point 1 (Bottom)
    glVertex3f( 1.0f, -1.0f, -1.0f);  // Point 2 (Bottom)
    glVertex3f( 1.0f, -1.0f,  1.0f);  // Point 3 (Bottom)
    glVertex3f(-1.0f, -1.0f,  1.0f);  // Point 4 (Bottom)
    // Right face
    glNormal3f( 1.0f, 0.0f, 0.0f);    // Normal Pointing Right
    glVertex3f( 1.0f, -1.0f, -1.0f);  // Point 1 (Right)
    glVertex3f( 1.0f,  1.0f, -1.0f);  // Point 2 (Right)
    glVertex3f( 1.0f,  1.0f,  1.0f);  // Point 3 (Right)
    glVertex3f( 1.0f, -1.0f,  1.0f);  // Point 4 (Right)
    // Left Face
    glNormal3f(-1.0f, 0.0f, 0.0f);    // Normal Pointing Left
    glVertex3f(-1.0f, -1.0f, -1.0f);  // Point 1 (Left)
    glVertex3f(-1.0f, -1.0f,  1.0f);  // Point 2 (Left)
    glVertex3f(-1.0f,  1.0f,  1.0f);  // Point 3 (Left)
    glVertex3f(-1.0f,  1.0f, -1.0f);  // Point 4 (Left)
    glEnd();
    
    glPopAttrib();
}

void ARDrawingContext::drawCameraFrame()
{
    // Initialize texture for background image
    if (!textureInitialized)
    {
        glGenTextures(1, &backgroundTextureID);
        glBindTexture(GL_TEXTURE_2D, backgroundTextureID);
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        textureInitialized = true;
    }
    
    int w = onlineframe.img.cols;
    int h = onlineframe.img.rows;
    
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, backgroundTextureID);
    
    // Upload new texture data:
    if (onlineframe.img.channels() == 3)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_BGR_EXT, GL_UNSIGNED_BYTE, onlineframe.img.data);
    else if(onlineframe.img.channels() == 4)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, onlineframe.img.data);
    else if (onlineframe.img.channels()==1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, onlineframe.img.data);
    
    const GLfloat bgTextureVertices[] = { 0, 0, static_cast<GLfloat>(w), 0, 0, static_cast<GLfloat>(h), static_cast<GLfloat>(w), static_cast<GLfloat>(h) };
    const GLfloat bgTextureCoords[]   = { 1, 0, 1, 1, 0, 0, 0, 1 };
    const GLfloat proj[]              = { 0, -2.f/w, 0, 0, -2.f/h, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1 };
    
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(proj);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, backgroundTextureID);
    
    // Update attribute values.
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    
    glVertexPointer(2, GL_FLOAT, 0, bgTextureVertices);
    glTexCoordPointer(2, GL_FLOAT, 0, bgTextureCoords);
    
    glColor4f(1,1,1,1);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);
}