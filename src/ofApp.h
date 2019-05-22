#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxOpenCv.h"
#include "ofxSyphon.h"
#include "ofxKinect.h"
#include "ofGstVideoPlayer.h"
#include "ofGstUtils.h"

class ofApp : public ofBaseApp {
public:
	void setup();
	void update();
	void draw();
    void keyPressed(int key);
	
	ofVideoGrabber cam;
	ofPixels previous;
    ofPixels camPix;
    ofImage camImg;
	ofImage diff;
	
	// a scalar is like an ofVec4f but normally used for storing color information
	cv::Scalar diffMean;
    int camWidth;  // try to grab at this size.
    int camHeight;
    ofTexture tex;
    ofImage texImg;
    ofFbo fbo;
    
    int ii;
    int jj;
    int w;
    int h;
    float temp1;
    int temp2;
    float theta;
    ofPixels ringPixels;
    ofTexture ringTex;
    ofImage ringImg;
    float contourScale;
    float contourPersistance;
    float circleOpacity;
    
    ofxKinect kinect;
    
    ofxCvColorImage colorImg;
    
    ofxCvGrayscaleImage grayImage; // grayscale depth image
    ofxCvGrayscaleImage grayThreshNear; // the near thresholded image
    ofxCvGrayscaleImage grayThreshFar; // the far thresholded image
    
    int nearThreshold;
    int farThreshold;
    
    ofxCv::ContourFinder contourFinder;
    ofxCv::ContourFinder contourFinderFull;
    
    float threshold;
    ofxCv::ObjectFinder faceFinder;
    ofVideoPlayer movie;

    ofxSyphonServer mainOutputSyphonServer;
    ofxSyphonServer individualTextureSyphonServer;
    float intensity;
    ofxSyphonClient mClient;
    ofColor red;
    
    ofGstVideoUtils gstv;
    
};
