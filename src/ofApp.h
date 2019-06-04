#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxOpenCv.h"
#include "ofxSyphon.h"
#include "ofxKinect.h"
#include "ofGstVideoPlayer.h"
#include "ofGstUtils.h"
#include "ofxGrt.h"
#include "ofxGui.h"
#include "ofxOsc.h"
#include "ofxCcvThreaded.h"

// where to send osc messages by default
#define DEFAULT_OSC_DESTINATION "localhost"
#define DEFAULT_OSC_ADDRESS "/wek/outputs"
#define DEFAULT_OSC_PORT 12000
#define DEFAULT_DEVICE_ID 0
#define SIZE_INPUT_VECTOR 4096

#define RELEASE
#define colAvgNum 15

class GestureRecognitionPipelineThreaded : public ofThread, public GestureRecognitionPipeline {
public:
    void startTraining(RegressionData *trainingData);
    void threadedFunction();
    RegressionData *trainingData;
    bool training=false;
    bool success=false;
};



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
    
    
    //////////////// WEKINATOR
    
    void updateParameters();
    void setupRegressor();
    void addSlider();
    void train();
    void setMLColor(float circleOpacity);
    
    void setupOSC();
    void sendOSC();
    
    void save();
    void load();
    void clear();
    
    void eSlider(float & v);
    void eChangeOscSettings();
    
    void exit();
    
    // input/cv
    //ofVideoGrabber cam;
    ofxCcvThreaded ccv;
    vector<float> featureEncoding;
    
    // learning
    RegressionData trainingData;
    GestureRecognitionPipelineThreaded pipeline;
    GRT::VectorFloat targetVector;
    vector<ofParameter<float> > values;
    vector<float> targetValues;
    bool isTraining;
    
    // draing/ui
    string infoText;
    ofTrueTypeFont largeFont;
    ofTrueTypeFont smallFont;
    ofTrueTypeFont hugeFont;
    
    // parameters
    ofParameter<float> lerpAmt;
    ofParameter<int> maxEpochs, numHiddenNeurons;
    ofParameter<string> gOscDestination, gOscAddress, gOscPort, gDeviceId;
    
    // gui
    ofxPanel gui, guiSliders;
    ofParameterGroup gOscSettings, gTraining;
    ofxButton bTrain, bSave, bLoad, bClear, bAddSlider, bOscSettings;
    ofxToggle tRecord, tPredict;
    
    // osc
    ofxOscSender sender;
    string oscDestination, oscAddress;
    int oscPort;
  
    //other
    ofFbo fbo2;
    ofColor colorOne;
    float colAvg[colAvgNum] ;
};
