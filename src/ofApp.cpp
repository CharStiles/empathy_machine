#include "ofApp.h"

using namespace ofxCv;
using namespace cv;

/////machine learning shit

//--------------------------------------------------------------
void GestureRecognitionPipelineThreaded::startTraining(RegressionData *trainingData) {
    this->trainingData = trainingData;
    startThread();
    training = true;
}

//--------------------------------------------------------------
void GestureRecognitionPipelineThreaded::threadedFunction() {
    while(isThreadRunning()) {
        if(lock()) {
            success = train(*trainingData);
            training = false;
            unlock();
            stopThread();
        } else {
            ofLogWarning("threadedFunction()") << "Unable to lock mutex.";
        }
    }
}

//-------

void ofApp::setup() {
    camWidth = 640;  // try to grab at this size.
    camHeight = 480;
    red = ofColor(255,0,0,255);
    ofLogVerbose();
    ofColor colorOne(255,255,255,255);
    
    for (int i = 0 ; i < colAvgNum; i ++){
        
        colAvg[i]=1.0;
    }
	//vofSetVerticalSync(true);
    //ofBackground(0);
    ofSetBackgroundAuto(false);
    ofSetVerticalSync(false);
    ofEnableAlphaBlending();
    circleOpacity = 255;
    contourScale = 17; //
    contourPersistance = 30; // 6
    ofSetColor(0,255);
    ofPixelFormat pf = OF_PIXELS_RGB;
    ofRect(0,0,ofGetScreenWidth(),ofGetScreenHeight());
    ringPixels.allocate(camWidth, camHeight, pf);
    ringTex.allocate(camWidth, camHeight, pf);
    tex.allocate(camWidth, camHeight, pf);
    fbo.allocate(camWidth,camHeight,GL_RGBA);
    fbo2.allocate(camWidth,camHeight,GL_RGBA);
    camImg.allocate(camWidth,camHeight,OF_IMAGE_COLOR);
    
    ringImg.allocate(camWidth, camHeight,OF_IMAGE_COLOR);
    texImg.allocate(camWidth, camHeight,OF_IMAGE_COLOR);

   // faceFinder.setup("haarcascade_frontalface_default.xml");
   // faceFinder.setPreset(ObjectFinder::Fast);
    
    //////////////////////////////////////////////////////// ETHERNET CAMERA STUFF
    gstv.setPixelFormat(pf);
    gstv.allocate(camWidth, camHeight, pf);
    gstv.setPipeline("rtspsrc location=rtsp://admin:@192.168.8.192:554/live0.264;stream=0;user=system;pass=system; width=640, height=480,framerate=15/1 gop-size=1 bitrate=20 drop-on-latency=true  latency=0 ! queue2 max-size-buffers=0 ! decodebin ! videoconvert ! videoscale", pf, true, camWidth, camHeight);
    //TODO: Speed this up
    
    gstv.startPipeline();
    gstv.play();

    /// CONTOUR
    contourFinder.setMinAreaRadius(6);
    contourFinder.setMaxAreaRadius(100);
    contourFinder.setThreshold(contourScale);
    // wait for half a second before forgetting something
    contourFinder.getTracker().setPersistence(contourPersistance);// second
    // an object can move up to 32 pixels per frame
    contourFinder.getTracker().setMaximumDistance(32);
    //END CONTOUR

    /// CONTOUR
    contourFinderFull.setMinAreaRadius(6);
    contourFinderFull.setMaxAreaRadius(100);
    contourFinderFull.setThreshold(contourScale);
    // wait for half a second before forgetting something
    contourFinderFull.getTracker().setPersistence(contourPersistance);// second
    // an object can move up to 32 pixels per frame
    contourFinderFull.getTracker().setMaximumDistance(32);
    //END CONTOUR

//    cam.setDeviceID(dID);
//    cam.initGrabber(camWidth, camHeight);

    camPix.allocate(camWidth, camHeight, 3);
	// imitate() will set up previous and diff
	// so they have the same size and type as cam
	imitate(previous, camPix);
	imitate(diff, camPix);
    
    // START SYPHION
    mainOutputSyphonServer.setName("Screen Output");
    // individualTextureSyphonServer.setName("Texture Output");
    mClient.setup();
    
    //using Syphon app Simple Server, found at http://syphon.v002.info/
    mClient.set("","Simple Server");
    //END SYPHON
    
    //////////////////Machine learning shiiiit
 
#ifdef RELEASE
    string ccvPath = ofToDataPath("image-net-2012.sqlite3");
#else
    string ccvPath = ofToDataPath("image-net-2012.sqlite3");
#endif
    
    oscDestination = DEFAULT_OSC_DESTINATION;
    oscAddress = DEFAULT_OSC_ADDRESS;
    oscPort = DEFAULT_OSC_PORT;
    
    // ccv
    ccv.setup(ccvPath);
    if (!ccv.isLoaded()) return;
    ccv.setEncode(true);
    ccv.start();
    
    //GUI
    bTrain.addListener(this, &ofApp::train);
    bSave.addListener(this, &ofApp::save);
    bLoad.addListener(this, &ofApp::load);
    bClear.addListener(this, &ofApp::clear);
    bAddSlider.addListener(this, &ofApp::addSlider);
    bOscSettings.addListener(this, &ofApp::eChangeOscSettings);
    
    gTraining.setName("Training");
    gTraining.add(numHiddenNeurons.set("hidden neurons", 10, 5, 50));
    gTraining.add(maxEpochs.set("epochs", 100, 20, 1000));
    
    gOscSettings.setName("OSC settings");
    gOscSettings.add(gOscDestination.set("IP", oscDestination));
    gOscSettings.add(gOscPort.set("port", ofToString(oscPort)));
    gOscSettings.add(gOscAddress.set("message", oscAddress));
    
    gui.setup();
    gui.setPosition(camWidth,10);
    gui.setName("Convnet regressor");
    gui.add(gDeviceId.set("deviceId", ofToString(DEFAULT_DEVICE_ID)));
    gui.add(gTraining);
    gui.add(tRecord.setup("Record", false));
    gui.add(bClear.setup("Clear training data"));
    gui.add(bTrain.setup("Train"));
    gui.add(tPredict.setup("Predict", false));
    gui.add(bSave.setup("Save model"));
    gui.add(bLoad.setup("Load model"));
    gui.add(lerpAmt.set("Prediction lerp", 0.2, 0.01, 1.0));
    gui.add(gOscSettings);
    gui.add(bOscSettings.setup("change OSC settings"));
    gui.loadFromFile(ofToDataPath("settings_convnetR.xml"));
    tPredict = false;
    
    guiSliders.setup();
    guiSliders.setPosition(camWidth,gui.getHeight()+25);
    guiSliders.setName("Sliders");
    guiSliders.add(bAddSlider.setup("Add Slider"));
    addSlider();
    
    // osc
    setupOSC();

}

//--------------------------------------------------------------
void ofApp::setupOSC() {
    sender.setup(oscDestination, oscPort);
}

//--------------------------------------------------------------
void ofApp::eChangeOscSettings() {
    string input = ofSystemTextBoxDialog("Send OSC to what destination IP", oscDestination);
    bool toSwitchOsc = false;
    if (input != "" && input != oscDestination) {
        oscDestination = input;
        gOscDestination.set(oscDestination);
        toSwitchOsc = true;
    }
    input = ofSystemTextBoxDialog("Send OSC to what destination port", ofToString(oscPort));
    if (ofToInt(input) > 0 && ofToInt(input) != oscPort) {
        oscPort = ofToInt(input);
        gOscPort.set(ofToString(oscPort));
        toSwitchOsc = true;
    }
    input = ofSystemTextBoxDialog("Send OSC with what message address", oscAddress);
    if (input != "" && input != oscAddress) {
        oscAddress = input;
        gOscAddress.set(oscAddress);
    }
    if (toSwitchOsc) {
        setupOSC();
    }
}

//--------------------------------------------------------------
void ofApp::eSlider(float & v) {
    sendOSC();
}

//--------------------------------------------------------------
void ofApp::exit() {
    gui.saveToFile(ofToDataPath("settings_convnetR.xml"));
    ccv.setEncode(false);
    ccv.stop();
    pipeline.stopThread();
}

//--------------------------------------------------------------
void ofApp::addSlider() {
    if (pipeline.getNumTrainingSamples() > 0) {
        ofLog(OF_LOG_ERROR, "Can't add slider, already have training samples");
        return;
    }
    ofParameter<float> newSlider;
    guiSliders.add(newSlider.set("y"+ofToString(1+values.size()), 0.5, 0.0, 1.0));
    newSlider.addListener(this, &ofApp::eSlider);
    values.push_back(newSlider);
    targetValues.resize(values.size());
    trainingData.setInputAndTargetDimensions( SIZE_INPUT_VECTOR, values.size() );
}

//--------------------------------------------------------------
void ofApp::setupRegressor() {
    unsigned int numInputNeurons = trainingData.getNumInputDimensions();
    unsigned int numOutputNeurons = 1; //1 as we are using multidimensional regression
    
    //Initialize the MLP
    MLP mlp;
    mlp.init(numInputNeurons, numHiddenNeurons, numOutputNeurons, Neuron::LINEAR, Neuron::SIGMOID, Neuron::SIGMOID );
    
    //Set the training settings
    mlp.setMaxNumEpochs( maxEpochs ); //This sets the maximum number of epochs (1 epoch is 1 complete iteration of the training data) that are allowed
    mlp.setMinChange( 1.0e-10 ); //This sets the minimum change allowed in training error between any two epochs
    mlp.setLearningRate( 0.01 ); //This sets the rate at which the learning algorithm updates the weights of the neural network
    mlp.setNumRandomTrainingIterations( 1 ); //This sets the number of times the MLP will be trained, each training iteration starts with new random values
    mlp.setUseValidationSet( true ); //This sets aside a small portiion of the training data to be used as a validation set to mitigate overfitting
    mlp.setValidationSetSize( 15 ); //Use 20% of the training data for validation during the training phase
    mlp.setRandomiseTrainingOrder( true ); //Randomize the order of the training data so that the training algorithm does not bias the training
    
    //The MLP generally works much better if the training and prediction data is first scaled to a common range (i.e. [0.0 1.0])
    mlp.enableScaling( true );
    
    pipeline << MultidimensionalRegression(mlp,true);
}

//--------------------------------------------------------------
void ofApp::updateParameters() {
    for (int i=0; i<values.size(); i++) {
        values[i] = ofLerp(values[i], targetValues[i], lerpAmt);
    }
    sendOSC();
}

void ofApp::update() {
    // START WEKINATOR STUFF
    if (!ccv.isLoaded()) {
        ofDrawBitmapString("Network file not found! Check your data folder to make sure it exists.", 20, 20);
        return;
    }
    
    if (isTraining) {
        if (!pipeline.training) {
            infoText = pipeline.success ? "Pipeline trained" : "WARNING: Failed to train pipeline";
            isTraining = false;
           // ofBackground(150);
        } else if (ofGetFrameNum() % 15 == 0) {
            gui.setTextColor(ofColor(ofRandom(255),ofRandom(255),ofRandom(255)));
        }
    }
    else if (tPredict) {
        updateParameters();
    }
    ////////////END WEKINATOR STUFFFFF
    
    gstv.update();
    if(gstv.isFrameNew()){
        camPix = gstv.getPixels();
        
        if (ccv.isReady()){
            ccv.update(gstv, ccv.numLayers()-1);
        }

        temp1 =(float)((camWidth)/2);
        //faceFinder.update(cam);
		// take the absolute difference of prev and cam and save it inside diff
		absdiff(camPix, previous, diff);
		diff.update();
		
		// like ofSetPixels, but more concise and cross-toolkit
		copy(camPix, previous);
		
		// mean() returns a Scalar. it's a cv:: function so we have to pass a Mat
		diffMean = mean(toCv(diff));
		
		// you can only do math between Scalars,
		// but it's easy to make a Scalar from an int (shown here)
		diffMean *= Scalar(50000000000);
        
        ofPixels & pixels = diff.getPixels();//vidGrabber.getPixels();
        // this turns into tex
        
        //ofPixels pixels;
        //fbo.readToPixels(pixels);
        for(int i = 0; i < contourFinder.size(); i++) {
            ofPoint center = toOf(contourFinder.getCenter(i));
        }
        for (int  j = 0; j < camHeight; j++){
            for(int i = 0; i < camWidth; i++){
                temp2 = ofDist((float)i, (float)j,(float)(camWidth)/2 , (float)(camHeight)/2);
                if(temp2 < temp1 && temp2 > (float)((camHeight)/4)+50){
                    ii = i- (camWidth/2);
                    jj = j- (camHeight/2);
                    if (ii != 0){ // replacement for atan2!
                        if (ii <0    && jj >=0){
                            theta =(PI) + atan((float)jj/(float)ii);
                        }
                        else if (ii >= 0 && jj >= 0){
                            theta = atan((float)jj/(float)ii);
                        }
                        else if (ii < 0 && jj < 0){
                            theta = ((PI) + atan((float)jj/(float)ii));
                        }
                        else{
                            theta = ((PI*2)) + atan((float)jj/(float)ii);
                        }
                    }
                    else{
                        theta = 0;
                    }
                    //c = pixels.getColor(i,j);
                    //                    ofColor cc = pixels.getColor(i,j);
                    //                    cc = ofColor(cc.r,cc.g,cc.b,30);
                    
                    ringPixels.setColor(camWidth*(theta/(2*PI)),temp1 - temp2, (pixels.getColor(i,j)));
                }
                else if (temp2 > temp1){ // outside circle
                    //videoInverted.setColor(i,j,ofColor(0,0,0));
                    ofColor c = pixels.getColor(i,j);
                    //ofColor cc = c.invert();
                    //cc = ofColor(cc.r,cc.g,cc.b,30);
                    pixels.setColor(i,j,c.invert());
                }
//                else if (temp2 < (float)((camHeight)/4)+50){ // outside circle
//                    ofColor c = pixels.getColor(i,j);
//                    pixels.setColor(i,j,c.invert());
//                }
            }
        }
            
    ringTex.loadData(ringPixels); // distorted diff
    ringImg.setFromPixels(ringPixels);
    texImg.setFromPixels(diff);
    tex.loadData(pixels); // diff feed
    camImg.setFromPixels(gstv.getPixels());
    contourFinder.findContours(ringImg);
    contourFinderFull.findContours(texImg);
	}
    else{
        
        return;
    }
    
    /////wek stuff 2
    if ((tRecord||tPredict)&&ccv.hasNewResults()) {
        featureEncoding = ccv.getEncoding();
        VectorFloat inputVector(featureEncoding.size());
        for (int i=0; i<featureEncoding.size(); i++) {
            inputVector[i] =  featureEncoding[i];
        }
        
        if( tRecord ) {
            VectorFloat targetVector(values.size());
            for (int p=0; p<values.size(); p++) {
                targetVector[p] = values[p];
            }
            if( !trainingData.addSample(inputVector, targetVector) ){
                infoText = "WARNING: Failed to add training sample to training data!";
            }
            
        }
        else if( tPredict ){
            if( pipeline.predict( inputVector ) ){
                for (int p=0; p<values.size(); p++) {
                    targetValues[p] = pipeline.getRegressionData()[p];
                }
            }else{
                infoText = "ERROR: Failed to run prediction!";
            }
        }
    }
}

void setMLColor(float circleOpacity){
    

}

void ofApp::draw() {
    fbo2.begin();
    
    if (tPredict){
        float avg = 0;
        colAvg[ofGetFrameNum() % colAvgNum] = targetValues[0];
        for (int i = 0 ; i < colAvgNum; i ++){
            
            avg += colAvg[i];
        }
        avg = avg/ (float)colAvgNum;
        
        colorOne.set(avg * 255,255,avg * 255,circleOpacity);
        if(values.size() > 1){
            colorOne.set(colAvgNum* 255, 255,targetValues[1] * 255,circleOpacity);
        }
        
    }
    
    //ofColor colorOne(255, 255, 255,2555);
    ofColor colorTwo(colorOne.r, colorOne.g, colorOne.b,0);
    
    ofBackgroundGradient(colorOne, colorTwo, OF_GRADIENT_CIRCULAR);
    fbo2.end();
    
    fbo.begin();
    ofSetColor(0,0,0,10);
    ofRectangle(0,0, ofGetWindowWidth() ,ofGetWindowHeight());
    
    ofRectangle(0,0, ofGetWindowWidth() ,ofGetWindowHeight());
    texImg.draw(0,0);
    
    ofSetColor(colorOne);
    for(int j = 0; j < contourFinderFull.size(); j++) {
        
        ofPoint center = toOf(contourFinderFull.getCenter(j));
        ofVec2f velocity = toOf(contourFinderFull.getVelocity(j));
        
        int v = (velocity.x + velocity.y);
        fbo2.draw(center.x - (v*2), center.y - (v*2),5*(velocity.x + velocity.y),(velocity.x + velocity.y)*4);
        fbo2.draw(center.x- (v*2), center.y- (v*2),(velocity.x + velocity.y),(velocity.x + velocity.y)*2);
        
    }

    fbo.end();
    
    ofSetColor(colorOne);
    fbo2.draw(camWidth,0,camWidth,camHeight); // rectangle thats the ML valjuer
    
    ofSetColor(0,10);
    ringTex.draw(0, 0);

    for(int i = 0; i < contourFinder.size(); i++) {
        ofPoint center = toOf(contourFinder.getCenter(i));
        ofVec2f velocity = toOf(contourFinder.getVelocity(i));
//        if (values.size() > 0){
//            ofSetColor(targetValues[0] * 255,255,255,circleOpacity);
//            if(values.size() > 1){
//                ofSetColor(255, targetValues[1] * 255,255,
        
         ofSetColor(colorOne);
        //ofEllipse(center.x, center.y,5*(velocity.x + velocity.y),500);
        fbo2.draw(center.x - (velocity.x + velocity.y), center.y - (velocity.x + velocity.y),2*(velocity.x + velocity.y), 2*(velocity.x + velocity.y));
        fbo2.draw(center.x - 4*(velocity.x + velocity.y), center.y- 4*(velocity.x + velocity.y) - 90,8*(velocity.x + velocity.y), 90);
    }

    
    ofSetColor(255,255,255,10);
    ringImg.draw(0,0);

    ofSetColor(255);
    contourFinder.draw();
    fbo.draw(0, camHeight/2);
    //tex.draw(camWidth, camHeight/2);
    camImg.draw(camWidth, camHeight/2);
    //grayImage.draw(10, 320, 400, 300);
    //tex.draw(0,camHeight/2);
    mClient.draw(50, 50);
    
    mainOutputSyphonServer.publishScreen();
    
    
    /////////WEKINATOR SHTUFF
    //cam.draw(270, 10);
    ofDrawBitmapStringHighlight( "Num Samples: " + ofToString( trainingData.getNumSamples() ), camWidth, gui.getHeight()+10 );
    ofDrawBitmapStringHighlight( infoText, camWidth,gui.getHeight() + 50 );
    
    gui.draw();
    guiSliders.draw();
    ofSetColor(0,255);
    ofDrawBitmapString("(left & right arrows)sensitivity: " + ofToString(contourScale), camWidth + gui.getWidth(), 50);
    ofDrawBitmapString("(up & down arrows)persistance: " + ofToString(contourPersistance), camWidth + gui.getWidth(), 100);
    ofDrawBitmapString("(<>) strength: " + ofToString(circleOpacity), camWidth + gui.getWidth(), 150);
}
void ofApp::keyPressed  (int key){
    
    switch (key) {
        case OF_KEY_LEFT:
            contourScale -= 1;
 
            break;
        case OF_KEY_RIGHT:
            contourScale +=1;

            break;

        case OF_KEY_UP:
            contourPersistance =abs(contourPersistance - 2);
            cout << "\n perst";
            cout << contourPersistance;
            break;
        case OF_KEY_DOWN:
            cout << "\n perst";
            contourPersistance +=2;
            cout << contourPersistance;
            break;
        case ',':
            circleOpacity -=5;
            break;
        case '.':
            circleOpacity +=5;
            break;
    }
    contourFinder.setThreshold(contourScale);
    contourFinderFull.setThreshold(contourScale);
    // wait for half a second before forgetting something
    contourFinder.getTracker().setPersistence(contourPersistance);// second
    contourFinderFull.getTracker().setPersistence(contourPersistance);// second
}

//--------------------------------------------------------------
void ofApp::train() {
    ofLog(OF_LOG_NOTICE, "Training...");
    tRecord = false;
    tPredict = false;
    setupRegressor();
    pipeline.startTraining( &trainingData );
    pipeline.startThread();
    infoText = "Training!! please wait.";
    isTraining = true;
    
    ofLog(OF_LOG_NOTICE, "Done training...");
}

//--------------------------------------------------------------
void ofApp::save() {
    if( trainingData.save( ofToDataPath("TrainingDataConvnetR.grt") ) ){
        infoText = "Training data saved to file";
    } else infoText = "WARNING: Failed to save training data to file";
}

//--------------------------------------------------------------
void ofApp::load() {
    if( trainingData.load( ofToDataPath("TrainingDataConvnetR.grt") ) ){
        infoText = "Training data loaded from file";
        train();
    } else infoText = "WARNING: Failed to load training data from file";
}

//--------------------------------------------------------------
void ofApp::clear() {
    trainingData.clear();
    pipeline.clear();
    infoText = "Training data cleared";
    tPredict = false;
}

//--------------------------------------------------------------
void ofApp::sendOSC() {
    ofxOscMessage m;
    m.setAddress(oscAddress);
    for (int i=0; i<values.size(); i++) {
        m.addFloatArg(values[i]);
    }
    sender.sendMessage(m, false);
}
