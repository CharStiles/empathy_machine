#include "ofApp.h"

using namespace ofxCv;
using namespace cv;


void ofApp::setup() {
    camWidth = 640;  // try to grab at this size.
    camHeight = 480;
    red = ofColor(255,0,0,255);
	//vofSetVerticalSync(true);
    ofSetBackgroundAuto(false);
    ofSetVerticalSync(false);
    ofEnableAlphaBlending();
    circleOpacity = 100;
    contourScale = 20; //
    contourPersistance = 30; // 6
    
    ringPixels.allocate(camWidth, camHeight, OF_PIXELS_RGB);
    ringTex.allocate(camWidth, camHeight, OF_PIXELS_RGB);
    tex.allocate(camWidth, camHeight, OF_PIXELS_RGB);
    fbo.allocate(camWidth,camHeight,GL_RGBA);
    ringImg.allocate(camWidth, camHeight,OF_IMAGE_COLOR);
    texImg.allocate(camWidth, camHeight,OF_IMAGE_COLOR);

   // faceFinder.setup("haarcascade_frontalface_default.xml");
   // faceFinder.setPreset(ObjectFinder::Fast);

    
    //////////////////////////////////KINECT STUFF
    // enable depth->video image calibration
    //kinect.setRegistration(true);
    
   // kinect.init();
    //kinect.init(true); // shows infrared instead of RGB video image
    //kinect.init(false, false); // disable video image (faster fps)
    
   // kinect.open();        // opens first available kinect
//    colorImg.allocate(kinect.width, kinect.height);
//    grayImage.allocate(kinect.width, kinect.height);
//    grayThreshNear.allocate(kinect.width, kinect.height);
//    grayThreshFar.allocate(kinect.width, kinect.height);
//
//    nearThreshold = 230;
//    farThreshold = 70;
    
    //////////////////////////////////KINECT SETUP STUFF END
    
    vector<ofVideoDevice> devices = cam.listDevices();
    int dID =0;
    for(int i = 0; i < devices.size(); i++){
        if(devices[i].bAvailable){
            ofLogNotice() << devices[i].id << ": " << devices[i].deviceName;
            if(devices[i].deviceName.find("FaceTime") == std::string::npos){
                dID = i; // if its not named facetime def use it
            }
        }else{
            ofLogNotice() << devices[i].id << ": " << devices[i].deviceName << " - unavailable ";
        }
    }
    
    /// CONTOUR
    contourFinder.setMinAreaRadius(1);
    contourFinder.setMaxAreaRadius(100);
    contourFinder.setThreshold(contourScale);
    // wait for half a second before forgetting something
    contourFinder.getTracker().setPersistence(contourPersistance);// second
    // an object can move up to 32 pixels per frame
    contourFinder.getTracker().setMaximumDistance(32);
    //END CONTOUR

    /// CONTOUR
    contourFinderFull.setMinAreaRadius(1);
    contourFinderFull.setMaxAreaRadius(100);
    contourFinderFull.setThreshold(contourScale);
    // wait for half a second before forgetting something
    contourFinderFull.getTracker().setPersistence(contourPersistance);// second
    // an object can move up to 32 pixels per frame
    contourFinderFull.getTracker().setMaximumDistance(32);
    //END CONTOUR
    
    cam.setDeviceID(dID);
    cam.initGrabber(camWidth, camHeight);
	// imitate() will set up previous and diff
	// so they have the same size and type as cam
	imitate(previous, cam);
	imitate(diff, cam);
    
    // START SYPHION
    mainOutputSyphonServer.setName("Screen Output");
    // individualTextureSyphonServer.setName("Texture Output");
    mClient.setup();
    
    //using Syphon app Simple Server, found at http://syphon.v002.info/
    mClient.set("","Simple Server");
    //END SYPHON
    
}

void ofApp::update() {
//
//    kinect.update();
//    if(kinect.isFrameNew()) {
//        grayImage.setFromPixels(kinect.getDepthPixels());
//        grayImage.flagImageChanged();
//    }
	cam.update();
	if(cam.isFrameNew()) {
        temp1 =(float)((camHeight)/2);
        //faceFinder.update(cam);
		// take the absolute difference of prev and cam and save it inside diff
		absdiff(cam, previous, diff);
		diff.update();
		
		// like ofSetPixels, but more concise and cross-toolkit
		copy(cam, previous);
		
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
                temp2 = ofDist((float)i, (float)j,(float)(camWidth)/2, (float)(camHeight)/2);
                if(temp2 < temp1 && temp2 > (float)((camHeight)/4)){
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
                else if (abs(temp2 - temp1)< 5){ // outside circle
                    
                    pixels.setColor(i,j,red);
                }
            }
        }
            
    ringTex.loadData(ringPixels); // distorted diff
    ringImg.setFromPixels(ringPixels);// TODO init this
    texImg.setFromPixels(diff);
    tex.loadData(pixels); // diff feed
    contourFinder.findContours(ringImg);
    contourFinderFull.findContours(texImg);
	}
}

void ofApp::draw() {
    
    ofSetColor(0,0,0,10);
    ofRectangle(0,0, ofGetWindowWidth() ,ofGetWindowHeight());
    fbo.begin();
    ofRectangle(0,0, ofGetWindowWidth() ,ofGetWindowHeight());
    texImg.draw(0,0);
    ofSetColor(255,255,255,255);
    //tex.draw(0,0);
    for(int j = 0; j < contourFinderFull.size(); j++) {
        
        ofPoint center = toOf(contourFinderFull.getCenter(j));
        ofVec2f velocity = toOf(contourFinderFull.getVelocity(j));
        ofSetColor(255,255,255,circleOpacity);
        ofEllipse(center.x, center.y,5*(velocity.x + velocity.y),(velocity.x + velocity.y)*4);
        ofEllipse(center.x, center.y,(velocity.x + velocity.y),(velocity.x + velocity.y)*2);
        
    }
//    for(int i = 0; i < faceFinder.size(); i++) {
//        ofSetColor(255,0,0,255);
//        ofRectangle object = faceFinder.getObjectSmoothed(i);
//        ofDrawRectangle(object);
//    }
    fbo.end();

    ringTex.draw(0, 0);

    for(int i = 0; i < contourFinder.size(); i++) {
        ofPoint center = toOf(contourFinder.getCenter(i));
        ofVec2f velocity = toOf(contourFinder.getVelocity(i));
        ofSetColor(255,255,255,circleOpacity);
        ofEllipse(center.x, center.y,5*(velocity.x + velocity.y),500);
        ofEllipse(center.x, center.y,(velocity.x + velocity.y),100);
    }

    
    ofSetColor(255,255,255,10);
    ringImg.draw(0,0);
    
    ofSetColor(255);
    contourFinder.draw();
    fbo.draw(0, camHeight/2);
    cam.draw(camWidth, camHeight/2);
    //grayImage.draw(10, 320, 400, 300);
    //tex.draw(0,camHeight/2);
    mClient.draw(50, 50);
    
    mainOutputSyphonServer.publishScreen();
}
void ofApp::keyPressed  (int key){
    
    switch (key) {
        case OF_KEY_LEFT:
            contourScale -= 1;
            cout << "\n sensitivity";
            cout << contourScale;
            break;
        case OF_KEY_RIGHT:
            contourScale +=1;
            cout << "\n sensitivity";
            cout << contourScale;
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
    // wait for half a second before forgetting something
    contourFinder.getTracker().setPersistence(contourPersistance);// second
}
