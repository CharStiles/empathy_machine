#include "ofApp.h"

using namespace ofxCv;
using namespace cv;


void ofApp::setup() {

    red = ofColor(255,0,0,255);
    //vofSetVerticalSync(true);
    ofSetBackgroundAuto(false);
    ofSetVerticalSync(false);
    ofEnableAlphaBlending();
    circleOpacity = 100;
    contourScale = 1; //
    contourPersistance = 1; // 6
    movie.load("simon2.mp4");
    movie.play();
    movie.setVolume(0);
    camWidth = movie.getWidth();  // try to grab at this size.
    camHeight =movie.getHeight();
    
    ringPixels.allocate(camWidth, camHeight, OF_PIXELS_RGB);
    ringTex.allocate(camWidth, camHeight, OF_PIXELS_RGB);
    pix.allocate(camWidth, camHeight, OF_PIXELS_RGBA);
    fbo.allocate(camWidth,camHeight,GL_RGBA);
    ringImg.allocate(camWidth, camHeight,OF_IMAGE_COLOR);
    texImg.allocate(camWidth, camHeight,OF_IMAGE_COLOR_ALPHA);
    
    // faceFinder.setup("haarcascade_frontalface_default.xml");
    // faceFinder.setPreset(ObjectFinder::Fast);

    /// CONTOUR
    contourFinder.setMinAreaRadius(20);//1
    contourFinder.setMaxAreaRadius(1000);//100
    contourFinder.setThreshold(contourScale);
    // wait for half a second before forgetting something
    contourFinder.getTracker().setPersistence(contourPersistance);// second
    // an object can move up to 32 pixels per frame
    contourFinder.getTracker().setMaximumDistance(8);
    //END CONTOUR
    
    // imitate() will set up previous and diff
    // so they have the same size and type as cam
    imitate(previous, movie);
    imitate(diff, movie);
}

void ofApp::update() {

    movie.update();
    if(movie.isFrameNew()) {
        temp1 =(float)((camHeight)/2);
        // take the absolute difference of prev and cam and save it inside diff
        absdiff(movie, previous, diff);
        diff.update();
        
        // like ofSetPixels, but more concise and cross-toolkit
        copy(movie, previous);
        
        // mean() returns a Scalar. it's a cv:: function so we have to pass a Mat
        diffMean = mean(toCv(diff));
        
        // you can only do math between Scalars,
        // but it's easy to make a Scalar from an int (shown here)
        diffMean *= Scalar(5000000000000);
        
        ofPixels & pixels = diff.getPixels();//vidGrabber.getPixels();
        // this turns into tex
        
        //ofPixels pixels;
        //fbo.readToPixels(pixels);
                for(int i = 0; i < contourFinder.size(); i++) {
                    ofPoint center = toOf(contourFinder.getCenter(i));
                }
                for (int  j = 0; j < camHeight; j++){
                    for(int i = 0; i < camWidth; i++){
                        //videoInverted.setColor(i,j,ofColor(0,0,0));
                        ofColor c = pixels.getColor(i,j);
                        //ofColor cc = c.invert();
                        if (c.r < 10 && c.g < 10 && c.b <10){
                           pix.setColor(i,j,ofColor(c.r,c.g,c.b,50));
                        }
                        else{
                            pix.setColor(i,j,ofColor(c.r,c.g,c.b,255));
                        }
                    }
                }
        
        ringTex.loadData(ringPixels); // distorted diff
        ringImg.setFromPixels(ringPixels);// TODO init this
        texImg.setFromPixels(pix);
        // tex.loadData(pixels); // diff feed
        contourFinder.findContours(diff);
        
        // contourFinder.findContours(ringImg);
        
        // contourFinderFull.findContours(texImg);
    }
}

void ofApp::draw() {
    
    
    //fbo.begin();
    
    //ofSetColor(255);
    
    //diff.draw(0,0);
    //ofSetColor(255,255,255,150);
    ofTranslate(-(camWidth/6), -(camHeight/6));
    ofSetLineWidth(ofNoise(ofGetFrameNum()));
    ofSetColor(255,255);
    texImg.draw(0,0);
    
    ofSetColor(255,255,255,circleOpacity);
    contourFinder.draw();
//
//    for (int j = 0; j < contourFinder.size(); j++) {
//        ofPolyline ithPolyline = contourFinder.getPolyline(j);
//        ofPolyline resampled = ithPolyline.getResampledBySpacing(10.0);
//        ofPolyline resampledSmoothed = resampled.getSmoothed(9);
//        //ofSetColor(255,255,0,50);
//        resampledSmoothed.draw();
//    }



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
        case ' ':
            movie.setPaused(!movie.isPaused());
            break;
        case 'm':
            movie.setSpeed(movie.getSpeed() + 0.1);
            break;
        case 'n':
            movie.setSpeed(movie.getSpeed() - 0.1);
            break;
    }
    
    contourFinder.setThreshold(contourScale);
    // wait for half a second before forgetting something
    contourFinder.getTracker().setPersistence(contourPersistance);// second
}
