#include "ofApp.h"

int main() {
    ofGLWindowSettings settings;
    settings.setGLVersion(3,3);
    settings.windowMode = OF_FULLSCREEN;
    ofCreateWindow(settings);
    
    // this kicks off the running of my app
    // can be OF_WINDOW or OF_FULLSCREEN
    // pass in width and height too:
    ofRunApp(new ofApp());
    
//    ofSetupOpenGL(1920,1080,OF_WINDOW);
//    ofRunApp(new ofApp());
}
