
This is the application that is specific to the setup to the production of slowdanger's Empathy Machine

We are using this [Fisheye RevoTech IR Ethernet Security Camera](https://www.aliexpress.com/item/Fisheye-FHD-1920-x-1080P-2-0MP-6-Array-LED-IR-Night-Vision-Panoramic-IP-Camera/32799381124.html?spm=a2g0s.13010208.99999999.265.3d7c3c007XJ1qu):

![](imagesForREADME/cam.png)

I am streaming it using ofxGStreamer. I think that this bit of code has the highest potential of helping anyone. 

With the feed from the camera, I am unwraveling it such that it reads a clockwise index around the center into a straight line, so it can be sent to an array of LEDs that is configured in a circle. This is so the LEDs can light up where the dancer is dancing in the circle 

(pictures coming soon)

Using ofxOpenCV and ofxCV I am getting the position of the dancers and highlighting it so that it works the same for different colored people. 