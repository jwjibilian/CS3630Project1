#!/usr/bin/env python3
#!c:/Python35/python3.exe -u
import asyncio
import sys
import cv2
import numpy as np
import cozmo
import time
import os
import _thread
import cv2


from cozmo.util import degrees, distance_mm

from glob import glob

from find_cube import *

try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter
except ImportError:
    # for Python3
    from tkinter import *

try:
    from PIL import ImageDraw, ImageFont, Image, ImageTk
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')
def nothing(x):
    pass



GREEN_LOWER = np.array([0,0,0])
GREEN_UPPER = np.array([179, 255, 60])
makeImage = 0


def filter_image(img, hsv_lower, hsv_upper):

    # Modify mask
    imgToBlur = cv2.medianBlur(img,5)
    imagehsv = cv2.cvtColor(imgToBlur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imagehsv,hsv_lower,hsv_upper)
    image2 = cv2.bitwise_and(255-img, 255-img, mask = mask)
    #cv2.namedWindow( "imagex", cv2.WINDOW_NORMAL );
    #cv2.imshow("imagex", image2)
    #cv2.namedWindow( "imagey", cv2.WINDOW_NORMAL );
    #cv2.imshow("imagey",img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return mask


def detect_blob(mask):
    img = cv2.medianBlur(255 - mask, 9)
    # cv2.namedWindow( "imagey", cv2.WINDOW_NORMAL )
    # cv2.imshow("imagey",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Set up the SimpleBlobdetector with default parameters with specific values.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = True

    params.filterByArea = True
    params.minArea = 500

    params.filterByConvexity = False
    params.minConvexity = 0.7
    params.filterByCircularity = False
    params.minCircularity = 0

    # builds a blob detector with the given parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # use the detector to detect blobs.
    keypoints = detector.detect(img)
    print("keypoints", keypoints)

    return len(keypoints)



# Define a decorator as a subclass of Annotator; displays the keypoint
class BoxAnnotator(cozmo.annotate.Annotator):

    cube = None

    def apply(self, image, scale):
        d = ImageDraw.Draw(image)
        bounds = (0, 0, image.width, image.height)

        if BoxAnnotator.cube is not None:

            #double size of bounding box to match size of rendered image
            BoxAnnotator.cube = np.multiply(BoxAnnotator.cube,2)

            #define and display bounding box with params:
            #msg.img_topLeft_x, msg.img_topLeft_y, msg.img_width, msg.img_height
            box = cozmo.util.ImageBox(BoxAnnotator.cube[0]-BoxAnnotator.cube[2]/2,
                                      BoxAnnotator.cube[1]-BoxAnnotator.cube[2]/2,
                                      BoxAnnotator.cube[2], BoxAnnotator.cube[2])
            cozmo.annotate.add_img_box_to_image(image, box, "green", text=None)

            BoxAnnotator.cube = None

def buttonclick():
    global makeImage
    makeImage= 1


async def run(sdk_conn):
    robot= await sdk_conn.wait_for_robot()

    mainWindow = Toplevel()
    coz = Tk()
    cozmo.tkview.TkImageViewer(tk_root=coz)

    Label(mainWindow, text="Gain").grid(row=0)

    gainx = Scale(mainWindow, from_=0.2, to=3.9, orient=HORIZONTAL, resolution=0.01)
    gainx.grid(row=0, column=1)

    Label(mainWindow, text="Exposure").grid(row=1)
    exposurex = Scale(mainWindow, from_=1, to=67, orient=HORIZONTAL, resolution=0.01)
    exposurex.grid(row=1, column=1)

    Label(mainWindow, text="Hue Lower").grid(row=2)
    hueL = Scale(mainWindow, from_=0, to=255, orient=HORIZONTAL)
    hueL.grid(row=2, column=1)
    Label(mainWindow, text="Hue Upper").grid(row=3)
    hueU = Scale(mainWindow, from_=0, to=255, orient=HORIZONTAL)
    hueU.grid(row=3, column=1)

    Label(mainWindow, text="Saturation Lower").grid(row=4)
    saturationL = Scale(mainWindow, from_=0, to=255, orient=HORIZONTAL)
    saturationL.grid(row=4, column=1)
    Label(mainWindow, text="Saturation Upper").grid(row=6)
    saturationU = Scale(mainWindow, from_=0, to=255, orient=HORIZONTAL)
    saturationU.grid(row=6, column=1)

    Label(mainWindow, text="Value Lower").grid(row=7)
    valueL = Scale(mainWindow, from_=0, to=255, orient=HORIZONTAL)
    valueL.grid(row=7, column=1)
    Label(mainWindow, text="Value Upper").grid(row=8)
    valueU = Scale(mainWindow, from_=0, to=255, orient=HORIZONTAL)
    valueU.grid(row=8, column=1)


    b = Button(mainWindow, text="filter", command = buttonclick)
    b.grid(row=9, column = 0)
    #imageView = Canvas();
    #imageView.pack(side='top', fill='both', expand='yes')
    l = Label(mainWindow)
    l.grid(row=10, column=0)



    robot.world.image_annotator.annotation_enabled = False
    robot.world.image_annotator.add_annotator('box', BoxAnnotator)

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True

    global makeImage













    
    #gain,exposure,mode = 390,3,0
    mode = 0

    try:

        while True:
            YELLOW_LOWER = np.array([hueL.get(), saturationL.get(), valueL.get()])
            YELLOW_UPPER = np.array([hueU.get(), saturationU.get(), valueU.get()])
            exposure = exposurex.get()
            gain = gainx.get()
            #print(gainx, "    ",exposurex)

            event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage)   #get camera image
            if event.image is not None:
                image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)
                mask = filter_image(np.asanyarray(event.image),YELLOW_LOWER,YELLOW_UPPER)
                #b,g,r = cv2.split(mask)
                #display = cv2.merge((mask))
                im = Image.fromarray(cv2.bitwise_and(image,image,mask = mask))
                imgtk = ImageTk.PhotoImage(image=im)
                l.configure(image = imgtk)
                # if (makeImage == 1):
                #     print("makeing window?")
                #
                #     cv2.namedWindow("window",cv2.WINDOW_NORMAL)
                #     cv2.imshow( "window", mask)
                #     makeImage=0
                if mode == 1:
                    robot.camera.enable_auto_exposure = True
                else:
                    robot.camera.set_manual_exposure(exposure,gain)

                #find the cube
                cube = find_cube(image, YELLOW_LOWER, YELLOW_UPPER)
                print(cube)
                BoxAnnotator.cube = cube

                ################################################################

                # Todo: Add Motion Here
                ################################################################
                await robot.set_head_angle(degrees(0)).wait_for_completed()
                #look_around = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
                # try:
                #     cube = await robot.world.wait_for_observed_light_cube(timeout=1)
                #     print("Found cube: %s" % cube)
                # except asyncio.TimeoutError:
                #     print("Didn't find a cube")
                # finally:
                #     # whether we find it or not, we want to stop the behavior
                #     look_around.stop()
                # if cube:
                #     action = robot.go_to_object(cube, distance_mm(50.0))
                #     await action.wait_for_completed()
                #     print("Completed action: result = %s" % action)
                #     print("Done.")



    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)
    #cv2.destroyAllWindows()
def runCozmoRun():
    try:

        cozmo.connect_with_tkviewer(run)

    except:
        print("Unexpected error:", sys.exc_info())




if __name__ == '__main__':

    cv2.namedWindow("window",cv2.WINDOW_NORMAL)
    runCozmoRun()



