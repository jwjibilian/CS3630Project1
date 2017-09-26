#!/usr/bin/env python3
#!c:/Python35/python3.exe -u

import asyncio
import sys
import cv2
import numpy as np
import cozmo
import time
import os

from cozmo.util import degrees, distance_mm
try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter
except ImportError:
    # for Python3
    from tkinter import *

mainWindow = Tk()

Label(mainWindow, text="Gain").grid(row=0)
gain = Scale(mainWindow, from_=0, to=500,orient=HORIZONTAL)
gain.grid(row=0,column=1)

Label(mainWindow, text="Exposure").grid(row=1)
exposure = Scale(mainWindow, from_=0.1, to=4.0, orient=HORIZONTAL,resolution = 0.01)
exposure.grid(row=1, column = 1)

Label(mainWindow, text="Hue").grid(row=2)
hue = Scale(mainWindow, from_=0, to=255, orient=HORIZONTAL)
hue.grid(row=2, column = 1)

Label(mainWindow, text="Saturation").grid(row=3)
saturation = Scale(mainWindow, from_=0, to=255, orient=HORIZONTAL)
saturation.grid(row=3, column = 1)

Label(mainWindow, text="Value").grid(row=4)
value = Scale(mainWindow, from_=0, to=255, orient=HORIZONTAL)
value.grid(row=4, column = 1)






mainloop()

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


async def run(robot: cozmo.robot.Robot):

    robot.world.image_annotator.annotation_enabled = False
    robot.world.image_annotator.add_annotator('box', BoxAnnotator)

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True



    try:

        while True:
            gain, exposure, mode = 390, 3, 1


            event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)   #get camera image
            if event.image is not None:
                image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)

                if mode == 1:
                    robot.camera.enable_auto_exposure = True
                else:
                    robot.camera.set_manual_exposure(exposure,fixed_gain)

                #find the cube
                cube = find_cube(image, YELLOW_LOWER, YELLOW_UPPER)
                print(cube)
                BoxAnnotator.cube = cube

                ################################################################

                # Todo: Add Motion Here
                ################################################################
                await robot.set_head_angle(degrees(0)).wait_for_completed()
                look_around = robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
                try:
                    cube = await robot.world.wait_for_observed_light_cube(timeout=30)
                    print("Found cube: %s" % cube)
                except asyncio.TimeoutError:
                    print("Didn't find a cube")
                finally:
                    # whether we find it or not, we want to stop the behavior
                    look_around.stop()
                if cube:
                    action = robot.go_to_object(cube, distance_mm(50.0))
                    await action.wait_for_completed()
                    print("Completed action: result = %s" % action)
                    print("Done.")



    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    cozmo.run_program(run, use_viewer = True, force_viewer_on_top = True)