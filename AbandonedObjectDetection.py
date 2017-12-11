import numpy as np
import cv2
import time
import sys
import os
from json_encoder import json
from os.path import expanduser
from pymongo import MongoClient


def getForegroundMask(frame, background, th):
    # reduce the nois in the farme
    frame =  cv2.blur(frame, (5,5))
    # get the absolute difference between the foreground and the background
    fgmask= cv2.absdiff(frame, background)
    # convert foreground mask to gray
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
    # apply threshold (th) on the foreground mask
    _, fgmask = cv2.threshold(fgmask, th, 255, cv2.THRESH_BINARY)
    # setting up a kernal for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply morpholoygy on the foreground mask to get a better result
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    return fgmask


def MOG2init(history, T, nMixtures):
    # create an instance of MoG and setting up its history length
    fgbg = cv2.createBackgroundSubtractorMOG2(history)
    # setting up the protion of the background model
    fgbg.setBackgroundRatio(T)
    # setting up the number of MoG
    fgbg.setNMixtures(nMixtures)
    return fgbg


def extract_objs(image, step_size, window_size):
    # a threshold for min static pixels needed to be found in the sliding window
    th = (window_size**2) * 0.1
    current_nonzero_elements = 0
    # penalty is how meny times the expanding process didn't manage to find new
    # static pixels, step is how much the expanding of the sliding will be and objs is a returned
    # value containing the objects in the image
    penalty, step, objs = 0, 5, []
    # a while loop for sliding window in x&y
    y = 0
    while(y < image.shape[0]):
        x = 0
        while(x < image.shape[1]):
            # counting the nonzero elements in the current window
            current_nonzero_elements = np.count_nonzero(image[y:y+window_size, x:x+window_size])
            print(current_nonzero_elements, th)
            if(current_nonzero_elements > th):
                width =  window_size
                height = window_size
                # expand in x & y
                penalty = 0
                while(penalty < 1):
                    dx = np.count_nonzero(image[y:y+height, x+width:x+width+step])
                    dy = np.count_nonzero(image[y+height: y+height+step, x:x+width])
                    if(dx == 0 and dy == 0):
                        penalty += 1
                        width += step
                        height += step
                    elif(dx >= dy):
                        width += step
                    else:
                        height += step

                objs.append([x, y, width, height])
                y += height
                break
            x += step_size
        y += step_size
    if(len(objs)):
        return objs
    return


def extract_objs2(im, min_w=15, min_h=15, max_w=500, max_h=500):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    arr = cv2.dilate(im, kernel, iterations=2)
    arr = np.array(arr, dtype=np.uint8)
    _, th = cv2.threshold(arr,127,255,0)
    im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    #cv2.imshow('arr', arr)
    cv2.imwrite("tmp2.jpg", arr)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if (w >= min_w) & (w < max_w) & (h >= min_h ) & (h < max_h):
            objs.append([x,y,w,h, 1]) # The last one means that it is still needed to check
                                      # if it is a human or an obj
        else:
            print(w,h)
    return objs


# this function returns static object map without pre-founded objects
def clean_map(m, o):
    rslt = np.copy(m)
    for i in range (0, len(o)):
        x, y = o[i][0], o[i][1]
        w, h = o[i][2], o[i][3]
        rslt[y:y+h, x:x+w] = 0
    return rslt


def checkConfigFile():
    home = expanduser("~")
    if not os.path.isfile(home+"/config.json"):
        with open(home+'/config.json', 'a') as the_file:
            init_json = {'min_time': 3, "max_time": 10, "width": 20, "height": 20}
            init_json_str = json.dumps(init_json)
            the_file.write(init_json_str)


def readConfig(filename):
    min_t = 0.0
    max_t = 0.0
    d_width = 0
    d_height = 0
    try:
        checkConfigFile()
        with open(expanduser("~")+"/"+filename) as json_data:
            d = json.load(json_data)
            min_t = d["min_time"]
            max_t = d["max_time"]
            d_width = d["width"]
            d_height = d["height"]
        json_data.close()
    except Exception as e:
        print("[x] config file format is wrong.")
        sys.exit(0)
    return min_t, max_t, d_width, d_height


def help():
    print("--------------------------------------------------\n")
    print("Usages:\n")
    print("./mydemo {-i <video filename> -db host:port}")
    print("--------------------------------------------------\n")


class AbandonedObjectDetection:

    def __init__(self, cap, background, history=300, T=0.4, nMixtures=3,
        longBackgroundInterval=20, shortBackgroundINterval=1,
        k=7, maxe=2000, thh=800):

        self.cap = cap
        # background model
        self.BG = background
        self.BL = None
        self.BS = None

        # setting up a kernal for morphology
        self.kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

        # MoG for long background model
        self.fgbgl = MOG2init(history, T, nMixtures)
        # MoG for short background model
        self.fgbgs = MOG2init(50, T, nMixtures)

        self.longBackgroundInterval = longBackgroundInterval
        self.shortBackgroundINterval = shortBackgroundINterval

        self.clfg = longBackgroundInterval   # counter for longBackgroundInterval
        self.csfg = shortBackgroundINterval  # counter for shortBackgroundInteral

        # static obj likelihood
        self.L = np.zeros(np.shape(self.cap.read()[1])[0:2])

        self.static_obj_map = np.zeros(np.shape(self.cap.read()[1])[0:2])

        # static obj likelihood constants
        self.k, self.maxe, self.thh = k, maxe, thh

        # obj-extraction constants
        self.slidewindowtime = 0
        self.static_objs = []
        self.th_sp = 20 # a th for number of static pixels


    def get_abandoned_objs(self, frame, d_fps_c):
        f2 = frame.copy()

        if self.clfg == self.longBackgroundInterval:
            frameL = np.copy(frame)
            self.fgbgl.apply(frameL)
            self.BL = self.fgbgl.getBackgroundImage(frameL)
            self.clfg = 0
        else:
            self.clfg += 1
        if self.csfg == self.shortBackgroundINterval:
            frameS = np.copy(frame)
            self.fgbgs.apply(frameS)
            self.BS = self.fgbgs.getBackgroundImage(frameS)
            self.csfg = 0
        else:
            self.csfg += 1

        # update short&long foregrounds
        FL = getForegroundMask(frame, self.BL, 50)
        FS = getForegroundMask(frame, self.BS, 50)
        FG = getForegroundMask(frame, self.BG, 50)

        # detec static pixels and apply morphology on it

        static = FL & cv2.bitwise_not(FS) & FG
        cv2.imshow("static", static)
        static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, self.kernal)
        # dectec non static objectes and apply morphology on it
        not_static = FS|cv2.bitwise_not(FL)
        not_static = cv2.morphologyEx(not_static, cv2.MORPH_CLOSE, self.kernal)

        #cv2.imshow("static", static)
        #cv2.imshow("not_static", not_static)

        # update static obj likelihood

        self.L = (static == 255) * (self.L+1) + ((static == 255)^1) * self.L
        self.L = (not_static == 255) * (self.L-self.k) + ((not_static == 255)^1) * self.L
        self.L[self.L>self.maxe] = self.maxe
        self.L[self.L<0] = 0
        cv2.imshow("L", self.L)

        # update static obj map
        self.static_obj_map[self.L >= self.thh] = 254
        self.static_obj_map[self.L < self.thh] = 0


        # if number of nonzero elements in static obj map greater than min window size squared there
        # could be a potential static obj, we will need to wait 200 frame to be pased if the condtion
        # still true we will call "extract_objs" function and try to find these objects.
        if(np.count_nonzero(clean_map(self.static_obj_map, self.static_objs)) > self.th_sp):

            if(self.slidewindowtime > d_fps_c):
                #new_objs = extract_objs2(clean_map(static_obj_map, static_objs))
                new_objs = extract_objs2(clean_map(self.static_obj_map, self.static_objs))
                # if we get new object, first we make sure that they are not dublicated ones and then
                # put the unique static objects in "static_objs" variable
                if(new_objs):
                    for i in range(0, len(new_objs)):
                        if new_objs[i] not in self.static_objs:
                            self.static_objs.append(new_objs[i])
                self.slidewindowtime = 0
            else:
                self.slidewindowtime += 1
        else:
                self.slidewindowtime = 0 if self.slidewindowtime < 0 else self.slidewindowtime - 1
        # draw recatngle around static obj/s
        c=0
        for i in range(0, len(self.static_objs)):
            if(self.static_objs[i-c]):
                x, y = self.static_objs[i-c][0], self.static_objs[i-c][1]
                w, h = self.static_objs[i-c][2], self.static_objs[i-c][3]
                check_human_flag = self.static_objs[i-c][4]
                # check if the current static obj still in the scene 
                cv2.imshow("t", frame[y:y+h, x:x+w])
                cv2.imwrite("test_img.jpg", frame[y:y+h, x:x+w])
                if((np.count_nonzero(self.static_obj_map[y:y+h, x:x+w]) < w * h * .1)):
                    self.static_objs.remove(self.static_objs[i-c])
                    c += 1
                    continue
                #if(check_human_flag):
                #    if(check_human_flag > 25): # check if the founded obj is a human ever 1 sec
                #        self.static_objs[i-c][4] = 0
                        #if(is_human(frame[y:y+h, x:x+w])):
                        #    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                        #    continue
                    #else:
                    #    self.static_objs[i-c][4] += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        return self.static_objs


if __name__ == '__main__':
    """
    Run a demo.
    """
    connection = None
    db = None
    help()
    min_t, max_t, d_width, d_height = readConfig("config.json")
    if (len(sys.argv) - 1) < 4:
        print("[x] Incorrect input params")
        sys.exit(0)

    if sys.argv[1] == "-i":
        if len(sys.argv[2]) <= 0:
            print("[x] Please input video address")
            sys.exit(0)
        #check db
        if sys.argv[3] == "-db":
            if len(sys.argv[4]) > 0:
                host = ""
                db_name = ""
                if ("//" in sys.argv[4]):
                    print("[x] Please input only host not contained protocol example: 127.0.0.1:27017")
                    sys.exit(0)
                if ("/" in sys.argv[4]):
                    host = sys.argv[4].split("/")[0]
                    db_name = sys.argv[4].split("/")[1]
                connection = MongoClient(host)
                # Issue the serverStatus command and print the results
                db = connection[db_name].stolen

    cap = cv2.VideoCapture(0)
    #BG = cv2.imread('bg.jpg')
    _, f = cap.read()
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    fps = 0
    if int(major_ver) < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps)
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        print "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps)
    BG = f
    init_time = time.time()
    slideshowframecount = int(fps * min_t)
    aod = AbandonedObjectDetection(cap, BG)
    while (1):
        _, frame = cap.read()

        if (time.time()-init_time) > max_t:
            pass
            #BG = frame
            #init_time = time.time()
            #aod = AbandonedObjectDetection(cap, BG)
        #aod = AbandonedObjectDetection(cap, BG)
        objs = aod.get_abandoned_objs(frame, slideshowframecount)

        for obj in objs:
            x, y, w, h, _ = obj
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("1", frame)
        
        key = cv2.waitKey(25) & 0xff
        if key == 27:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
