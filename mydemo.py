#!/usr/bin/python

import cv2
import time
import sys
import json
import copy
import os
from bson.binary import Binary
from skimage.measure import compare_ssim
from pymongo import MongoClient
from os.path import expanduser

min_t = 5.0
max_t = 10.0
d_width = 10
d_height = 10


#global variables
e1 = None
e2 = None
firstchecker = True
updatebg_checker = False
start_tracker_checker = False
post_event_checker = False


def checkConfigFile():
    home = expanduser("~")
    if not os.path.isfile(home+"/config.json"):
        with open(home+'/config.json', 'a') as the_file:
            init_json_str = '''
            {
                "min_time":3,
                "max_time":6,
                "width": 20,
                "height": 20
            }
            '''
            the_file.write(init_json_str)


def readConfig(filename):
    global min_t, max_t, d_width, d_height
    try:
        with open(expanduser("~")+"/"+filename) as json_data:
            d = json.load(json_data)
            min_t = d["min_time"]
            max_t = d["max_time"]
            d_width = d["width"]
            d_height = d["height"]
    except Exception as e:
        print("[x] config file format is wrong.")
        sys.exit(0)


def help():
    print("--------------------------------------------------\n")
    print("Usages:\n")
    print("./mydemo {-i <video filename> -db host:port}")
    print("--------------------------------------------------\n")


def stolenModule(bg, frame):
    global start_tracker_checker
    global e1, e2
    global post_event_checker
    display_frame = frame.copy()
    graybg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(graybg, grayframe, full=True)
    diff = (diff * 255).astype("uint8")
    diffMask = cv2.threshold(diff, 0, 255,  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.medianBlur(diffMask, 15, diffMask)
    img, cnts, hierarchy = cv2.findContours(diffMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    exist_interested_obj_checker = False
    dimenson_json = []
    for contour in cnts:
        try:
            rect = cv2.boundingRect(contour)
        except Exception as e:
            continue
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        if w>d_width and h > d_height:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if not start_tracker_checker:
                start_tracker_checker = True
                e1 = time.time()
            exist_interested_obj_checker = True
            demension_data = {"x": x, "y": y, "w": w, "h": h}
            dimenson_json.append(json.dumps(demension_data))
    record_data = {'event_time':None, 'frame':None, 'background': None, 'dimensions':dimenson_json}
    #record_json = json.dumps(record_data)
    if exist_interested_obj_checker and start_tracker_checker:
        e2 = time.time()
        d_t = int(e2 - e1)
        if d_t > min_t and not post_event_checker:
            print("[!] Raised event: ")
            cv2.imwrite("tmp_frame.jpg", frame)
            post_event_checker = True
            record_data["event_time"] = e1
            with open("tmp_bg.jpg", "rb") as binary_bg_file:
                record_data["background"] = Binary(binary_bg_file.read())
            with open("tmp_frame.jpg", "rb") as binary_frame_file:
                record_data["frame"] = Binary(binary_frame_file.read())
            return record_data
        if d_t > max_t:
            firstchecker = True
            start_tracker_checker = False
            post_event_checker = False
            cv2.imwrite("tmp_bg.jpg", bg)

    if not exist_interested_obj_checker:
        start_tracker_checker = False

    try:
        cv2.imshow("Frame", display_frame)
    except Exception as e:
        pass
    return None


if __name__ == "__main__":
    checkConfigFile()
    connection = None
    db = None
    help()
    readConfig("config.json")
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

                cap = cv2.VideoCapture(sys.argv[2])
                skipcount = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if(frame is not None):
                            if skipcount == 10:
                                skipcount = 0
                                if firstchecker:
                                    bg = copy.copy(frame)
                                    firstchecker = False
                                    continue
                                try:
                                    record_data = stolenModule(bg, frame)
                                    if record_data is not None:
                                        db.insert(record_data)
                                        print(" [!] Inserted document successful in DB")
                                except Exception as e:
                                    print("exception")

                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                            skipcount = skipcount + 1
                cap.release()
                cv2.destroyAllWindows()

        else:
            print("[x] Missing Param of Database: -db host:port")
            sys.exit(0)
