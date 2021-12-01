# sort: 
# <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> .txt
# in detection: <id> = -1
# <conf>, <x>, <y>, <z> = -1
# OpenTrafficCam:
# det: .otdet, track: .ottrk
# det: oben zwei Bloecke mit allg Infos -> werden in .ottrk mitgenommen -> dann ein Block mit allg Tracking Infos
# det -> det.sort:
# "data": {
#         "1": {                                <frame>
#             "classified": [                   
#                 "class": "car",
#                 "conf": 0.9132447242736816,   
#                 "x": 850.5742797851562,       <bb_left>
#                 "y": 208.87606811523438,      <bb_top>
#                 "w": 176.73046875,            <bb_width>
#                 "h": 79.70358276367188,       <bb_heigth>
# trk.sort -> ottrk:
# "data": {
#         "1": {                                <frame>
#             "1": {                            <id>      
#                 "class": "car",               <>
#                 "conf": 0.9132447242736816,   <>
#                 "x": 850.5742797851562,       <bb_left>
#                 "y": 208.87606811523438,      <bb_top>
#                 "w": 176.73046875,            <bb_width>
#                 "h": 79.70358276367188,       <bb_heigth>
#                 "first": true,                <>
#                 "finished": true              <>


from os import close, write
import pandas as pd
import csv
import numpy as np

def conv_det(otc_det_file):
    otc_det = open(otc_det_file, "r")
    print('Testtext')
    otc_det_arr = (otc_det.readlines())
    otc_det.close()
    print(otc_det_arr[0])
    # die ersten 18 Zeilen sind nur Kopf
    kopf = 19

    datei_name = otc_det.name
    datei_name = datei_name.replace(".otdet","_")
    print(datei_name)
    sort_det = open(f'{datei_name}det-for-sort.txt',"w")
    zeile = kopf

    while zeile <= len(otc_det_arr)-7:
        if zeile != kopf:
            zeile = zeile + 4
        
        if zeile == kopf:
            start_frame = otc_det_arr[zeile].index("\"")+1
            frame = (otc_det_arr[zeile])[start_frame]
            sort_det.write((frame+",-1,"))
            zeile = zeile + 5
        
        if "{" in otc_det_arr[zeile] and zeile != kopf:
            # print(zeile)
            start_frame = otc_det_arr[zeile].index("\"")+1
            end_frame = otc_det_arr[zeile].index(":")-1
            frame = ""
            for ziffer in range(start_frame, end_frame):
                frame = frame + (otc_det_arr[zeile])[ziffer]
            sort_det.write("\n"+ (frame+",-1,"))
            zeile = zeile + 5
        
        if "conf" in otc_det_arr[zeile]:
            zeile = zeile + 1
            sort_det.write("\n"+ (frame+",-1,"))

        for zeile in range(zeile,(zeile+4)):
            start = otc_det_arr[zeile].index(":")
            stop = len(otc_det_arr[zeile])-1
            for ziffer in range((start+2),stop):
                sort_det.writelines((otc_det_arr[zeile])[(ziffer)])
        sort_det.write(",-1,-1,-1,-1")


    sort_det.close()
    return(otc_det_arr)

def conv_trk(sort_trk_file,otc_det_arr):
    sort_trk = open(sort_trk_file, "r")
    # sort_trk_arr = sort_trk.readlines()
    sort_trk.close()
    sort_trk_np = np.loadtxt(sort_trk_file, delimiter=',')
    print(sort_trk_np[1])

    datei_name = sort_trk.name
    datei_name = datei_name.replace(".txt","_")
    print(datei_name)
    otc_trk = open(f'{datei_name}trk-from-sort-false.ottrk',"w")

    kopf = 18
    for zeile in range(0, kopf):
        otc_trk.write(otc_det_arr[zeile])
    otc_trk.write("    \"trk_config\": {\n")
    # otc_trk.write("        \"sigma_l\": 0.25,\n")
    # otc_trk.write("        \"sigma_h\": 0.8,\n")
    # otc_trk.write("        \"sigma_iou\": 0.3,\n")
    # otc_trk.write("        \"t_min\": 5.0,\n")
    # otc_trk.write("        \"save_age\": 10.0,\n")
    otc_trk.write("        \"tracker\": \"SORT\"\n")
    otc_trk.write("    },\n")
    otc_trk.write(otc_det_arr[18])

    frame = 0
    for row in range(0,len(sort_trk_np)):
        # if neues frame
        if frame != str(int(sort_trk_np[row][0])):
            frame = str(int(sort_trk_np[row][0]))
            if row > 0:
                otc_trk.write("\n")
                otc_trk.write("        },\n")
            otc_trk.write("        \""+frame+"\": {\n")
        else: 
            otc_trk.write(",\n")
        id = str(int(sort_trk_np[row][1]))
        conf = "80"
        otc_trk.write("            \""+id+"\": {\n")
        otc_trk.write("                \"class\": \"car\",\n")
        otc_trk.write("                \"conf\":"+conf+",\n")
        otc_trk.write("                \"x\": "+str(sort_trk_np[row][2])+",\n")
        otc_trk.write("                \"y\": "+str(sort_trk_np[row][3])+",\n")
        otc_trk.write("                \"w\": "+str(sort_trk_np[row][4])+",\n")
        otc_trk.write("                \"h\": "+str(sort_trk_np[row][5])+",\n")
        otc_trk.write("                \"first\": "+"false,\n")
        otc_trk.write("                \"finished\": "+"false\n")
        otc_trk.write("            }")
    otc_trk.write("        }\n    }\n}")
    
    otc_trk.close()



otc_det_arr = conv_det("data/projektarbeit_janina/Radeberg/raspberrypi_FR20_2020-02-20_12-00-00.otdet")
conv_trk("data/projektarbeit_janina/Radeberg/raspberrypi_FR20_2020-02-20_12-00-00_maxage20_iou003.txt", otc_det_arr)
