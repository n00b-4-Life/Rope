import os
import cv2
import json
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from skimage import transform as trans
import subprocess
from math import floor, ceil
import bisect
import onnxruntime
import torchvision
from torchvision.transforms.functional import normalize #update to v2
import torch
from torchvision import transforms
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
torch.set_grad_enabled(False)
onnxruntime.set_default_logger_severity(4)

import inspect #print(inspect.currentframe().f_back.f_code.co_name, 'resize_image')

device = 'cuda'

lock=threading.Lock()

class VideoManager():  
    def __init__(self, models ):
        self.models = models
        # Model related
        self.swapper_model = []             # insightface swapper model
        # self.faceapp_model = []             # insight faceapp model
        self.input_names = []               # names of the inswapper.onnx inputs
        self.input_size = []                # size of the inswapper.onnx inputs

        self.output_names = []              # names of the inswapper.onnx outputs    
        self.arcface_dst = np.array( [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)     

        self.video_file = []

        self.FFHQ_kps = np.array([[ 192.98138, 239.94708 ], [ 318.90277, 240.1936 ], [ 256.63416, 314.01935 ], [ 201.26117, 371.41043 ], [ 313.08905, 371.15118 ] ])
        
     
        
        #Video related
        self.capture = []                   # cv2 video
        self.is_video_loaded = False        # flag for video loaded state    
        self.video_frame_total = None       # length of currently loaded video
        self.play = False                   # flag for the play button toggle
        self.current_frame = 0              # the current frame of the video
        self.create_video = False
        self.output_video = []       
        self.file_name = []
        self.track = []

        
        # Play related
        # self.set_read_threads = []          # Name of threaded function
        self.frame_timer = 0.0      # used to set the framerate during playing
        
        # Queues
        self.action_q = []                  # queue for sending to the coordinator
        self.frame_q = []                   # queue for frames that are ready for coordinator

        self.r_frame_q = []                 # queue for frames that are requested by the GUI
        self.read_video_frame_q = []
        
        # swapping related
        # self.source_embedding = []          # array with indexed source embeddings

        self.found_faces = []   # array that maps the found faces to source faces    

        self.parameters = []


        self.target_video = []

        self.fps = 1.0
        self.temp_file = []


        self.clip_session = []

        self.start_time = []
        self.record = False
        self.output = []
        self.image = []

        self.saved_video_path = []
        self.sp = []
        self.timer = []
        self.fps_average = []
        self.total_thread_time = 0.0
        
        self.start_play_time = []
        self.start_play_frame = []
        
        self.rec_thread = []
        self.markers = []
        self.is_image_loaded = False
        self.stop_marker = -1
        self.perf_test = False

        self.control = []

        



        self.process_q =    {
                            "Thread":                   [],
                            "FrameNumber":              [],
                            "ProcessedFrame":           [],
                            "Status":                   'clear',
                            "ThreadTime":               []
                            }   
        self.process_qs = []
        self.rec_q =    {
                            "Thread":                   [],
                            "FrameNumber":              [],
                            "Status":                   'clear'
                            }   
        self.rec_qs = []

    def assign_found_faces(self, found_faces):
        self.found_faces = found_faces


    def load_target_video( self, file ):
        # If we already have a video loaded, release it
        if self.capture:
            self.capture.release()
            
        # Open file   
        self.video_file = file
        self.capture = cv2.VideoCapture(file)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        
        if not self.capture.isOpened():
            print("Cannot open file: ", file)
            
        else:
            self.target_video = file
            self.is_video_loaded = True
            self.is_image_loaded = False
            self.video_frame_total = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.play = False 
            self.current_frame = 0
            self.frame_timer = time.time()
            self.frame_q = []            
            self.r_frame_q = []             
            self.found_faces = []
            self.add_action("set_slider_length",self.video_frame_total-1)
            self.add_action("update_markers_canvas", self.markers)

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)        
        success, image = self.capture.read() 
        
        if success:
            crop = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
            temp = [crop, False]
            self.r_frame_q.append(temp)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
    
    def load_target_image(self, file):
        if self.capture:
            self.capture.release()
        self.is_video_loaded = False
        self.play = False 
        self.frame_q = []            
        self.r_frame_q = [] 
        self.found_faces = []
        self.image = cv2.imread(file) # BGR
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) # RGB
        temp = [self.image, False]
        self.frame_q.append(temp)

        self.is_image_loaded = True

    
    ## Action queue
    def add_action(self, action, param):
        # print(inspect.currentframe().f_back.f_code.co_name, '->add_action: '+action)
        temp = [action, param]
        self.action_q.append(temp)    
    
    def get_action_length(self):
        return len(self.action_q)

    def get_action(self):
        action = self.action_q[0]
        self.action_q.pop(0)
        return action
     
    ## Queues for the Coordinator
    def get_frame(self):
        frame = self.frame_q[0]
        self.frame_q.pop(0)
        return frame
    
    def get_frame_length(self):
        return len(self.frame_q)  
        
    def get_requested_frame(self):
        frame = self.r_frame_q[0]
        self.r_frame_q.pop(0)
        return frame
    
    def get_requested_frame_length(self):
        return len(self.r_frame_q)          
    

    def get_requested_video_frame(self, frame, marker=True):  
        temp = []
        if self.is_video_loaded:
        
            if self.play == True:            
                self.play_video("stop")
                self.process_qs = []
                
            self.current_frame = int(frame)

            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            success, target_image = self.capture.read() #BGR

            if success:
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB) #RGB 
                if not self.control['SwapFacesButton']:   
                    temp = [target_image, self.current_frame] #temp = RGB
                else:
                    temp = [self.swap_video(target_image, self.current_frame, marker), self.current_frame] # temp = RGB

                self.r_frame_q.append(temp)  
        
        elif self.is_image_loaded:
            if not self.control['SwapFacesButton']:
                temp = [self.image, self.current_frame] # image = RGB
        
            else:  
                temp = [self.swap_video(self.image, self.current_frame, False), self.current_frame] # image = RGB
            
            self.r_frame_q.append(temp)  


    def find_lowest_frame(self, queues):
        min_frame=999999999
        index=-1
        
        for idx, thread in enumerate(queues):
            frame = thread['FrameNumber']
            if frame != []:
                if frame < min_frame:
                    min_frame = frame
                    index=idx
        return index, min_frame


    def play_video(self, command):
        # print(inspect.currentframe().f_back.f_code.co_name, '->play_video: ')
        if command == "play":
            # Initialization
            self.play = True
            self.fps_average = []            
            self.process_qs = []
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.frame_timer = time.time()
            self.track.clear()
            # Create reusable queue based on number of threads
            for i in range(self.parameters['ThreadsSlider']):
                    new_process_q = self.process_q.copy()
                    self.process_qs.append(new_process_q)
                    
            
            # Start up audio if requested
            if self.control['AudioButton']:  
                seek_time = (self.current_frame)/self.fps
                args =  ["ffplay", 
                        '-vn', 
                        '-ss', str(seek_time),
                        '-nodisp',
                        '-stats',
                        '-loglevel',  'quiet', 
                        '-sync',  'audio',
                        self.video_file]
 
                
                self.audio_sp = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                # Parse the console to find where the audio started
                while True:
                    temp = self.audio_sp.stdout.read(69)    
                    if temp[:7] != b'    nan':
                        sought_time = float(temp[:7])
                        self.current_frame = int(self.fps*sought_time)
                        
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

                        break


#'    nan    :  0.000
#'   1.25 M-A:  0.000 fd=   0 aq=   12KB vq=    0KB sq=    0B f=0/0' 
                

        elif command == "stop":
            self.play = False
            self.add_action("stop_play", True)
            
            index, min_frame = self.find_lowest_frame(self.process_qs)
            
            if index != -1:
                self.current_frame = min_frame-1   
            
            if self.control['AudioButton']:    
                self.audio_sp.terminate()

            torch.cuda.empty_cache()
                
        elif command=='stop_from_gui':
            self.play = False

            # Find the lowest frame in the current render queue and set the current frame to the one before it
            index, min_frame = self.find_lowest_frame(self.process_qs)
            if index != -1:
                self.current_frame = min_frame-1   
            
            if self.control['AudioButton']:    
                self.audio_sp.terminate()

            torch.cuda.empty_cache()

        elif command == "record":
            self.record = True
            self.play = True
            self.total_thread_time = 0.0
            self.process_qs = []
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            print(f"Current Frame: {self.current_frame}")
            for i in range(self.parameters['ThreadsSlider']):
                    new_process_q = self.process_q.copy()
                    self.process_qs.append(new_process_q)

           # Initialize
            self.timer = time.time()
            frame_width = int(self.capture.get(3))
            frame_width = int(self.capture.get(3))
            frame_height = int(self.capture.get(4))

            self.start_time = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))            
            
            self.file_name = os.path.splitext(os.path.basename(self.target_video))
            base_filename =  self.file_name[0]+"_"+str(time.time())[:10]
            self.output = os.path.join(self.saved_video_path, base_filename)
            self.temp_file = self.output+"_temp"+self.file_name[1]  
            
            if self.parameters['RecordTypeTextSel']=='FFMPEG':
                args =  ["ffmpeg", 
                        '-hide_banner',
                        '-loglevel',    'error',
                        "-an",       
                        "-r",           str(self.fps),
                        "-i",           "pipe:",
                        # '-g',           '25',
                        "-vf",          "format=yuvj420p",
                        "-c:v",         "libx264",
                        "-crf",         str(self.parameters['VideoQualSlider']),
                        "-r",           str(self.fps),
                        "-s",           str(frame_width)+"x"+str(frame_height),
                        self.temp_file]  

                self.sp = subprocess.Popen(args, stdin=subprocess.PIPE)
            
            elif self.parameters['RecordTypeTextSel']=='OPENCV':    
                size = (frame_width, frame_height)
                self.sp = cv2.VideoWriter(self.temp_file,  cv2.VideoWriter_fourcc(*'mp4v') , self.fps, size) 
      
    # @profile
    def process(self):
        process_qs_len = range(len(self.process_qs))

        # Add threads to Queue
        if self.play == True and self.is_video_loaded == True:
            for item in self.process_qs:
                if item['Status'] == 'clear' and self.current_frame < self.video_frame_total:

                    item['Thread'] = threading.Thread(target=self.thread_video_read, args = [self.current_frame]).start()
                    item['FrameNumber'] = self.current_frame
                    item['Status'] = 'started'
                    item['ThreadTime'] = time.time()

                    self.current_frame += 1
                    break
          
        else:
            self.play = False

        # Always be emptying the queues
        time_diff = time.time() - self.frame_timer

        if not self.record and time_diff >= 1.0/float(self.fps) and self.play:

            index, min_frame = self.find_lowest_frame(self.process_qs)

            if index != -1:
                if self.process_qs[index]['Status'] == 'finished':
                    temp = [self.process_qs[index]['ProcessedFrame'], self.process_qs[index]['FrameNumber']]
                    self.frame_q.append(temp)

                    # Report fps, other data
                    self.fps_average.append(1.0/time_diff)
                    if len(self.fps_average) >= floor(self.fps):
                        fps = round(np.average(self.fps_average), 2)
                        msg = "%s fps, %s process time" % (fps, round(self.process_qs[index]['ThreadTime'], 4))
                        self.fps_average = []

                    if self.process_qs[index]['FrameNumber'] >= self.video_frame_total-1 or self.process_qs[index]['FrameNumber'] == self.stop_marker:
                        self.play_video('stop')
                        
                    self.process_qs[index]['Status'] = 'clear'
                    self.process_qs[index]['Thread'] = []
                    self.process_qs[index]['FrameNumber'] = []
                    self.process_qs[index]['ThreadTime'] = []
                    self.frame_timer += 1.0/self.fps
                    
        elif self.record:
           
            index, min_frame = self.find_lowest_frame(self.process_qs)           
            
            if index != -1:

                # If the swapper thread has finished generating a frame
                if self.process_qs[index]['Status'] == 'finished':
                    image = self.process_qs[index]['ProcessedFrame']  
                    
                    if self.parameters['RecordTypeTextSel']=='FFMPEG':
                        pil_image = Image.fromarray(image)
                        pil_image.save(self.sp.stdin, 'BMP')   
                    
                    elif self.parameters['RecordTypeTextSel']=='OPENCV':
                        self.sp.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    temp = [image, self.process_qs[index]['FrameNumber']]
                    self.frame_q.append(temp)

                    # Close video and process
                    if self.process_qs[index]['FrameNumber'] >= self.video_frame_total-1 or self.process_qs[index]['FrameNumber'] == self.stop_marker or self.play == False:
                        self.play_video("stop")
                        print("video stopped")

                        stop_time = float(self.capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))
                        print("stop time registered")
                        if stop_time == 0:
                            stop_time = float(self.video_frame_total) / float(self.fps)
                        print("second stop time calculated")
                        if self.parameters['RecordTypeTextSel']=='OPENCV':
                            self.sp.release()
                        elif self.parameters['RecordTypeTextSel'] == 'FFMPEG':
                            self.sp.stdin.close()
                            self.sp.wait()
                        print("sp released")
                        orig_file = self.target_video
                        print("orig_file defined")
                        final_file = self.output+self.file_name[1]
                        print("adding audio...")    
                        args = ["ffmpeg",
                                '-hide_banner',
                                '-loglevel',    'error',
                                "-i", self.temp_file,
                                "-ss", str(self.start_time), "-to", str(stop_time), "-i",  orig_file,
                                "-c",  "copy", # may be c:v
                                "-map", "0:v:0", "-map", "1:a:0?",
                                "-shortest",
                                final_file]
                        
                        four = subprocess.run(args)
                        os.remove(self.temp_file)

                        # ✅ Save similarity tracking JSON here (append + skip duplicates + sort)
                        try:
                            if hasattr(self, "track") and isinstance(self.track, list) and len(self.track) > 0:
                                base, _ = os.path.splitext(os.path.basename(orig_file))
                                dest_dir = os.path.dirname(orig_file)
                                json_path = os.path.join(dest_dir, base + "_times.json")

                                # Load existing data if file exists
                                if os.path.isfile(json_path):
                                    with open(json_path, 'r') as f:
                                        existing_data = json.load(f)
                                else:
                                    existing_data = []

                                # Collect existing frame numbers for fast lookup
                                existing_frames = {entry["frame"] for entry in existing_data}

                                # Only keep new entries that aren't already in the file
                                new_entries = [entry for entry in self.track if entry["frame"] not in existing_frames]

                                if new_entries:
                                    combined = existing_data + new_entries
                                    # Sort all entries by frame number
                                    combined.sort(key=lambda x: x["frame"])

                                    with open(json_path, 'w') as f:
                                        json.dump(combined, f, indent=2)
                                    print(f"Appended {len(new_entries)} new frame(s) to: {json_path}")
                                else:
                                    print("No new frames to append. JSON file unchanged.")

                                self.track.clear()
                        except Exception as e:
                            print(f"[Warning] Failed to save similarity tracking JSON: {e}")

                        timef= time.time() - self.timer
                        self.record = False
                        print('Video saved as:', final_file)
                        msg = "Total time: %s s." % (round(timef,1))
                        print(msg)

                        
                    self.total_thread_time = []
                    self.process_qs[index]['Status'] = 'clear'
                    self.process_qs[index]['FrameNumber'] = []
                    self.process_qs[index]['Thread'] = []
                    self.frame_timer = time.time()
    # @profile
    def thread_video_read(self, frame_number):  
        with lock:
            success, target_image = self.capture.read()

        if success:
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            if not self.control['SwapFacesButton']:
                temp = [target_image, frame_number]
            
            else:
                temp = [self.swap_video(target_image, frame_number, True), frame_number]

            for item in self.process_qs:
                if item['FrameNumber'] == frame_number:
                    item['ProcessedFrame'] = temp[0]
                    item['Status'] = 'finished'
                    item['ThreadTime'] = time.time() - item['ThreadTime']
                    break

    def detect_face_rotation(self, face_kps):
        """
        Detects the face rotation angle using the same similarity transform logic as `recognize()`,
        without modifying the `recognize()` function.

        Parameters:
            face_kps (numpy.ndarray): 5 facial keypoints [(left_eye), (right_eye), (nose), (left_mouth), (right_mouth)]

        Returns:
            float: Rotation angle in degrees. Positive = counterclockwise, Negative = clockwise.
        """

        # Copy the ArcFace reference keypoints
        dst = self.arcface_dst.copy()
        dst[:, 0] += 8.0  # Small shift to match ArcFace standard alignment

        # Estimate transformation
        tform = trans.SimilarityTransform()
        tform.estimate(face_kps, dst)

        # Extract the rotation angle (in degrees)
        rotation_angle = tform.rotation * 57.2958  # Convert radians to degrees

        return rotation_angle























    # @profile

    # @profile
    def swap_video_orig(self, target_image, frame_number, use_markers):

        # Grab a local copy of the parameters to prevent threading issues
        parameters = self.parameters.copy()
        control = self.control.copy()

        # Define the specific parameters to copy
        desired_parameters = {'OrientSwitch', 'OrientSlider', 'MouthParserSlider'}

        # Find out if the frame is in a marker zone and copy the parameters if true
        if self.markers and use_markers:
            temp=[]
            for i in range(len(self.markers)):
                temp.append(self.markers[i]['frame'])

            idx = bisect.bisect(temp, frame_number)

            if frame_number in temp:
                idx = temp.index(frame_number)  # Get the exact index of the frame in the markers list
                print("idx:", idx)
                for key in desired_parameters:
                    if key in self.markers[idx]['parameters']:
                        parameters[key] = self.markers[idx]['parameters'][key]
                        print("Updating", key, "to", parameters[key])

        # Load frame into VRAM
        img = torch.from_numpy(target_image.astype('uint8')).to('cuda') #HxWxc
        img = img.permute(2,0,1)#cxHxW

        #Scale up frame if it is smaller than 512
        img_x = img.size()[2]
        img_y = img.size()[1]

        if img_x<512 and img_y<512:
            if img_x <= img_y:
                tscale = v2.Resize((int(512*img_y/img_x), 512), antialias=True)
            else:
                tscale = v2.Resize((512, int(512*img_x/img_y)), antialias=True)
            img = tscale(img)
        elif img_x<512:
            tscale = v2.Resize((int(512*img_y/img_x), 512), antialias=True)
            img = tscale(img)
        elif img_y<512:
            tscale = v2.Resize((512, int(512*img_x/img_y)), antialias=True)
            img = tscale(img)

        # Rotate the frame
        if parameters['OrientSwitch']:
            img = v2.functional.rotate(img, angle=parameters['OrientSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        # Find all faces in frame and return a list of 5-pt kpss
        kpss = self.func_w_test("detect", self.models.run_detect, img, parameters['DetectTypeTextSel'], max_num=50,
                                score=parameters['DetectScoreSlider'] / 100.0)

        ret = []
        angles = []
        for face_kps in kpss:
            if face_kps is not None:
                face_emb, _ = self.func_w_test('recognize', self.models.run_recognize, img, face_kps)
                ret.append([face_kps, face_emb])
                rotation_angle = self.detect_face_rotation(face_kps)
                angles.append(rotation_angle)

        if ret:
            if parameters["ThresholdSlider"] == 0:
                best_matches = []
                if self.found_faces:
                    assigned_faces = [ff for ff in self.found_faces if ff["SourceFaceAssignments"]]
                    if assigned_faces:
                        s_e = assigned_faces[0]["AssignedEmbedding"]
                        for i, (face_kps, _) in enumerate(ret):
                            best_matches.append(([face_kps, None], assigned_faces[0], angles[i]))

                self.track.append({
                    "frame": frame_number,
                    "target_face_count": len(ret),
                    "faces_swapped": len(best_matches),
                    "similarities": [
                        {"source_index": 0, "sim": 100.0}
                    ] * len(best_matches),
                    "angles": [
                        {"source_index": 0, "angle": angle} for (_, _, angle) in best_matches
                    ]
                })

            else:
                best_matches = []
                max_sims_per_source = {
                    i: 0.0 for i, ff in enumerate(self.found_faces) if ff["SourceFaceAssignments"]
                }
                match_angles = []

                for i, fface in enumerate(ret):
                    best_sim = 0.0
                    best_match = None
                    best_src_idx = None

                    for source_index, found_face in enumerate(self.found_faces):
                        if not found_face["SourceFaceAssignments"]:
                            continue

                        sim = self.findCosineDistance(fface[1], found_face["Embedding"])
                        threshold = float(parameters["ThresholdSlider"])

                        if sim > max_sims_per_source[source_index]:
                            max_sims_per_source[source_index] = sim

                        if sim > best_sim and sim >= threshold:
                            best_sim = sim
                            best_match = (fface, found_face, angles[i])
                            best_src_idx = source_index

                    if best_match:
                        best_matches.append(best_match)
                        match_angles.append({"source_index": best_src_idx, "angle": angles[i]})

                self.track.append({
                    "frame": frame_number,
                    "target_face_count": len(ret),
                    "faces_swapped": len(best_matches),
                    "similarities": [
                        {"source_index": idx, "sim": sim}
                        for idx, sim in max_sims_per_source.items()
                    ],
                    "angles": match_angles
                })

            for fface, found_face, _ in best_matches:
                s_e = found_face["AssignedEmbedding"]
                img = self.func_w_test("swap_video", self.swap_core, img, fface[0], s_e, parameters, control)

            img = img.permute(1, 2, 0)
            if not control['MaskViewButton'] and parameters['OrientSwitch']:
                img = img.permute(2, 0, 1)
                img = transforms.functional.rotate(img, angle=-parameters['OrientSlider'], expand=True)
                img = img.permute(1, 2, 0)
        else:
            max_sims_per_source = {
                i: 0.0 for i, ff in enumerate(self.found_faces) if ff["SourceFaceAssignments"]
            }

            self.track.append({
                "frame": frame_number,
                "target_face_count": 0,
                "faces_swapped": 0,
                "similarities": [
                    {"source_index": idx, "sim": 0.0}
                    for idx in max_sims_per_source
                ],
                "angles": []
            })

            img = img.permute(1,2,0)
            if parameters['OrientSwitch']:
                img = img.permute(2,0,1)
                img = v2.functional.rotate(img, angle=-parameters['OrientSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)
                img = img.permute(1,2,0)

        if self.perf_test:
            print('------------------------')

        if img_x <512 or img_y < 512:
            tscale = v2.Resize((img_y, img_x), antialias=True)
            img = img.permute(2,0,1)
            img = tscale(img)
            img = img.permute(1,2,0)

        img = img.cpu().numpy()
        return img.astype(np.uint8)

    # @profile
    def swap_video_orig2(self, target_image, frame_number, use_markers):

        parameters = self.parameters.copy()
        control = self.control.copy()

        desired_parameters = {'OrientSwitch', 'OrientSlider', 'MouthParserSlider'}

        if self.markers and use_markers:
            temp = [marker['frame'] for marker in self.markers]
            idx = bisect.bisect(temp, frame_number)

            if frame_number in temp:
                idx = temp.index(frame_number)
                for key in desired_parameters:
                    if key in self.markers[idx]['parameters']:
                        parameters[key] = self.markers[idx]['parameters'][key]

        img = torch.from_numpy(target_image.astype('uint8')).to('cuda')
        img = img.permute(2, 0, 1)

        img_x = img.size()[2]
        img_y = img.size()[1]

        if img_x < 512 or img_y < 512:
            if img_x <= img_y:
                tscale = v2.Resize((int(512 * img_y / img_x), 512), antialias=True)
            else:
                tscale = v2.Resize((512, int(512 * img_x / img_y)), antialias=True)
            img = tscale(img)

        if parameters['OrientSwitch']:
            img = v2.functional.rotate(img, angle=parameters['OrientSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        kpss = self.func_w_test("detect", self.models.run_detect, img, parameters['DetectTypeTextSel'], max_num=50,
                                score=parameters['DetectScoreSlider'] / 100.0)

        ret = []
        angles = []
        for face_kps in kpss:
            if face_kps is not None:
                face_emb, _ = self.func_w_test('recognize', self.models.run_recognize, img, face_kps)
                ret.append([face_kps, face_emb])
                angles.append(self.detect_face_rotation(face_kps))

        if ret:
            if parameters["ThresholdSlider"] == 0:
                best_matches = []
                if self.found_faces:
                    assigned_faces = [ff for ff in self.found_faces if ff["SourceFaceAssignments"]]
                    if assigned_faces:
                        s_e = assigned_faces[0]["AssignedEmbedding"]
                        for i, (face_kps, _) in enumerate(ret):
                            best_matches.append(([face_kps, None], assigned_faces[0], angles[i]))

                self.track.append({
                    "frame": frame_number,
                    "target_face_count": len(ret),
                    "faces_swapped": len(best_matches),
                    "similarities": [
                        {"source_index": 0, "sim": 100.0}
                    ] * len(best_matches),
                    "angles": [
                        {"source_index": 0, "angle": angle} for (_, _, angle) in best_matches
                    ]
                })

            else:
                best_matches = []
                max_sims_per_source = {
                    i: 0.0 for i, ff in enumerate(self.found_faces) if ff["SourceFaceAssignments"]
                }
                match_angles = []
                matched_indices = set()

                for i, fface in enumerate(ret):
                    best_sim = 0.0
                    best_match = None
                    best_src_idx = None

                    for source_index, found_face in enumerate(self.found_faces):
                        if not found_face["SourceFaceAssignments"]:
                            continue

                        sim = self.findCosineDistance(fface[1], found_face["Embedding"])
                        threshold = float(parameters["ThresholdSlider"])

                        if sim > max_sims_per_source[source_index]:
                            max_sims_per_source[source_index] = sim

                        if sim > best_sim and sim >= threshold:
                            best_sim = sim
                            best_match = (fface, found_face, angles[i])
                            best_src_idx = source_index

                    if best_match:
                        best_matches.append(best_match)
                        match_angles.append({"source_index": best_src_idx, "angle": angles[i]})
                        matched_indices.add(i)

                # Fallback for 2 target, 2 source, 1 matched case
                if (
                    parameters["ThresholdSlider"] > 0 and
                    len(ret) == 2 and
                    len([ff for ff in self.found_faces if ff["SourceFaceAssignments"]]) == 2 and
                    len(best_matches) == 1
                ):
                    fallback_idx = (set([0, 1]) - matched_indices).pop()
                    fface = ret[fallback_idx]
                    fallback_angle = angles[fallback_idx]

                    for source_index, found_face in enumerate(self.found_faces):
                        if not found_face["SourceFaceAssignments"]:
                            continue

                        sim = self.findCosineDistance(fface[1], found_face["Embedding"])
                        if sim > 40.0:
                            best_matches.append((fface, found_face, fallback_angle))
                            match_angles.append({"source_index": source_index, "angle": fallback_angle})
                            break

                self.track.append({
                    "frame": frame_number,
                    "target_face_count": len(ret),
                    "faces_swapped": len(best_matches),
                    "similarities": [
                        {"source_index": idx, "sim": sim}
                        for idx, sim in max_sims_per_source.items()
                    ],
                    "angles": match_angles
                })

            for fface, found_face, _ in best_matches:
                s_e = found_face["AssignedEmbedding"]
                img = self.func_w_test("swap_video", self.swap_core, img, fface[0], s_e, parameters, control)

            img = img.permute(1, 2, 0)
            if not control['MaskViewButton'] and parameters['OrientSwitch']:
                img = img.permute(2, 0, 1)
                img = transforms.functional.rotate(img, angle=-parameters['OrientSlider'], expand=True)
                img = img.permute(1, 2, 0)
        else:
            max_sims_per_source = {
                i: 0.0 for i, ff in enumerate(self.found_faces) if ff["SourceFaceAssignments"]
            }

            self.track.append({
                "frame": frame_number,
                "target_face_count": 0,
                "faces_swapped": 0,
                "similarities": [
                    {"source_index": idx, "sim": 0.0} for idx in max_sims_per_source
                ],
                "angles": []
            })

            img = img.permute(1, 2, 0)
            if parameters['OrientSwitch']:
                img = img.permute(2, 0, 1)
                img = v2.functional.rotate(img, angle=-parameters['OrientSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)
                img = img.permute(1, 2, 0)

        if self.perf_test:
            print('------------------------')

        if img_x < 512 or img_y < 512:
            tscale = v2.Resize((img_y, img_x), antialias=True)
            img = img.permute(2, 0, 1)
            img = tscale(img)
            img = img.permute(1, 2, 0)

        img = img.cpu().numpy()
        return img.astype(np.uint8)


    # @profile
    def swap_video(self, target_image, frame_number, use_markers):

        parameters = self.parameters.copy()
        control = self.control.copy()

        desired_parameters = {'OrientSwitch', 'OrientSlider', 'MouthParserSlider'}

        if self.markers and use_markers:
            temp = [marker['frame'] for marker in self.markers]
            idx = bisect.bisect(temp, frame_number)

            if frame_number in temp:
                idx = temp.index(frame_number)
                for key in desired_parameters:
                    if key in self.markers[idx]['parameters']:
                        parameters[key] = self.markers[idx]['parameters'][key]

        img = torch.from_numpy(target_image.astype('uint8')).to('cuda')
        img = img.permute(2, 0, 1)

        img_x = img.size()[2]
        img_y = img.size()[1]

        if img_x < 512 or img_y < 512:
            if img_x <= img_y:
                tscale = v2.Resize((int(512 * img_y / img_x), 512), antialias=True)
            else:
                tscale = v2.Resize((512, int(512 * img_x / img_y)), antialias=True)
            img = tscale(img)

        if parameters['OrientSwitch']:
            img = v2.functional.rotate(img, angle=parameters['OrientSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        kpss = self.func_w_test("detect", self.models.run_detect, img, parameters['DetectTypeTextSel'], max_num=50,
                                score=parameters['DetectScoreSlider'] / 100.0)

        ret = []
        angles = []
        for face_kps in kpss:
            if face_kps is not None:
                face_emb, _ = self.func_w_test('recognize', self.models.run_recognize, img, face_kps)
                ret.append([face_kps, face_emb])
                angles.append(self.detect_face_rotation(face_kps))

        if ret:
            if parameters["ThresholdSlider"] == 0:
                best_matches = []
                if self.found_faces:
                    assigned_faces = [ff for ff in self.found_faces if ff["SourceFaceAssignments"]]
                    if assigned_faces:
                        s_e = assigned_faces[0]["AssignedEmbedding"]
                        for i, (face_kps, _) in enumerate(ret):
                            best_matches.append(([face_kps, None], assigned_faces[0], angles[i]))

                self.track.append({
                    "frame": frame_number,
                    "target_face_count": len(ret),
                    "faces_swapped": len(best_matches),
                    "similarities": [
                        {"source_index": 0, "sim": 100.0}
                    ] * len(best_matches),
                    "angles": [
                        {"source_index": 0, "angle": angle} for (_, _, angle) in best_matches
                    ]
                })

            else:
                best_matches = []
                max_sims_per_source = {
                    i: 0.0 for i, ff in enumerate(self.found_faces) if ff["SourceFaceAssignments"]
                }
                match_angles = []
                matched_indices = set()

                for i, fface in enumerate(ret):
                    best_sim = 0.0
                    best_match = None
                    best_src_idx = None

                    for source_index, found_face in enumerate(self.found_faces):
                        if not found_face["SourceFaceAssignments"]:
                            continue

                        sim = self.findCosineDistance(fface[1], found_face["Embedding"])
                        threshold = float(parameters["ThresholdSlider"])

                        if sim > max_sims_per_source[source_index]:
                            max_sims_per_source[source_index] = sim

                        if sim > best_sim and sim >= threshold:
                            best_sim = sim
                            best_match = (fface, found_face, angles[i])
                            best_src_idx = source_index

                    if best_match:
                        best_matches.append(best_match)
                        match_angles.append({"source_index": best_src_idx, "angle": angles[i]})
                        matched_indices.add(i)

                # Unified fallback block for any unmatched faces when 2 sources exist
                assigned_sources = [ff for ff in self.found_faces if ff["SourceFaceAssignments"]]
                unmatched_indices = set(range(len(ret))) - matched_indices

                if len(assigned_sources) == 2 and unmatched_indices:
                    fallback_threshold = 40.0
                    for fallback_idx in unmatched_indices:
                        fface = ret[fallback_idx]
                        fallback_angle = angles[fallback_idx]

                        best_sim = 0.0
                        best_source = None
                        best_src_idx = None

                        for source_index, found_face in enumerate(self.found_faces):
                            if not found_face["SourceFaceAssignments"]:
                                continue

                            sim = self.findCosineDistance(fface[1], found_face["Embedding"])
                            if sim > fallback_threshold and sim > best_sim:
                                best_sim = sim
                                best_source = found_face
                                best_src_idx = source_index

                        if best_source is not None:
                            best_matches.append((fface, best_source, fallback_angle))
                            match_angles.append({"source_index": best_src_idx, "angle": fallback_angle})

                self.track.append({
                    "frame": frame_number,
                    "target_face_count": len(ret),
                    "faces_swapped": len(best_matches),
                    "similarities": [
                        {"source_index": idx, "sim": sim}
                        for idx, sim in max_sims_per_source.items()
                    ],
                    "angles": match_angles
                })

            for fface, found_face, _ in best_matches:
                s_e = found_face["AssignedEmbedding"]
                img = self.func_w_test("swap_video", self.swap_core, img, fface[0], s_e, parameters, control)

            img = img.permute(1, 2, 0)
            if not control['MaskViewButton'] and parameters['OrientSwitch']:
                img = img.permute(2, 0, 1)
                img = transforms.functional.rotate(img, angle=-parameters['OrientSlider'], expand=True)
                img = img.permute(1, 2, 0)
        else:
            max_sims_per_source = {
                i: 0.0 for i, ff in enumerate(self.found_faces) if ff["SourceFaceAssignments"]
            }

            self.track.append({
                "frame": frame_number,
                "target_face_count": 0,
                "faces_swapped": 0,
                "similarities": [
                    {"source_index": idx, "sim": 0.0} for idx in max_sims_per_source
                ],
                "angles": []
            })

            img = img.permute(1, 2, 0)
            if parameters['OrientSwitch']:
                img = img.permute(2, 0, 1)
                img = v2.functional.rotate(img, angle=-parameters['OrientSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)
                img = img.permute(1, 2, 0)

        if self.perf_test:
            print('------------------------')

        if img_x < 512 or img_y < 512:
            tscale = v2.Resize((img_y, img_x), antialias=True)
            img = img.permute(2, 0, 1)
            img = tscale(img)
            img = img.permute(1, 2, 0)

        img = img.cpu().numpy()
        return img.astype(np.uint8)



    def findCosineDistance(self, vector1, vector2):
        cos_dist = 1.0 - np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2)) # 2..0

        return 100.0-cos_dist*50.0



    def func_w_test(self, name, func, *args, **argsv):
        timing = time.time()
        result = func(*args, **argsv)
        if self.perf_test:
            print(name, round(time.time()-timing, 5), 's')
        return result

    # @profile    
    def swap_core(self, img, kps, s_e, parameters, control): # img = RGB
        # 512 transforms
        dst = self.arcface_dst * 4.0
        dst[:,0] += 32.0
        
        # Change the ref points
        if parameters['FaceAdjSwitch']:
            dst[:,0] += parameters['KPSXSlider']
            dst[:,1] += parameters['KPSYSlider']
            dst[:,0] -= 255
            dst[:,0] *= (1+parameters['KPSScaleSlider']/100)
            dst[:,0] += 255
            dst[:,1] -= 255
            dst[:,1] *= (1+parameters['KPSScaleSlider']/100)
            dst[:,1] += 255

        tform = trans.SimilarityTransform()
        tform.estimate(kps, dst) 

        # Scaling Transforms
        t512 = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t128 = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        # Grab 512 face from image and create 256 and 128 copys
        original_face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=v2.InterpolationMode.BILINEAR )
        original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)# 3, 512, 512
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)
        # print(f"Rotation angle: {tform.inverse.rotation * 57.2958} degrees")

        latent = torch.from_numpy(self.models.calc_swapper_latent(s_e)).float().to('cuda')

        dim = 1
        if parameters['SwapperTypeTextSel'] == '128':
            dim = 1
            input_face_affined = original_face_128
        elif parameters['SwapperTypeTextSel'] == '256':
            dim = 2
            input_face_affined = original_face_256
        elif parameters['SwapperTypeTextSel'] == '512':
            dim = 4
            input_face_affined = original_face_512

        # Optional Scaling # change the thransform matrix
        if parameters['FaceAdjSwitch']:
            input_face_affined = v2.functional.affine(input_face_affined, 0, (0, 0), 1 + parameters['FaceScaleSlider'] / 100, 0, center=(dim*128-1, dim*128-1), interpolation=v2.InterpolationMode.BILINEAR)

        itex = 1
        if parameters['StrengthSwitch']:
            itex = ceil(parameters['StrengthSlider'] / 100.)

        output_size = int(128 * dim)
        output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device='cuda')
        input_face_affined = input_face_affined.permute(1, 2, 0)
        input_face_affined = torch.div(input_face_affined, 255.0)

        for k in range(itex):
            for j in range(dim):
                for i in range(dim):
                    input_face_disc = input_face_affined[j::dim,i::dim]
                    input_face_disc = input_face_disc.permute(2, 0, 1)
                    input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()

                    swapper_output = torch.empty((1,3,128,128), dtype=torch.float32, device='cuda').contiguous()
                    self.models.run_swapper(input_face_disc, latent, swapper_output)

                    swapper_output = torch.squeeze(swapper_output)
                    swapper_output = swapper_output.permute(1, 2, 0)


                    output[j::dim, i::dim] = swapper_output.clone()
            prev_face = input_face_affined.clone()
            input_face_affined = output.clone()
            output = torch.mul(output, 255)
            output = torch.clamp(output, 0, 255)


        output = output.permute(2, 0, 1)


        swap = t512(output)

        if parameters['StrengthSwitch']:
            if itex == 0:
                swap = original_face_512.clone()
            else:
                alpha = np.mod(parameters['StrengthSlider'], 100)*0.01
                if alpha==0:
                    alpha=1

                # Blend the images
                prev_face = torch.mul(prev_face, 255)
                prev_face = torch.clamp(prev_face, 0, 255)
                prev_face = prev_face.permute(2, 0, 1)
                prev_face = t512(prev_face)
                swap = torch.mul(swap, alpha)
                prev_face = torch.mul(prev_face, 1-alpha)
                swap = torch.add(swap, prev_face)

        def rgb_to_hsv(img):
            """Convert RGB image to HSV (PyTorch implementation)."""
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

            maxc, _ = torch.max(img, dim=2)
            minc, _ = torch.min(img, dim=2)
            delta = maxc - minc

            # Hue calculation
            hue = torch.zeros_like(maxc)
            mask = delta > 0
            idx = (maxc == r) & mask
            hue[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6
            idx = (maxc == g) & mask
            hue[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2
            idx = (maxc == b) & mask
            hue[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4
            hue /= 6  # Normalize to 0-1
            hue[~mask] = 0  # Undefined hue when delta == 0

            # Saturation calculation
            sat = torch.zeros_like(maxc)
            sat[maxc > 0] = delta[maxc > 0] / maxc[maxc > 0]

            # Value calculation
            val = maxc

            return torch.stack([hue, sat, val], dim=2)

        def hsv_to_rgb(img):
            """Convert HSV image to RGB (PyTorch implementation)."""
            h, s, v = img[:, :, 0], img[:, :, 1], img[:, :, 2]

            h = (h * 6) % 6  # Convert hue to range [0,6]
            i = torch.floor(h).to(torch.int32)
            f = h - i
            p = v * (1 - s)
            q = v * (1 - s * f)
            t = v * (1 - s * (1 - f))

            rgb = torch.stack([
                torch.where((i == 0) | (i == 5), v,
                            torch.where((i == 1), q, torch.where((i == 2), p, torch.where((i == 3), p, t)))),
                torch.where((i == 1) | (i == 2), v,
                            torch.where((i == 3), q, torch.where((i == 4), p, torch.where((i == 5), p, t)))),
                torch.where((i == 3) | (i == 4), v,
                            torch.where((i == 5), q, torch.where((i == 0), p, torch.where((i == 1), p, t))))
            ], dim=2)

            return rgb

            # swap = torch.squeeze(swap)
            # swap = torch.mul(swap, 255)
            # swap = torch.clamp(swap, 0, 255)
            # # swap_128 = swap
            # swap = t256(swap)
            # swap = t512(swap)

        
        # Apply color corerctions
        # is_hsv = parameters.get('HSVSwitchState', False)  # Read from GUI switch


        if parameters['ColorSwitch']:
            # Apply gamma correction
            swap = torch.unsqueeze(swap, 0)
            swap = v2.functional.adjust_gamma(swap, parameters['ColorGammaSlider'], 1.0)
            swap = torch.squeeze(swap)
            swap = swap.permute(1, 2, 0).type(torch.float32)

            if parameters['HSVSwitch']:
                # **Convert to HSV**
                swap = swap / 255.0  # Normalize before HSV conversion
                swap_hsv = rgb_to_hsv(swap)

                # **Repurpose RGB sliders for HSV adjustments**
                h_shift = parameters['ColorRedSlider'] / 720.0  # Convert degrees to normalized shift
                s_factor = parameters['ColorGreenSlider'] / 200.0  # Convert percentage to scale
                v_factor = parameters['ColorBlueSlider'] / 100.0  # Convert percentage to scale

                # **Apply adjustments**
                swap_hsv[:, :, 0] = (swap_hsv[:, :, 0] + h_shift) % 1.0  # Hue shift (wrap around)
                swap_hsv[:, :, 1] = torch.clamp(swap_hsv[:, :, 1] * (1 + s_factor), 0, 1)  # Saturation scale
                swap_hsv[:, :, 2] = torch.clamp(swap_hsv[:, :, 2] * (1 + v_factor), 0,
                                                1)  # **Fixed Brightness scaling**

                # **Convert back to RGB**
                swap = hsv_to_rgb(swap_hsv)
                swap = torch.clamp(swap * 255, 0, 255)  # De-normalize

                swap = swap.permute(2, 0, 1).type(torch.uint8)  # Ensure final format


            else:
                # **Original RGB Adjustments (No Changes)**
                del_color = torch.tensor(
                    [parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']],
                    device=swap.device
                )
                swap += del_color
                swap = torch.clamp(swap, min=0., max=255.)
                swap = swap.permute(2, 0, 1).type(torch.uint8)

        # Create border mask
        border_mask = torch.ones((128, 128), dtype=torch.float32, device=device)
        border_mask = torch.unsqueeze(border_mask,0)
        
        # if parameters['BorderState']:
        top = parameters['BorderTopSlider']
        left = parameters['BorderSidesSlider']
        right = 128-parameters['BorderSidesSlider']
        bottom = 128-parameters['BorderBottomSlider']

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        gauss = transforms.GaussianBlur(parameters['BorderBlurSlider']*2+1, (parameters['BorderBlurSlider']+1)*0.2)
        border_mask = gauss(border_mask)        

        # Create image mask
        swap_mask = torch.ones((128, 128), dtype=torch.float32, device=device)
        swap_mask = torch.unsqueeze(swap_mask,0)    

        # Face Diffing
        if parameters["DiffSwitch"]:
            mask = self.apply_fake_diff(swap, original_face_512, parameters["DiffSlider"])
            # mask = t128(mask)
            gauss = transforms.GaussianBlur(parameters['BlendSlider']*2+1, (parameters['BlendSlider']+1)*0.2)
            mask = gauss(mask.type(torch.float32)) 
            swap = swap*mask + original_face_512*(1-mask)
 
        # Restorer
        if parameters["RestorerSwitch"]: 
            swap = self.func_w_test('Restorer', self.apply_restorer, swap, parameters)  
        
            
        # Occluder
        if parameters["OccluderSwitch"]:
            mask = self.func_w_test('occluder', self.apply_occlusion , original_face_256, parameters["OccluderSlider"])
            mask = t128(mask)  
            swap_mask = torch.mul(swap_mask, mask)


        if parameters["FaceParserSwitch"]:
            mask = self.apply_face_parser(swap, parameters["FaceParserSlider"], parameters['MouthParserSlider'])
            mask = t128(mask)
            swap_mask = torch.mul(swap_mask, mask)            
        
        # CLIPs 
        if parameters["CLIPSwitch"]:
            with lock:
                mask = self.func_w_test('CLIP', self.apply_CLIPs, original_face_512, parameters["CLIPTextEntry"], parameters["CLIPSlider"])
            mask = cv2.resize(mask, (128,128))
            mask = torch.from_numpy(mask).to('cuda')
            swap_mask *= mask


        # Add blur to swap_mask results
        gauss = transforms.GaussianBlur(parameters['BlendSlider']*2+1, (parameters['BlendSlider']+1)*0.2)
        swap_mask = gauss(swap_mask)  
        

        # Combine border and swap mask, scale, and apply to swap
        swap_mask = torch.mul(swap_mask, border_mask)
        swap_mask = t512(swap_mask)
        swap = torch.mul(swap, swap_mask)

        if not control['MaskViewButton']:
            # Cslculate the area to be mergerd back to the original frame
            IM512 = tform.inverse.params[0:2, :]
            corners = np.array([[0,0], [0,511], [511, 0], [511, 511]])
            x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
            y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])
            
            left = floor(np.min(x))
            if left<0:
                left=0
            top = floor(np.min(y))
            if top<0: 
                top=0
            right = ceil(np.max(x))
            if right>img.shape[2]:
                right=img.shape[2]            
            bottom = ceil(np.max(y))
            if bottom>img.shape[1]:
                bottom=img.shape[1]   

            # Untransform the swap
            swap = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
            swap = v2.functional.affine(swap, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0,interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
            swap = swap[0:3, top:bottom, left:right]
            swap = swap.permute(1, 2, 0)


            # Untransform the swap mask
            swap_mask = v2.functional.pad(swap_mask, (0,0,img.shape[2]-512, img.shape[1]-512))
            swap_mask = v2.functional.affine(swap_mask, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
            swap_mask = swap_mask[0:1, top:bottom, left:right]                        
            swap_mask = swap_mask.permute(1, 2, 0)
            swap_mask = torch.sub(1, swap_mask) 

            # Apply the mask to the original image areas
            img_crop = img[0:3, top:bottom, left:right]
            img_crop = img_crop.permute(1,2,0)            
            img_crop = torch.mul(swap_mask,img_crop)
            
            #Add the cropped areas and place them back into the original image
            swap = torch.add(swap, img_crop)
            swap = swap.type(torch.uint8)
            swap = swap.permute(2,0,1)
            img[0:3, top:bottom, left:right] = swap  

        else:
            # Invert swap mask
            swap_mask = torch.sub(1, swap_mask)
            
            # Combine preswapped face with swap
            original_face_512 = torch.mul(swap_mask, original_face_512)
            original_face_512 = torch.add(swap, original_face_512)            
            original_face_512 = original_face_512.type(torch.uint8)
            original_face_512 = original_face_512.permute(1, 2, 0)

            # Uninvert and create image from swap mask
            swap_mask = torch.sub(1, swap_mask) 
            swap_mask = torch.cat((swap_mask,swap_mask,swap_mask),0)
            swap_mask = swap_mask.permute(1, 2, 0)

            # Place them side by side
            img = torch.hstack([original_face_512, swap_mask*255])
            img = img.permute(2,0,1)

        return img
        
    # @profile    
    def apply_occlusion(self, img, amount):        
        img = torch.div(img, 255)
        img = torch.unsqueeze(img, 0)
        outpred = torch.ones((256,256), dtype=torch.float32, device=device).contiguous()
        
        self.models.run_occluder(img, outpred)        
                
        outpred = torch.squeeze(outpred)
        outpred = (outpred > 0)
        outpred = torch.unsqueeze(outpred, 0).type(torch.float32)
        
        if amount >0:                   
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

            for i in range(int(amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                outpred = torch.clamp(outpred, 0, 1)
            
            outpred = torch.squeeze(outpred)
            
        if amount <0:      
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

            for i in range(int(-amount)):
                outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))       
                outpred = torch.clamp(outpred, 0, 1)
            
            outpred = torch.squeeze(outpred)
            outpred = torch.neg(outpred)
            outpred = torch.add(outpred, 1)
            
        outpred = torch.reshape(outpred, (1, 256, 256)) 
        return outpred         
    
      
    def apply_CLIPs(self, img, CLIPText, CLIPAmount):
        clip_mask = np.ones((352, 352))
        img = img.permute(1,2,0)
        img = img.cpu().numpy()
        # img = img.to(torch.float)
        # img = img.permute(1,2,0)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                                        transforms.Resize((352, 352))])
        CLIPimg = transform(img).unsqueeze(0)
        
        if CLIPText != "":
            prompts = CLIPText.split(',')

            with torch.no_grad():
                preds = self.clip_session(CLIPimg.repeat(len(prompts),1,1,1), prompts)[0]
                # preds = self.clip_session(CLIPimg,  maskimg, True)[0]

            clip_mask = 1 - torch.sigmoid(preds[0][0])
            for i in range(len(prompts)-1):
                clip_mask *= 1-torch.sigmoid(preds[i+1][0])
            clip_mask = clip_mask.data.cpu().numpy()
            
            thresh = CLIPAmount/100.0
            clip_mask[clip_mask>thresh] = 1.0
            clip_mask[clip_mask<=thresh] = 0.0
        return clip_mask    
        
    # @profile
    def apply_face_parser(self, img, FaceAmount, MouthAmount):

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
       
        outpred = torch.ones((512,512), dtype=torch.float32, device='cuda').contiguous()
        

        img = torch.div(img, 255)
        img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = torch.reshape(img, (1, 3, 512, 512))
        outpred = torch.empty((1,19,512,512), dtype=torch.float32, device='cuda').contiguous()

        self.models.run_faceparser(img, outpred)

        outpred = torch.squeeze(outpred)
        outpred = torch.argmax(outpred, 0)

        # Mouth Parse
        if MouthAmount <0:
            mouth_idxs = torch.tensor([11], device='cuda')
            iters = int(-MouthAmount)

            mouth_parse = torch.isin(outpred, mouth_idxs)
            mouth_parse = torch.clamp(~mouth_parse, 0, 1).type(torch.float32)
            mouth_parse = torch.reshape(mouth_parse, (1, 1, 512, 512))
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)

            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32,
                                device='cuda')

            for i in range(iters):
                mouth_parse = torch.nn.functional.conv2d(mouth_parse, kernel,
                                                         padding=(1, 1))
                mouth_parse = torch.clamp(mouth_parse, 0, 1)

            mouth_parse = torch.squeeze(mouth_parse)
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)
            mouth_parse = torch.reshape(mouth_parse, (1, 512, 512))

        elif MouthAmount >0:
            mouth_idxs = torch.tensor([11,12,13], device='cuda')
            iters = int(MouthAmount)

            mouth_parse = torch.isin(outpred, mouth_idxs)
            mouth_parse = torch.clamp(~mouth_parse, 0, 1).type(torch.float32)
            mouth_parse = torch.reshape(mouth_parse, (1,1,512,512))
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)

            kernel = torch.ones((1,1,3,3), dtype=torch.float32, device='cuda')

            for i in range(iters):
                mouth_parse = torch.nn.functional.conv2d(mouth_parse, kernel, padding=(1, 1))
                mouth_parse = torch.clamp(mouth_parse, 0, 1)

            mouth_parse = torch.squeeze(mouth_parse)
            mouth_parse = torch.neg(mouth_parse)
            mouth_parse = torch.add(mouth_parse, 1)
            mouth_parse = torch.reshape(mouth_parse, (1, 512, 512))

        else:
            mouth_parse = torch.ones((1, 512, 512), dtype=torch.float32, device='cuda')

        # BG Parse
        bg_idxs = torch.tensor([0, 14, 15, 16, 17, 18], device=device)
        bg_parse = torch.isin(outpred, bg_idxs)
        bg_parse = torch.clamp(~bg_parse, 0, 1).type(torch.float32)
        bg_parse = torch.reshape(bg_parse, (1, 1, 512, 512))

        if FaceAmount > 0:
            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)

            for i in range(int(FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))
                bg_parse = torch.clamp(bg_parse, 0, 1)

            bg_parse = torch.squeeze(bg_parse)

        elif FaceAmount < 0:
            bg_parse = torch.neg(bg_parse)
            bg_parse = torch.add(bg_parse, 1)

            kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)

            for i in range(int(-FaceAmount)):
                bg_parse = torch.nn.functional.conv2d(bg_parse, kernel, padding=(1, 1))
                bg_parse = torch.clamp(bg_parse, 0, 1)

            bg_parse = torch.squeeze(bg_parse)
            bg_parse = torch.neg(bg_parse)
            bg_parse = torch.add(bg_parse, 1)
            bg_parse = torch.reshape(bg_parse, (1, 512, 512))
        else:
            bg_parse = torch.ones((1,512,512), dtype=torch.float32, device='cuda')

        out_parse = torch.mul(bg_parse, mouth_parse)

        return out_parse

    def apply_bg_face_parser(self, img, FaceParserAmount):

        # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
        # out = np.ones((512, 512), dtype=np.float32)  
        
        outpred = torch.ones((512,512), dtype=torch.float32, device='cuda').contiguous()

        # turn mouth parser off at 0 so someone can just use the mouth parser
        if FaceParserAmount != 0:
            img = torch.div(img, 255)
            img = v2.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            img = torch.reshape(img, (1, 3, 512, 512))      
            outpred = torch.empty((1,19,512,512), dtype=torch.float32, device=device).contiguous()

            self.models.run_faceparser(img, outpred)

            outpred = torch.squeeze(outpred)
            outpred = torch.argmax(outpred, 0)

            test = torch.tensor([ 0, 14, 15, 16, 17, 18], device=device)
            outpred = torch.isin(outpred, test)
            outpred = torch.clamp(~outpred, 0, 1).type(torch.float32)
            outpred = torch.reshape(outpred, (1,1,512,512))
            
            if FaceParserAmount >0:                   
                kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

                for i in range(int(FaceParserAmount)):
                    outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                    outpred = torch.clamp(outpred, 0, 1)
                
                outpred = torch.squeeze(outpred)
                
            if FaceParserAmount <0:      
                outpred = torch.neg(outpred)
                outpred = torch.add(outpred, 1)

                kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device)

                for i in range(int(-FaceParserAmount)):
                    outpred = torch.nn.functional.conv2d(outpred, kernel, padding=(1, 1))
                    outpred = torch.clamp(outpred, 0, 1)
                
                outpred = torch.squeeze(outpred)
                outpred = torch.neg(outpred)
                outpred = torch.add(outpred, 1)

        outpred = torch.reshape(outpred, (1, 512, 512))
        
        return outpred
    

        
    def apply_restorer(self, swapped_face_upscaled, parameters):     
        temp = swapped_face_upscaled
        t512 = v2.Resize((512, 512), antialias=False)
        t256 = v2.Resize((256, 256), antialias=False)  
        
        # If using a separate detection mode
        if parameters['RestorerDetTypeTextSel'] == 'Blend' or parameters['RestorerDetTypeTextSel'] == 'Reference':
            if parameters['RestorerDetTypeTextSel'] == 'Blend':
                # Set up Transformation
                dst = self.arcface_dst * 4.0
                dst[:,0] += 32.0        

            elif parameters['RestorerDetTypeTextSel'] == 'Reference':
                try:
                    dst = self.models.resnet50(swapped_face_upscaled, score=parameters['DetectScoreSlider']/100.0) 
                except:
                    return swapped_face_upscaled       
            
            tform = trans.SimilarityTransform()
            tform.estimate(dst, self.FFHQ_kps)

            # Transform, scale, and normalize
            temp = v2.functional.affine(swapped_face_upscaled, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0) )
            temp = v2.functional.crop(temp, 0,0, 512, 512)        
        
        temp = torch.div(temp, 255)
        temp = v2.functional.normalize(temp, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
        if parameters['RestorerTypeTextSel'] == 'GPEN256':
            temp = t256(temp)
        temp = torch.unsqueeze(temp, 0).contiguous()

        # Bindings
        outpred = torch.empty((1,3,512,512), dtype=torch.float32, device=device).contiguous()

        if parameters['RestorerTypeTextSel'] == 'GFPGAN':
            self.models.run_GFPGAN(temp, outpred)            
            
        elif parameters['RestorerTypeTextSel'] == 'CF':
            self.models.run_codeformer(temp, outpred) 
            
        elif parameters['RestorerTypeTextSel'] == 'GPEN256':
            outpred = torch.empty((1,3,256,256), dtype=torch.float32, device=device).contiguous()
            self.models.run_GPEN_256(temp, outpred) 
            
        elif parameters['RestorerTypeTextSel'] == 'GPEN512':
            self.models.run_GPEN_512(temp, outpred) 

        
        # Format back to cxHxW @ 255
        outpred = torch.squeeze(outpred)      
        outpred = torch.clamp(outpred, -1, 1)
        outpred = torch.add(outpred, 1)
        outpred = torch.div(outpred, 2)
        outpred = torch.mul(outpred, 255)
        if parameters['RestorerTypeTextSel'] == 'GPEN256':
            outpred = t512(outpred)
            
        # Invert Transform
        if parameters['RestorerDetTypeTextSel'] == 'Blend' or parameters['RestorerDetTypeTextSel'] == 'Reference':
            outpred = v2.functional.affine(outpred, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )

        # Blend
        alpha = float(parameters["RestorerSlider"])/100.0  
        outpred = torch.add(torch.mul(outpred, alpha), torch.mul(swapped_face_upscaled, 1-alpha))

        return outpred        
        
    def apply_fake_diff(self, swapped_face, original_face, DiffAmount):
        swapped_face = swapped_face.permute(1,2,0)
        original_face = original_face.permute(1,2,0)

        diff = swapped_face-original_face
        diff = torch.abs(diff)
        
        # Find the diffrence between the swap and original, per channel
        fthresh = DiffAmount*2.55
        
        # Bimodal
        diff[diff<fthresh] = 0
        diff[diff>=fthresh] = 1 
        
        # If any of the channels exceeded the threshhold, them add them to the mask
        diff = torch.sum(diff, dim=2)
        diff = torch.unsqueeze(diff, 2)
        diff[diff>0] = 1
        
        diff = diff.permute(2,0,1)

        return diff    
    

    
    def clear_mem(self):
        del self.swapper_model
        del self.GFPGAN_model
        del self.occluder_model
        del self.face_parsing_model
        del self.codeformer_model
        del self.GPEN_256_model
        del self.GPEN_512_model
        del self.resnet_model
        del self.detection_model
        del self.recognition_model
        
        self.swapper_model = []  
        self.GFPGAN_model = []
        self.occluder_model = []
        self.face_parsing_model = []
        self.codeformer_model = []
        self.GPEN_256_model = []
        self.GPEN_512_model = []
        self.resnet_model = []
        self.detection_model = []
        self.recognition_model = []
                
        # test = swap.permute(1, 2, 0)
        # test = test.cpu().numpy()
        # cv2.imwrite('2.jpg', test) 
