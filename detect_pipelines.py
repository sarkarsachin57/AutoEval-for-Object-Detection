from init import *


def detect_on_image_folder(folder_path, given_model, aux_model):

    folder_name = os.path.split(folder_path)[1]

    pred_results_json = {}
    
    pred_results_json['idx'] = []
    pred_results_json['raw_save_path'] = []
    pred_results_json['given_model_pred_results'] = []
    pred_results_json['aux_model_pred_results'] = []

    fps = FPS().start() 

    image_idx = 0

    image_files = [os.path.join(folder_path,file_name) for file_name in sorted(os.listdir(folder_path))]

    total_images = len(image_files)

    print('Total number of images found :', total_images)

    for image_file in tqdm(image_files):

        try:

            image = cv2.imread(image_file)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            given_model_pred_results = detect(rgb, model=given_model)
            aux_model_pred_results = detect(rgb, model=aux_model)         

            # cv2.imwrite(os.path.join(raw_frames_save_dir, str(frame_no)+'.jpg'), frame)
            
            pred_results_json['idx'].append(image_idx)
            pred_results_json['raw_save_path'].append(image_file)
            pred_results_json['given_model_pred_results'].append(given_model_pred_results)
            pred_results_json['aux_model_pred_results'].append(aux_model_pred_results)

            image_idx += 1

            
            fps.update()
            fps.stop()

            current_fps = round(fps.fps(), 2)

            eta = get_estd_processing_time(current_frame = image_idx, total_frame_count = total_images, current_fps = current_fps)

        except:

            

            raise ValueError(image_file)



    return folder_name, pred_results_json, None, False



def detect_on_video(vid_path, eval_fps, given_model, aux_model):

    vid_name = os.path.split(vid_path)[1]

    video = cv2.VideoCapture(vid_path)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    current_fps = video_fps

    eval_frame = int(max(1, video_fps // eval_fps))

    print('Video_FPS :', video_fps)
    print('eval_frame :', eval_frame)

    raw_frames_save_dir = os.path.join('process_out', 'video_raw_frames', vid_name)

    os.makedirs(raw_frames_save_dir, exist_ok=True)

    pred_results_json = {}
    
    pred_results_json['idx'] = []
    pred_results_json['frame_no'] = []
    pred_results_json['timestamp'] = []
    pred_results_json['raw_save_path'] = []
    pred_results_json['given_model_pred_results'] = []
    pred_results_json['aux_model_pred_results'] = []

    fps = FPS().start() 

    frame_idx = 0

    for frame_no in tqdm(range(video_frame_count)):

        _, frame = video.read()
        
        if frame is None:

            break


        if frame_no % eval_frame == 0:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            given_model_pred_results = detect(rgb, model=given_model)
            aux_model_pred_results = detect(rgb, model=aux_model)         

            time_stamp = frame_no_to_timestamp_str(frame_no, video_fps)

            raw_frames_save_path = os.path.join(raw_frames_save_dir, vid_name+'-'+str(time_stamp)+'.jpg')

            cv2.imwrite(raw_frames_save_path, frame)
            
            pred_results_json['idx'].append(frame_idx)
            pred_results_json['frame_no'].append(frame_no)
            pred_results_json['timestamp'].append(time_stamp)
            pred_results_json['raw_save_path'].append(raw_frames_save_path)
            pred_results_json['given_model_pred_results'].append(given_model_pred_results)
            pred_results_json['aux_model_pred_results'].append(aux_model_pred_results)

            frame_idx += 1

        
        fps.update()
        fps.stop()

        current_fps = round(fps.fps(), 2)

        eta = get_estd_processing_time(current_frame = frame_no, total_frame_count = video_frame_count, current_fps = current_fps)


    return vid_name, pred_results_json, eval_fps, True



def detect_on_multi_videos_or_folders(folder_path, eval_fps, given_model, aux_model):

    videos_or_image_folders = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path))]

    multi_pred_results_json = {}
    
    multi_pred_results_json['iter_no'] = []
    multi_pred_results_json['name'] = []
    multi_pred_results_json['pred_results_json'] = []
    multi_pred_results_json['eval_fps'] = []
    multi_pred_results_json['is_video'] = []

    print('\nGetting Detections...')

    for i,path in enumerate(videos_or_image_folders):

        print('\nIteration :',i)
        if os.path.isdir(path):
            print(f'Inferencing on {path} image folder.')
            name, pred_results_json, eval_fps, is_video = detect_on_image_folder(folder_path = path, given_model = given_model, aux_model = aux_model)
        else:
            print(f'Inferencing on {path} video.')
            name, pred_results_json, eval_fps, is_video = detect_on_video(vid_path = path, eval_fps = eval_fps, given_model = given_model, aux_model = aux_model)

        multi_pred_results_json['iter_no'].append(i)
        multi_pred_results_json['name'].append(name)
        multi_pred_results_json['pred_results_json'].append(pred_results_json)
        multi_pred_results_json['eval_fps'].append(eval_fps)
        multi_pred_results_json['is_video'].append(is_video)

    return multi_pred_results_json
