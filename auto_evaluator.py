from init_auto_eval import *

def get_false_detections(given_model_dets, aux_model_dets, iou_thresh):
    # format_input - given_model_dets : [[x1, y1, x2, y2, conf, class]] 
    # format_input - aux_model_dets : [[x1, y1, x2, y2, conf, class]] 
    pass

def get_low_conf_detections(given_model_dets, conf_thresh):
    # format_input - given_model_dets : [[x1, y1, x2, y2, conf, class]] 
    pass

def get_detections_with_high_overlap(given_model_dets, iou_thresh):
    # format_input - given_model_dets : [[x1, y1, x2, y2, conf, class]] 
    pass



def evaluate_on_video_or_stream(path, is_stream, conf_thresh, iou_thresh, trust_acc_thresh, trust_iou_thresh, debug, save_path = None):
    
    
    # print(f'\nvideo = {video}, conf_thresh = {conf_thresh}, iou_thresh = {iou_thresh}, trust_acc_thresh = {trust_acc_thresh}, trust_iou_thresh = {trust_iou_thresh}, debug = {debug}, save_path = {save_path}')

    video = cv2.VideoCapture(video)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_no = 0

    current_fps = video_fps

    eval_fps = 1

    eval_frame = int(max(1, video_fps // eval_fps))

    print('Video_FPS :', video_fps)
    print('eval_frame :', eval_frame)

    msg = None
    prev_msg = None

    fps = FPS().start() 

    writer = None

    while True:

        _, frame = video.read()
        
        if frame is None:

            break

        frame_processed = frame.copy()
        rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)



        if frame_no % eval_frame == 0:

            given_model_dets = detect(rgb, model=given_model, conf_thresh=conf_thresh, iou_thresh=iou_thresh).cpu().detach().numpy()
            aux_model_dets = detect(rgb, model=aux_model, conf_thresh=conf_thresh, iou_thresh=iou_thresh).cpu().detach().numpy()


        
        
        
        if debug:

            cv2.imshow('debug display - trust object detection', frame_processed)

            key = cv2.waitKey(1)

            if key == ord('q'):
              break


        fps.update()
        fps.stop()

        frame_no += 1
        current_fps = round(fps.fps(), 2)

        if save_path is not None and writer is None:   
            fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
            writer = cv2.VideoWriter(save_path, fourcc, video_fps, (frame_processed.shape[1], frame_processed.shape[0]), True)
            
        if writer is not None:
            writer.write(frame_processed)

    if writer is not None:
        writer.release()

        
    video.release()

    if debug:
        cv2.destroyWindow(f'debug display - trust object detection')


