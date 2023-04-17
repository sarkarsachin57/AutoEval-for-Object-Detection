from init import *

def get_false_detections(given_model_dets, aux_model_dets):
    # format_input - given_model_dets : [[x1, y1, x2, y2, conf, class]] 
    # format_input - aux_model_dets : [[x1, y1, x2, y2, conf, class]] 

    all_ious = []

    tp_dets, fp_dets, fn_dets = [], [], []

    for i,given_det in enumerate(given_model_dets):
        if len(aux_model_dets) != 0:
            ious = [get_iou(given_det[:4],aux_det[:4]) for aux_det in aux_model_dets]
            j = np.argmax(ious)
            if i == np.argmax([get_iou(given_det[:4],aux_model_dets[j][:4]) for given_det in given_model_dets]) and given_model_dets[i][-1] == aux_model_dets[j][-1] and get_iou(given_model_dets[i][:4],aux_model_dets[j][:4]) > eval_iou_thresh:
                all_ious.append(float(max(ious)))
                tp_dets.append(i)
            else:
                all_ious.append(0)
                fp_dets.append(i)
        else:
            all_ious.append(0)
            fp_dets.append(i)


    for i,aux_det in enumerate(aux_model_dets):
        if len(given_model_dets) != 0:
            ious = [get_iou(aux_det[:4],given_det[:4]) for given_det in given_model_dets]
            j = np.argmax(ious)
            if i == np.argmax([get_iou(aux_det[:4],given_model_dets[j][:4]) for aux_det in aux_model_dets]) and aux_model_dets[i][-1] == given_model_dets[j][-1] and get_iou(aux_model_dets[i][:4],given_model_dets[j][:4]) > eval_iou_thresh:
                pass
            else:
                all_ious.append(0)
                fn_dets.append(i)
        else:
            all_ious.append(0)
            fn_dets.append(i)

    miou = round(np.mean(all_ious), 4)

    return tp_dets, fp_dets, fn_dets, miou

    

def get_low_conf_detections(given_model_dets):
    # format_input - given_model_dets : [[x1, y1, x2, y2, conf, class]] 
    
    low_conf_ids = []
    for idx, det in enumerate(given_model_dets):
        x1, y1, x2, y2, conf, class_name = det
        if conf < conf_thresh:
            low_conf_ids.append(idx)

    return low_conf_ids 


def get_detections_with_high_overlap(given_model_dets):
    # format_input - given_model_dets : [[x1, y1, x2, y2, conf, class]] 

    overlap_det_ids = []
    for i,idet in enumerate(given_model_dets):
        for j,jdet in enumerate(given_model_dets):
            if i != j and get_iou(idet[:4],jdet[:4]) > eval_overlap_iou_thresh:
                overlap_det_ids.append([i,j])

    return overlap_det_ids





def show_false_detections(image, given_model_dets, aux_model_dets, tp_dets, fp_dets, fn_dets):

    image = image.copy()
    
    for i,det in enumerate(given_model_dets):

        if i in tp_dets:
            x1, y1, x2, y2, conf, class_name = det
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 255, 100), 2)
            draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (100, 255, 100))
                    
        elif i in fp_dets:
            x1, y1, x2, y2, conf, class_name = det
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 100, 255), 2)
            draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (100, 100, 255))
                    

    for i,det in enumerate(aux_model_dets):
        
        if i in fn_dets:
            x1, y1, x2, y2, conf, class_name = det
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 100), 2)
            draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (255, 100, 100))

    image = draw_text_center_top(image, 'Detections with True Positives, False Positives and Negetives', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, (200, 200, 200))
    
    return image
                    


def show_low_conf_detections(image, given_model_dets, low_conf_ids):

    image = image.copy()

    for i,det in enumerate(given_model_dets):

        if i in low_conf_ids:
            x1, y1, x2, y2, conf, class_name = det
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 100, 255), 2)
            draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (100, 100, 255))
                    
        else:
            x1, y1, x2, y2, conf, class_name = det
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 100), 2)
            draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (255, 100, 100))
        
    image = draw_text_center_top(image, 'Detections with low Confidence', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, (200, 200, 200))

    return image





def show_detections_with_high_overlap(image, given_model_dets, overlap_det_ids):

    image = image.copy()

    overlap_det_ids = np.unique(np.ravel(overlap_det_ids))

    for i,det in enumerate(given_model_dets):

        if i in overlap_det_ids:
            x1, y1, x2, y2, conf, class_name = det
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 100, 255), 2)
            draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (100, 100, 255))
                    
    image = draw_text_center_top(image, 'Detections with high overlap', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, (200, 200, 200))
    
    return image






def show_only_detections(image, given_model_dets):

    image = image.copy()

    for i,det in enumerate(given_model_dets):
        x1, y1, x2, y2, conf, class_name = det
        color = get_color(class_names.index(class_name)+1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, color)
                    
    image = draw_text_center_top(image, 'Detections Result', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, (200, 200, 200))

    return image





def evaluate_on_video_or_stream(path, is_stream = False, save_path = None):
    
    
    # print(f'\nvideo = {video}, conf_thresh = {conf_thresh}, iou_thresh = {iou_thresh}, trust_acc_thresh = {trust_acc_thresh}, trust_iou_thresh = {trust_iou_thresh}, debug = {debug}, save_path = {save_path}')

    video = cv2.VideoCapture(path)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_no = 0

    current_fps = video_fps

    eval_frame = int(max(1, video_fps // eval_fps))

    print('Video_FPS :', video_fps)
    print('eval_frame :', eval_frame)

    eval_status = 'ok'

    msg = None
    prev_msg = None

    fps = FPS().start() 

    writer = None

    for i in tqdm(range(video_frame_count)):

        _, frame = video.read()
        
        if frame is None:

            break

        frame_processed = frame.copy()
        rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)

        given_model_dets = detect(rgb, model=given_model, conf_thresh=0.25, iou_thresh=nms_iou_thresh)
        frame_with_dets = show_only_detections(frame_processed, given_model_dets)

        if frame_no % eval_frame == 0:

            aux_model_dets = detect(rgb, model=aux_model, conf_thresh=conf_thresh, iou_thresh=nms_iou_thresh)

            tp_dets, fp_dets, fn_dets, miou = get_false_detections(given_model_dets, aux_model_dets)
            low_conf_ids = get_low_conf_detections(given_model_dets)
            overlap_det_ids = get_detections_with_high_overlap(given_model_dets)

            tp = len(tp_dets)
            fp = len(fp_dets)
            fn = len(fn_dets)

            acc = round(tp / (tp+fp+fn), 4)
            eval_status = 'ok' if acc > eval_acc_thresh else 'poor'

            frame_false_dets = show_false_detections(frame_processed, given_model_dets, aux_model_dets, tp_dets, fp_dets, fn_dets)
            frame_low_conf_dets = show_low_conf_detections(frame_processed, given_model_dets, low_conf_ids)       
            frame_overlap_dets = show_detections_with_high_overlap(frame_processed, given_model_dets, overlap_det_ids)

            last_processed_frame_no = frame_no

        
        
        
        frame_processed_up = np.hstack([frame_with_dets, frame_false_dets])
        frame_processed_down = np.hstack([frame_low_conf_dets, frame_overlap_dets])
        frame_processed = np.vstack([frame_processed_up, frame_processed_down])


        frame_processed = draw_text_center_top(frame_processed, 'Automated Evaluation', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, (250, 100, 150))
        frame_processed = draw_text_center_bottom(frame_processed, f'Frame_no : {last_processed_frame_no}    ACC : {str(round(acc * 100, 2))}%    MIOU : {str(round(miou * 100, 2))}%    FPS : {str(current_fps)}    Detection Status : {eval_status}', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, (150, 180, 250))

        
        
        if debug:

            if frame_processed.shape[0] > frame_processed.shape[1]:
                frame_processed_show = imutils.resize(frame_processed, height=800)
            else:
                frame_processed_show = imutils.resize(frame_processed, width=1300)

            cv2.imshow('debug display - Auto Eval of object detection', frame_processed_show)

            key = cv2.waitKey(1)

            if key == ord('q'):
              break


        fps.update()
        fps.stop()

        frame_no += 1
        current_fps = round(fps.fps(), 2)

        if save_path is not None and writer is None:   
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
            writer = cv2.VideoWriter(save_path, fourcc, video_fps, (frame_processed.shape[1], frame_processed.shape[0]), True)
            
        if writer is not None:
            writer.write(frame_processed)
 
    if writer is not None:
        writer.release()

        
    video.release()

    if debug:
        cv2.destroyWindow(f'debug display - Auto Eval of object detection')




def evaluate_on_image(path, save_path = None):

    img = cv2.imread(path)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    given_model_dets = detect(rgb, model=given_model, conf_thresh=0.25, iou_thresh=nms_iou_thresh)

    aux_model_dets = detect(rgb, model=aux_model, conf_thresh=conf_thresh, iou_thresh=nms_iou_thresh)

    tp_dets, fp_dets, fn_dets, miou = get_false_detections(given_model_dets, aux_model_dets)
    low_conf_ids = get_low_conf_detections(given_model_dets)
    overlap_det_ids = get_detections_with_high_overlap(given_model_dets)

    tp = len(tp_dets)
    fp = len(fp_dets)
    fn = len(fn_dets)

    acc = round(tp / (tp+fp+fn), 4)
    eval_status = 'ok' if acc > eval_acc_thresh else 'poor'

    img_with_dets = show_only_detections(img, given_model_dets)    
    img_false_dets = show_false_detections(img, given_model_dets, aux_model_dets, tp_dets, fp_dets, fn_dets)
    img_low_conf_dets = show_low_conf_detections(img, given_model_dets, low_conf_ids)       
    img_overlap_dets = show_detections_with_high_overlap(img, given_model_dets, overlap_det_ids)

    img_processed_up = np.hstack([img_with_dets, img_false_dets])
    img_processed_down = np.hstack([img_low_conf_dets, img_overlap_dets])
    img_processed = np.vstack([img_processed_up, img_processed_down])

    img_processed = draw_text_center_top(img_processed, 'Automated Evaluation', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, (250, 100, 150))
    img_processed = draw_text_center_bottom(img_processed, f'ACC : {str(round(acc * 100, 2))}%    MIOU : {str(round(miou * 100, 2))}%    Detection Status : {eval_status}', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, (150, 180, 250))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_processed)



def evaluate_on_image_folder(path, save_path):

    files = os.listdir(path)
    
    print('Folder length :',len(files))

    for file in tqdm(files):

        evaluate_on_image(os.path.join(path, file), os.path.join(save_path, file))

    print('Completed!')


        


evaluate_on_video_or_stream(path = 'demo_videos\ipcam1.mp4', is_stream = False, save_path = None)