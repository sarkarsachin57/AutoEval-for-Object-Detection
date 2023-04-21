from init import *


def get_false_detections(given_model_dets, aux_model_dets, eval_iou_thresh = cfg['eval_iou_thresh']):
    # format_input - given_model_dets : [[x1, y1, x2, y2, conf, class]] 
    # format_input - aux_model_dets : [[x1, y1, x2, y2, conf, class]] 

    all_ious = []

    tp_dets_ids, fp_dets_ids, fn_dets_ids = [], [], []
    tp_dets_classes, fp_dets_classes, fn_dets_classes = [], [], []

    for i,given_det in enumerate(given_model_dets):
        if len(aux_model_dets) != 0:
            ious = [get_iou(given_det[:4],aux_det[:4]) for aux_det in aux_model_dets]
            j = np.argmax(ious)
            if i == np.argmax([get_iou(given_det[:4],aux_model_dets[j][:4]) for given_det in given_model_dets]) and given_model_dets[i][-1] == aux_model_dets[j][-1] and get_iou(given_model_dets[i][:4],aux_model_dets[j][:4]) > eval_iou_thresh:
                all_ious.append(float(max(ious)))
                tp_dets_ids.append(i)
                tp_dets_classes.append(given_det[-1])
            else:
                all_ious.append(0)
                fp_dets_ids.append(i)
                fp_dets_classes.append(given_det[-1])
        else:
            all_ious.append(0)
            fp_dets_ids.append(i)
            fp_dets_classes.append(given_det[-1])


    for i,aux_det in enumerate(aux_model_dets):
        if len(given_model_dets) != 0:
            ious = [get_iou(aux_det[:4],given_det[:4]) for given_det in given_model_dets]
            j = np.argmax(ious)
            if i == np.argmax([get_iou(aux_det[:4],given_model_dets[j][:4]) for aux_det in aux_model_dets]) and aux_model_dets[i][-1] == given_model_dets[j][-1] and get_iou(aux_model_dets[i][:4],given_model_dets[j][:4]) > eval_iou_thresh:
                pass
            else:
                all_ious.append(0)
                fn_dets_ids.append(i)
                fn_dets_classes.append(aux_det[-1])
        else:
            all_ious.append(0)
            fn_dets_ids.append(i)
            fn_dets_classes.append(aux_det[-1])

    miou = round(np.mean(all_ious), 4)

    return tp_dets_ids, fp_dets_ids, fn_dets_ids, tp_dets_classes, fp_dets_classes, fn_dets_classes, miou

    

def get_low_conf_detections(given_model_dets, eval_low_conf_thresh = cfg['eval_low_conf_thresh']):
    # format_input - given_model_dets : [[x1, y1, x2, y2, conf, class]] 
    
    low_conf_ids = []
    low_conf_classes = []
    all_conf_classes = []
    for idx, det in enumerate(given_model_dets):
        x1, y1, x2, y2, conf, class_name = det
        if conf < eval_low_conf_thresh:
            low_conf_ids.append(idx)
            low_conf_classes.append(class_name)
        all_conf_classes.append(class_name)

    return low_conf_ids , low_conf_classes, all_conf_classes


def get_detections_with_high_overlap(given_model_dets, eval_overlap_iou_thresh = cfg['eval_overlap_iou_thresh']):
    # format_input - given_model_dets : [[x1, y1, x2, y2, conf, class]] 

    overlap_det_ids = []
    overlap_det_classes = []
    for (i,j) in combinations(range(len(given_model_dets)), 2):
        idet, jdet = given_model_dets[i], given_model_dets[j]
        if get_iou(idet[:4],jdet[:4]) > eval_overlap_iou_thresh:
            overlap_det_ids.append([i,j])
            overlap_det_classes.append(sorted([idet[-1], jdet[-1]]))

    return overlap_det_ids, overlap_det_classes





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






def frame_no_to_timestamp_str(frame_no, fps):

    time_stamp = frame_no // fps

    time_stamp_str = "%.2d"%(time_stamp // 3600)+":%.2d"%((time_stamp // 60) % 60)+":%.2d"%(time_stamp % 60)

    return time_stamp_str


def get_estd_processing_time(current_frame, total_frame_count, current_fps):

    time_stamp = (total_frame_count - current_frame) // current_fps

    time_stamp_str = "%.2d"%(time_stamp // 3600)+":%.2d"%((time_stamp // 60) % 60)+":%.2d"%(time_stamp % 60)

    return time_stamp_str




def detect_on_video(vid_name, vid_path, eval_fps = eval_fps):

    video = cv2.VideoCapture(vid_path)

    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    current_fps = video_fps

    eval_frame = int(max(1, video_fps // eval_fps))

    print('Video_FPS :', video_fps)
    print('eval_frame :', eval_frame)

    raw_frames_save_dir = os.path.join('video_process_out', 'all_raw_frames', vid_name)

    os.makedirs(raw_frames_save_dir, exist_ok=True)

    pred_results_json = {}
    
    pred_results_json['frame_idx'] = []
    pred_results_json['frame_no'] = []
    pred_results_json['timestamp'] = []
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

            cv2.imwrite(os.path.join(raw_frames_save_dir, str(frame_no)+'.jpg'), frame)
            
            pred_results_json['frame_idx'].append(frame_idx)
            pred_results_json['frame_no'].append(frame_no)
            pred_results_json['timestamp'].append(frame_no_to_timestamp_str(frame_no, video_fps))
            pred_results_json['given_model_pred_results'].append(given_model_pred_results)
            pred_results_json['aux_model_pred_results'].append(aux_model_pred_results)

            frame_idx += 1

        
        fps.update()
        fps.stop()

        current_fps = round(fps.fps(), 2)

        eta = get_estd_processing_time(current_frame = frame_no, total_frame_count = video_frame_count, current_fps = current_fps)


    return pred_results_json



def evaluate_predictions(vid_name, pred_results_json, eval_fps,
                    nms_conf_thresh = cfg['nms_conf_thresh'],
                    aux_conf_thresh = cfg['aux_conf_thresh'],
                    nms_iou_thresh = cfg['nms_iou_thresh'],
                    eval_iou_thresh = cfg['eval_iou_thresh'],
                    eval_acc_thresh = cfg['eval_acc_thresh'],
                    eval_low_conf_thresh = cfg['eval_low_conf_thresh'],
                    eval_overlap_iou_thresh = cfg['eval_overlap_iou_thresh']):
    

    save_dir = os.path.join('video_process_out', 'processed_video')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join('video_process_out', 'processed_video', vid_name+'.webm')
    
    writer = None
    
    framewise_stats = {}    
    framewise_stats['frame_no'] = []
    framewise_stats['timestamp'] = []
    framewise_stats['ndets'] = []
    framewise_stats['ACC'] = []
    framewise_stats['PREC'] = []
    framewise_stats['RECL'] = []
    framewise_stats['TP'] = []
    framewise_stats['FP'] = []
    framewise_stats['FN'] = []
    framewise_stats['nLowConf'] = []
    framewise_stats['pLowConf'] = []
    framewise_stats['nOverLap'] = []

    framewise_classdist = {}
    framewise_classdist['frame_no'] = []
    framewise_classdist['timestamp'] = []
    framewise_classdist['alldets'] = []
    framewise_classdist['TP'] = []
    framewise_classdist['FP'] = []
    framewise_classdist['FN'] = []
    framewise_classdist['LowConf'] = []
    framewise_classdist['OverLap'] = []

    eval_status = 'ok'

    for frame_idx in tqdm(pred_results_json['frame_idx']):

        frame_no = pred_results_json['frame_no'][frame_idx]
        timestamp = pred_results_json['timestamp'][frame_idx]
        given_model_pred_results = pred_results_json['given_model_pred_results'][frame_idx]
        aux_model_pred_results = pred_results_json['aux_model_pred_results'][frame_idx]

        given_model_dets = get_final_detections_post_nms(given_model_pred_results, nms_conf_thresh, nms_iou_thresh, target_classes=None)
        aux_model_dets = get_final_detections_post_nms(aux_model_pred_results, aux_conf_thresh, nms_iou_thresh, target_classes=None)


        tp_dets_ids, fp_dets_ids, fn_dets_ids, tp_dets_classes, fp_dets_classes, fn_dets_classes, miou = get_false_detections(given_model_dets, aux_model_dets)
        low_conf_ids , low_conf_classes, all_conf_classes = get_low_conf_detections(given_model_dets)
        overlap_det_ids, overlap_det_classes = get_detections_with_high_overlap(given_model_dets)

        tp = len(tp_dets_ids)
        fp = len(fp_dets_ids)
        fn = len(fn_dets_ids)

        n_dets = len(given_model_dets)
        n_low_conf = len(low_conf_ids)

        p_low_conf = round((n_low_conf*100) / n_dets, 2)

        n_overlaps = len(overlap_det_ids)

        acc = round(tp / (tp+fp+fn), 4)
        prec = round(tp / (tp+fp), 4)
        recl = round(tp / (tp+fn), 4)
        
        framewise_stats['frame_no'].append(frame_no)
        framewise_stats['timestamp'].append(timestamp)
        framewise_stats['ndets'].append(n_dets)
        framewise_stats['ACC'].append(acc*100)
        framewise_stats['PREC'].append(prec*100)
        framewise_stats['RECL'].append(recl*100)
        framewise_stats['TP'].append(tp)
        framewise_stats['FP'].append(fp)
        framewise_stats['FN'].append(fn)
        framewise_stats['nLowConf'].append(n_low_conf)
        framewise_stats['pLowConf'].append(p_low_conf)
        framewise_stats['nOverLap'].append(n_overlaps)

        framewise_classdist['frame_no'].append(frame_no)
        framewise_classdist['timestamp'].append(timestamp)
        framewise_classdist['alldets'].append(all_conf_classes)
        framewise_classdist['TP'].append(tp_dets_classes)
        framewise_classdist['FP'].append(fp_dets_classes)
        framewise_classdist['FN'].append(fn_dets_classes)
        framewise_classdist['LowConf'].append(low_conf_classes)
        framewise_classdist['OverLap'].append(overlap_det_classes)

        eval_status = 'ok' if acc > eval_acc_thresh else 'poor'
        bg_color = (250, 180, 150) if eval_status == 'ok' else (150, 180, 250) 

        frame = cv2.imread(os.path.join('video_process_out', 'all_raw_frames', vid_name, str(frame_no) + '.jpg'))
        
        frame_with_dets = show_only_detections(frame, given_model_dets)
        frame_false_dets = show_false_detections(frame, given_model_dets, aux_model_dets, tp_dets_ids, fp_dets_ids, fn_dets_ids)
        frame_low_conf_dets = show_low_conf_detections(frame, given_model_dets, low_conf_ids)       
        frame_overlap_dets = show_detections_with_high_overlap(frame, given_model_dets, overlap_det_ids)

        frame_processed_up = np.hstack([frame_with_dets, frame_false_dets])
        frame_processed_down = np.hstack([frame_low_conf_dets, frame_overlap_dets])
        frame_processed = np.vstack([frame_processed_up, frame_processed_down])

        frame_processed = draw_text_center_top(frame_processed, 'Automated Evaluation', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, (250, 100, 150))
        frame_processed = draw_text_center_bottom(frame_processed, f'Frame No : {frame_no}    Timestamp : {timestamp}     ACC : {str(round(acc * 100, 2))}%    MIOU : {str(round(miou * 100, 2))}%    PREC : {str(round(prec * 100, 2))}%    RECL : {str(round(recl * 100, 2))}%', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, bg_color)
        frame_processed = draw_text_center_bottom(frame_processed, f'TP : {tp}    FP : {fp}    FN : {fn}    No. of Low Conf. : {str(n_low_conf)} ({str(p_low_conf)}%)    No. of Overlaps : {n_overlaps}    Detection Status : {eval_status}', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, bg_color)

        
        if writer is None:   
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
            writer = cv2.VideoWriter(save_path, fourcc, eval_fps, (frame_processed.shape[1], frame_processed.shape[0]), True)
            
        if writer is not None:
            writer.write(frame_processed)


    if writer is not None:
        writer.release()

    
    return framewise_stats, framewise_classdist



def describe(df, stats):
    d = df.describe()
    return pd.concat([d,df.reindex(d.columns, axis = 1).agg(stats)])

def get_class_based_stats(class_specific_stats, tp_threshold):

    alldets_counts = np.unique(np.concatenate(class_specific_stats['alldets']), return_counts=True)
    tp_counts = np.unique(np.concatenate(class_specific_stats['TP']), return_counts=True)
    fp_counts = np.unique(np.concatenate(class_specific_stats['FP']), return_counts=True)
    fn_counts = np.unique(np.concatenate(class_specific_stats['FN']), return_counts=True)
    lowconf_counts = np.unique(np.concatenate(class_specific_stats['LowConf']), return_counts=True)

    class_based_values = {}
    class_based_values['class_name'] = []
    class_based_values['TP'] = []
    class_based_values['FP'] = []
    class_based_values['FN'] = []


    for class_name in np.unique(np.concatenate([tp_counts[0],fp_counts[0],fn_counts[0]])):

        class_based_values['class_name'].append(class_name)

        try:
            class_based_values['TP'].append(tp_counts[1][list(tp_counts[0]).index(class_name)])
        except:
            class_based_values['TP'].append(0)

        try:
            class_based_values['FP'].append(fp_counts[1][list(fp_counts[0]).index(class_name)])
        except:
            class_based_values['FP'].append(0)    
        
        try:
            class_based_values['FN'].append(fn_counts[1][list(fn_counts[0]).index(class_name)])
        except:
            class_based_values['FN'].append(0)

        
    class_based_values = pd.DataFrame(class_based_values) 
    class_based_values['ACC'] = (class_based_values['TP'] * 100) / (class_based_values['TP'] + class_based_values['FP'] + class_based_values['FN'])
    class_based_values['PREC'] = (class_based_values['TP'] * 100) / (class_based_values['TP'] + class_based_values['FP'])
    class_based_values['RECL'] = (class_based_values['TP'] * 100) / (class_based_values['TP'] + class_based_values['FN'])

    class_based_values = class_based_values[class_based_values['TP'] >= tp_threshold]

    overall_agg = class_based_values.agg({'TP': 'sum', 'FP': 'sum', 'FN': 'sum', 'ACC' : 'mean', 'PREC': 'mean', 'RECL': 'mean'})
    class_based_values = pd.concat([class_based_values, pd.DataFrame(pd.concat([pd.Series({'class_name':'Overall'}),overall_agg])).T]).round(2).sort_values(by='ACC', ascending=False).reset_index().drop(['index'], axis=1).to_dict()

    return class_based_values



def get_low_conf_stats(class_specific_stats, alldets_threshold):

    alldets_counts = np.unique(np.concatenate(class_specific_stats['alldets']), return_counts=True)
    lowconf_counts = np.unique(np.concatenate(class_specific_stats['LowConf']), return_counts=True)

    class_based_values = {}
    class_based_values['class_name'] = []
    class_based_values['alldets'] = [] 
    class_based_values['lowconf'] = []


    for class_name in np.unique(alldets_counts[0]):

        class_based_values['class_name'].append(class_name)

        try:
            class_based_values['alldets'].append(alldets_counts[1][list(alldets_counts[0]).index(class_name)])
        except:
            class_based_values['alldets'].append(0)

        try:
            class_based_values['lowconf'].append(lowconf_counts[1][list(lowconf_counts[0]).index(class_name)])
        except:
            class_based_values['lowconf'].append(0)
            
        
    class_based_values = pd.DataFrame(class_based_values) 
    class_based_values['LowConfPercent'] = (class_based_values['lowconf'] * 100) / class_based_values['alldets'] 

    class_based_values = class_based_values[class_based_values['alldets'] >= alldets_threshold]

    overall_agg = class_based_values.agg({'alldets': 'sum', 'lowconf' : 'sum', 'LowConfPercent': 'mean'})
    class_based_values = pd.concat([class_based_values, pd.DataFrame(pd.concat([pd.Series({'class_name':'Overall'}),overall_agg])).T]).round(2).sort_values(by='alldets', ascending=False).reset_index().drop(['index'], axis=1).to_dict()

    return class_based_values




def get_overlap_stats(framewise_classdist, count_threshold):

    overlap_pair = []
    for pairs in framewise_classdist['OverLap']:
        if len(pairs) != 0:
            for pair in pairs:
                overlap_pair.append(pair)

    alldets_uni_counts = np.unique(np.concatenate(framewise_classdist['alldets']), return_counts=True)
    all_overlap_combinations = [sorted(x) for x in combinations(alldets_uni_counts[0], 2)]+[sorted(x,reverse=True) for x in combinations(alldets_uni_counts[0], 2)]

    overlaps_stats_percentage = {}
    overlaps_stats_percentage['class_pairs'] = []
    overlaps_stats_percentage['percentage'] = []

    overlaps_stats_count = {}
    overlaps_stats_count['class_pairs'] = []
    overlaps_stats_count['count'] = []

    for class1, class2 in all_overlap_combinations:
        class1_count = alldets_uni_counts[1][list(alldets_uni_counts[0]).index(class1)]
        class1_class2_overlap_count = len([1 for x in overlap_pair if x == sorted([class1, class2])]) 
        if class1_class2_overlap_count >= count_threshold:
            overlaps_stats_percentage['class_pairs'].append(class1 + ' on ' + class2)
            overlaps_stats_percentage['percentage'].append(round((class1_class2_overlap_count * 100) / class1_count, 2))
            if class2+', '+class1 not in overlaps_stats_count['class_pairs']:
                overlaps_stats_count['class_pairs'].append(class1+', '+class2)
                overlaps_stats_count['count'].append(class1_class2_overlap_count)

    overlaps_stats_percentage = pd.DataFrame(overlaps_stats_percentage).sort_values(by='percentage', ascending=False).reset_index().drop(['index'], axis=1)
    overlaps_stats_percentage = overlaps_stats_percentage.to_dict()

    overlaps_stats_count = pd.DataFrame(overlaps_stats_count).sort_values(by='count', ascending=False).reset_index().drop(['index'], axis=1)
    overlaps_stats_count = overlaps_stats_count.to_dict()

    return overlaps_stats_percentage, overlaps_stats_count


def generate_report(framewise_stats, framewise_classdist, tp_threshold, alldets_threshold, overlap_count_threshold, print_report = False):

    report_json = {}

    report_json['overall'] = {}
        
    report_json['overall']['absolute'] = describe(pd.DataFrame(framewise_stats).loc[:,['ndets', 'TP', 'FP', 'FN', 'nLowConf', 'nOverLap']], ['sum']).iloc[1:].round(2).to_dict()
    report_json['overall']['relative'] = pd.DataFrame(framewise_stats).loc[:,['ACC', 'PREC', 'RECL', 'pLowConf']].describe().iloc[1:].round(2).to_dict()

    report_json['class_based'] = get_class_based_stats(framewise_classdist, tp_threshold)
    report_json['low_conf_stats'] = get_low_conf_stats(framewise_classdist, alldets_threshold)
    report_json['overlaps_stats_percentage'], report_json['overlaps_stats_count'] = get_overlap_stats(framewise_classdist, overlap_count_threshold)

    if print_report:
        print("\nOverall Absolute Results :-\n")
        print(pd.DataFrame(report_json['overall']['absolute']))
        print("\nOverall Relative Results :-\n")
        print(pd.DataFrame(report_json['overall']['relative']))
        print('\nClass Based Results :-\n')
        print(pd.DataFrame(report_json['class_based']))
        print('\nLow Conf results :-\n')
        print(pd.DataFrame(report_json['low_conf_stats']))
        print('\nOverlap detection classwise counts :-\n')
        print(pd.DataFrame(report_json['overlaps_stats_count']))
        print('\nOverlap detection classwise percentages :-\n')
        print(pd.DataFrame(report_json['overlaps_stats_percentage']))

    return report_json




def RollingPositiveAverage(listA, window=3):
     s = pd.Series(listA)
     s[s < 0] = np.nan
     result = s.rolling(window, center=True, min_periods=1).mean()
     result.iloc[:window // 2] = np.nan
     result.iloc[-(window // 2):] = np.nan
     return list(result)  # or result.values or list(result) if you prefer array or list


def generate_charts_from_video_report(name, report, framewise_stats, show = False):


    save_dir = os.path.join('video_process_out', 'report_charts', name)
    os.makedirs(save_dir, exist_ok = True)

    
    class_based_df = pd.DataFrame(report['class_based']).round(2)

    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=class_based_df['class_name'], y=class_based_df['ACC']),
        go.Bar(name='Precision', x=class_based_df['class_name'], y=class_based_df['PREC']),
        go.Bar(name='Recall', x=class_based_df['class_name'], y=class_based_df['RECL']),
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_layout(
        title={'text':"Class Wise Bar Chart Visualization of Accuracy, Precision and Recall",
                            'x':0.5,'y':0.97},
        xaxis_title="classes",
        yaxis_title="Scores (%) -->",
        font=dict(
            family="Georgia",
            size=18
        )
    )

    fig.write_html(os.path.join(save_dir, 'class_wise_acc_prec_recl_bar.html'))
    if show:
        print('\n\n')
        fig.show()
        print('\n\n')

    fig = go.Figure(data=[
        go.Bar(name='TP', x=class_based_df.sort_values(by='TP', ascending=False)['class_name'], y=class_based_df.sort_values(by='TP', ascending=False)['TP']),
        go.Bar(name='FP', x=class_based_df.sort_values(by='TP', ascending=False)['class_name'], y=class_based_df.sort_values(by='TP', ascending=False)['FP']),
        go.Bar(name='FN', x=class_based_df.sort_values(by='TP', ascending=False)['class_name'], y=class_based_df.sort_values(by='TP', ascending=False)['FN']),
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_layout(
        title={'text':"Class Wise Bar Chart Visualization of TP, FP and FN",
                            'x':0.5,'y':0.97},
        xaxis_title="Classes",
        yaxis_title="Number of Occurrence -->",
        font=dict(
            family="Georgia",
            size=18
        )
    )
    
    fig.write_html(os.path.join(save_dir, 'class_wise_tp_fp_fn_bar.html'))
    if show:
        fig.show()
        print('\n\n')

    low_conf_stats = pd.DataFrame(report['low_conf_stats']).round(2)

    fig = go.Figure(data=[
        go.Bar(name='Alldets', x=low_conf_stats.sort_values(by='alldets', ascending=False)['class_name'], y=low_conf_stats.sort_values(by='alldets', ascending=False)['alldets']),
        go.Bar(name='LowConf', x=low_conf_stats.sort_values(by='alldets', ascending=False)['class_name'], y=low_conf_stats.sort_values(by='alldets', ascending=False)['lowconf'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_layout(
        title={'text':"Class Wise Bar Chart Visualization of Number of Low Conf detections with respect to All detections",
                            'x':0.5,'y':0.97},
        xaxis_title="Classes",
        yaxis_title="Number of Occurrence -->",
        font=dict(
            family="Georgia",
            size=18
        )
    )

    fig.write_html(os.path.join(save_dir, 'class_wise_low_conf_count_bar.html'))
    if show:
        fig.show()
        print('\n\n')

    fig = px.bar(x=low_conf_stats.sort_values(by='LowConfPercent', ascending=False)['class_name'], y=low_conf_stats.sort_values(by='LowConfPercent', ascending=False)['LowConfPercent'], color=low_conf_stats.sort_values(by='LowConfPercent', ascending=False)['LowConfPercent'],
                labels={"x":"Classes","y":"LowConf Percentages"})
    fig.update_layout(
        title={'text':"Class Wise Bar Chart Visualization of Percentages of Low Conf detections with respect to All detections",
                            'x':0.5,'y':0.97},
        xaxis_title="Classes",
        yaxis_title="Scores (%) -->",
        font=dict(
            family="Georgia",
            size=18
        )
    )

    fig.write_html(os.path.join(save_dir, 'class_wise_low_conf_percent_bar.html'))
    if show:
        fig.show()
        print('\n\n')


    overlap_stat_count = pd.DataFrame(report['overlaps_stats_count'])
    fig = px.bar(x=overlap_stat_count['class_pairs'], y=overlap_stat_count['count'], color=overlap_stat_count['count'],
                labels={"x":"Classes","y":"Number of Overlaps"})
    fig.update_layout(
        title={'text':"Class Wise Bar Chart Visualization of number of overlap between two classes",
                            'x':0.5,'y':0.97},
        xaxis_title="Class Pairs",
        yaxis_title="Number of Occurrence -->",
        font=dict(
            family="Georgia",
            size=18
        )
    )
    
    fig.write_html(os.path.join(save_dir, 'class_wise_overlap_count_bar.html'))
    if show:
        fig.show()
        print('\n\n')


    overlaps_stats_percentage = pd.DataFrame(report['overlaps_stats_percentage'])
    fig = px.bar(x=overlaps_stats_percentage['class_pairs'], y=overlaps_stats_percentage['percentage'], color=overlaps_stats_percentage['percentage'],
                labels={"x":"Classes","y":"Percentages of Overlaps"})
    fig.update_layout(
        title={'text':"Class Wise Bar Chart Visualization of estimated probablity of overlap of one class with another",
                            'x':0.5,'y':0.97},
        xaxis_title="Class Pairs",
        yaxis_title="Scores (%) -->",
        font=dict(
            family="Georgia",
            size=18
        )
    )
    
    fig.write_html(os.path.join(save_dir, 'class_wise_overlap_percent_bar.html'))
    if show:
        fig.show()
        print('\n\n')

    rolling_window = 5

    fig = go.Figure(data=[
        
        go.Scatter(name=str(int(cfg['eval_acc_thresh']*100))+'%', x=framewise_stats['timestamp'], y=[cfg['eval_acc_thresh']*100 for i in range(len(framewise_stats['ACC']))], line = dict(shape = 'linear', color = 'rgb(100, 100, 100)', dash = 'dot')),
        go.Scatter(name='Accuracy', x=framewise_stats['timestamp'], y=RollingPositiveAverage(framewise_stats['ACC'], window=rolling_window)),
        go.Scatter(name='Precision', x=framewise_stats['timestamp'], y=RollingPositiveAverage(framewise_stats['PREC'], window=rolling_window)),
        go.Scatter(name='Recall', x=framewise_stats['timestamp'], y=RollingPositiveAverage(framewise_stats['RECL'], window=rolling_window))
    ])

    fig.update_layout(
        title={'text':"Time Series Visualzation of Accuracy, Precision and Recall",
                            'x':0.5,'y':0.97},
        xaxis_title="Video Timestamp -->",
        yaxis_title="Scores (%) -->",
        font=dict(
            family="Georgia",
            size=18
        )
    )

    
    fig.write_html(os.path.join(save_dir, 'time_series_acc_prec_recl_line.html'))
    if show:
        fig.show()
        print('\n\n')


    fig = go.Figure(data=[
        go.Scatter(name='TP', x=framewise_stats['timestamp'], y=RollingPositiveAverage(framewise_stats['TP'], window=rolling_window)),
        go.Scatter(name='FP', x=framewise_stats['timestamp'], y=RollingPositiveAverage(framewise_stats['FP'], window=rolling_window)),
        go.Scatter(name='FN', x=framewise_stats['timestamp'], y=RollingPositiveAverage(framewise_stats['FN'], window=rolling_window))
    ])

    fig.update_layout(
        title={'text':"Time Series Visualzation of TP, FP and FN",
                            'x':0.5,'y':0.97},
        xaxis_title="Video Timestamp -->",
        yaxis_title="Number of Occurrence -->",
        font=dict(
            family="Georgia",
            size=18
        )
    )

    
    fig.write_html(os.path.join(save_dir, 'time_series_tp_fp_fn_line.html'))
    if show:
        fig.show()
        print('\n\n')


    fig = go.Figure(data=[
        go.Scatter(name='Alldets', x=framewise_stats['timestamp'], y=RollingPositiveAverage(framewise_stats['ndets'], window=rolling_window)),
        go.Scatter(name='LowConfDets', x=framewise_stats['timestamp'], y=RollingPositiveAverage(framewise_stats['nLowConf'], window=rolling_window)),
    ])

    fig.update_layout(
        title={'text':"Time Series Visualzation of Number of Low Conf detections with respect to All detections",
                            'x':0.5,'y':0.97},
        xaxis_title="Video Timestamp -->",
        yaxis_title="Number of Occurrence -->",
        font=dict(
            family="Georgia",
            size=18
        )
    )

    
    fig.write_html(os.path.join(save_dir, 'time_series_lowconf_count_line.html'))
    if show:
        fig.show()
        print('\n\n')


    fig = go.Figure(data=[
        go.Scatter(name='LowConfDets', x=framewise_stats['timestamp'], y=RollingPositiveAverage(framewise_stats['pLowConf'], window=rolling_window)),
    ])

    fig.update_layout(
        title={'text':"Time Series Visualzation of Percentages of Low Conf detections with respect to All detections",
                            'x':0.5,'y':0.97},
        xaxis_title="Video Timestamp -->",
        yaxis_title="Scores (%) -->",
        font=dict(
            family="Georgia",
            size=18
        )
    )


    fig.write_html(os.path.join(save_dir, 'time_series_lowconf_percent_line.html'))
    if show:
        fig.show()
        print('\n\n')


    fig = go.Figure(data=[
        go.Scatter(name='No. of OverLap', x=framewise_stats['timestamp'], y=RollingPositiveAverage(framewise_stats['nOverLap'], window=rolling_window)),
    ])

    fig.update_layout(
        title={'text':"Time Series Visualzation of Number of Overlaps",
                            'x':0.5,'y':0.97},
        xaxis_title="Video Timestamp -->",
        yaxis_title="Number of Occurrence -->",
        font=dict(
            family="Georgia",
            size=18
        )
    )


    fig.write_html(os.path.join(save_dir, 'time_series_overlap_count_line.html'))
    if show:
        fig.show()
        print('\n\n')
