from init import *
from eval_utils import *



def evaluate_predictions(name, pred_results_json, eval_fps = cfg['eval_fps'],
                    nms_conf_thresh = cfg['nms_conf_thresh'],
                    aux_conf_thresh = cfg['aux_conf_thresh'],
                    nms_iou_thresh = cfg['nms_iou_thresh'],
                    eval_iou_thresh = cfg['eval_iou_thresh'],
                    eval_acc_thresh = cfg['eval_acc_thresh'],
                    eval_low_conf_thresh = cfg['eval_low_conf_thresh'],
                    eval_overlap_iou_thresh = cfg['eval_overlap_iou_thresh'],
                    class_names_list = [],
                    ignore_classes_from_aux_det = [],
                    save_outputs = True,
                    is_video = True):
    
    if save_outputs and is_video:

        save_dir = os.path.join('process_out', 'processed_video')
        os.makedirs(save_dir, exist_ok=True)
        save_path_eval = os.path.join('process_out', 'processed_video', name+'_eval.webm')
        save_path_aux = os.path.join('process_out', 'processed_video', name+'_aux.webm')
        
        writer_eval = None
        writer_aux = None 
    
    imagewise_stats = {}    
    imagewise_stats['idx'] = []
    if is_video:
        imagewise_stats['frame_no'] = []
        imagewise_stats['timestamp'] = []
    imagewise_stats['raw_save_path' ] = []
    imagewise_stats['ndets'] = []
    imagewise_stats['ACC'] = []
    imagewise_stats['PREC'] = []
    imagewise_stats['RECL'] = []
    imagewise_stats['TP'] = []
    imagewise_stats['FP'] = []
    imagewise_stats['FN'] = []
    imagewise_stats['nLowConf'] = []
    imagewise_stats['pLowConf'] = []
    imagewise_stats['nOverLap'] = []

    if save_outputs:
    
        imagewise_stats['dets_save_path'] = []
        imagewise_stats['aux_dets_save_path'] = []
        imagewise_stats['tp_fp_fn_save_path'] = []
        imagewise_stats['low_conf_save_path'] = []
        imagewise_stats['overlap_save_path'] = []
        
        dets_save_dir = os.path.join('process_out', 'processed_images', name+'_dets')
        aux_dets_save_dir = os.path.join('process_out', 'processed_images', name+'_aux_dets')
        tp_fp_fn_save_dir = os.path.join('process_out', 'processed_images', name+'_tp_fp_fn')
        low_conf_save_dir = os.path.join('process_out', 'processed_images', name+'_low_conf')
        overlap_save_dir = os.path.join('process_out', 'processed_images', name+'_overlap')

        os.makedirs(dets_save_dir, exist_ok=True)
        os.makedirs(aux_dets_save_dir, exist_ok=True)
        os.makedirs(tp_fp_fn_save_dir, exist_ok=True)
        os.makedirs(low_conf_save_dir, exist_ok=True)
        os.makedirs(overlap_save_dir, exist_ok=True)


    imagewise_classdist = {}
    imagewise_classdist['idx'] = []
    if is_video:
        imagewise_classdist['frame_no'] = []
        imagewise_classdist['timestamp'] = []
    imagewise_classdist['alldets'] = []
    imagewise_classdist['TP'] = []
    imagewise_classdist['FP'] = []
    imagewise_classdist['FN'] = []
    imagewise_classdist['LowConf'] = []
    imagewise_classdist['OverLap'] = []

    eval_status = 'ok'

    for idx in tqdm(pred_results_json['idx']):

        if is_video:
            frame_no = pred_results_json['frame_no'][idx]
            timestamp = pred_results_json['timestamp'][idx]

        raw_save_path = pred_results_json['raw_save_path'][idx]
        given_model_pred_results = pred_results_json['given_model_pred_results'][idx]
        aux_model_pred_results = pred_results_json['aux_model_pred_results'][idx]

        image_file_name = os.path.split(raw_save_path)[1]

        given_model_dets = get_final_detections_post_nms(given_model_pred_results, nms_conf_thresh, nms_iou_thresh, class_names_list, target_classes=None)
        aux_model_dets = get_final_detections_post_nms(aux_model_pred_results, aux_conf_thresh, nms_iou_thresh, class_names_list, target_classes=None, ignore_classes=ignore_classes_from_aux_det)


        tp_dets_ids, fp_dets_ids, fn_dets_ids, tp_dets_classes, fp_dets_classes, fn_dets_classes, miou = get_false_detections(given_model_dets, aux_model_dets)
        low_conf_ids , low_conf_classes, all_conf_classes = get_low_conf_detections(given_model_dets)
        overlap_det_ids, overlap_det_classes = get_detections_with_high_overlap(given_model_dets)

        tp = len(tp_dets_ids)
        fp = len(fp_dets_ids)
        fn = len(fn_dets_ids)

        n_dets = len(given_model_dets)
        n_low_conf = len(low_conf_ids)

        p_low_conf = round(((n_low_conf+(1e-10))*100) / (n_dets+(1e-10)), 2)

        n_overlaps = len(overlap_det_ids)

        acc = round((tp+(1e-10)) / (tp+fp+fn+(1e-10)), 4)
        prec = round((tp+(1e-10)) / (tp+fp+(1e-10)), 4)
        recl = round((tp+(1e-10)) / (tp+fn+(1e-10)), 4)
        
        imagewise_stats['idx'].append(idx)
        if is_video:
            imagewise_stats['frame_no'].append(frame_no)
            imagewise_stats['timestamp'].append(timestamp)
        imagewise_stats['raw_save_path'].append(raw_save_path)
        imagewise_stats['ndets'].append(n_dets)
        imagewise_stats['ACC'].append(acc*100)
        imagewise_stats['PREC'].append(prec*100)
        imagewise_stats['RECL'].append(recl*100)
        imagewise_stats['TP'].append(tp)
        imagewise_stats['FP'].append(fp)
        imagewise_stats['FN'].append(fn)
        imagewise_stats['nLowConf'].append(n_low_conf)
        imagewise_stats['pLowConf'].append(p_low_conf)
        imagewise_stats['nOverLap'].append(n_overlaps)


        imagewise_classdist['idx'].append(idx)
        if is_video:
            imagewise_classdist['frame_no'].append(frame_no)
            imagewise_classdist['timestamp'].append(timestamp)
        imagewise_classdist['alldets'].append(all_conf_classes)
        imagewise_classdist['TP'].append(tp_dets_classes)
        imagewise_classdist['FP'].append(fp_dets_classes)
        imagewise_classdist['FN'].append(fn_dets_classes)
        imagewise_classdist['LowConf'].append(low_conf_classes)
        imagewise_classdist['OverLap'].append(overlap_det_classes)

        if save_outputs:

            eval_status = 'ok' if acc > eval_acc_thresh else 'poor'
            bg_color = (250, 180, 150) if eval_status == 'ok' else (150, 180, 250) 

            image = cv2.imread(raw_save_path)
        
            
            image_with_dets = show_only_detections(image, given_model_dets, class_names_list)
            image_with_aux_dets = show_only_detections(image, aux_model_dets, class_names_list, aux=True)
            image_false_dets = show_false_detections(image, given_model_dets, aux_model_dets, tp_dets_ids, fp_dets_ids, fn_dets_ids)
            image_low_conf_dets = show_low_conf_detections(image, given_model_dets, low_conf_ids)       
            image_overlap_dets = show_detections_with_high_overlap(image, given_model_dets, overlap_det_ids)

            dets_save_path = os.path.join(dets_save_dir, image_file_name)
            aux_dets_save_path = os.path.join(aux_dets_save_dir, image_file_name)
            tp_fp_fn_save_path = os.path.join(tp_fp_fn_save_dir, image_file_name)
            low_conf_save_path = os.path.join(low_conf_save_dir, image_file_name)
            overlap_save_path = os.path.join(overlap_save_dir, image_file_name)

            cv2.imwrite(dets_save_path, image_with_dets)
            cv2.imwrite(aux_dets_save_path, image_with_aux_dets)
            cv2.imwrite(tp_fp_fn_save_path, image_false_dets)
            cv2.imwrite(low_conf_save_path, image_low_conf_dets)
            cv2.imwrite(overlap_save_path, image_overlap_dets)

            imagewise_stats['dets_save_path'].append(dets_save_path)
            imagewise_stats['aux_dets_save_path'].append(aux_dets_save_path)
            imagewise_stats['tp_fp_fn_save_path'].append(tp_fp_fn_save_path)
            imagewise_stats['low_conf_save_path'].append(low_conf_save_path)
            imagewise_stats['overlap_save_path'].append(overlap_save_path)

            
            if is_video:

                image_processed_up = np.hstack([image_with_dets, image_false_dets])
                image_processed_down = np.hstack([image_low_conf_dets, image_overlap_dets])
                image_processed = np.vstack([image_processed_up, image_processed_down])

                # frame_processed = draw_text_center_top(frame_processed, 'Automated Evaluation Display', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, (250, 100, 150))
                image_processed = draw_text_center_bottom(image_processed, f'Frame No : {frame_no}    Timestamp : {timestamp}     ACC : {str(round(acc * 100, 2))}%    MIOU : {str(round(miou * 100, 2))}%    PREC : {str(round(prec * 100, 2))}%    RECL : {str(round(recl * 100, 2))}%', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, bg_color)
                image_processed = draw_text_center_bottom(image_processed, f'TP : {tp}    FP : {fp}    FN : {fn}    No. of Low Conf. : {str(n_low_conf)} ({str(p_low_conf)}%)    No. of Overlaps : {n_overlaps}    Detection Status : {eval_status}', (0, 30),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 3, bg_color)

                if writer_eval is None and writer_aux is None:   
                    fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
                    os.makedirs(os.path.dirname(save_path_eval), exist_ok=True)
                    os.makedirs(os.path.dirname(save_path_aux), exist_ok=True)
                    writer_eval = cv2.VideoWriter(save_path_eval, fourcc, eval_fps, (image_processed.shape[1], image_processed.shape[0]), True)
                    writer_aux = cv2.VideoWriter(save_path_aux, fourcc, eval_fps, (image_with_aux_dets.shape[1], image_with_aux_dets.shape[0]), True)
                    
                if writer_eval is not None and writer_aux is not None:
                    writer_eval.write(image_processed)
                    writer_aux.write(image_with_aux_dets)


    if save_outputs and is_video:
        if writer_eval is not None and writer_aux is not None:
            writer_eval.release()
            writer_aux.release()

    
    return imagewise_stats, imagewise_classdist





def evaluate_multi_predictions(multi_pred_results_json, 
                                nms_conf_thresh = cfg['nms_conf_thresh'],
                                aux_conf_thresh = cfg['aux_conf_thresh'],
                                nms_iou_thresh = cfg['nms_iou_thresh'],
                                eval_iou_thresh = cfg['eval_iou_thresh'],
                                eval_acc_thresh = cfg['eval_acc_thresh'],
                                eval_low_conf_thresh = cfg['eval_low_conf_thresh'],
                                eval_overlap_iou_thresh = cfg['eval_overlap_iou_thresh'],
                                class_names_list = [],
                                ignore_classes_from_aux_det = [],
                                save_outputs=True):
    
    
    multi_stats_json = {}
    multi_stats_json['iter_no'] = []
    multi_stats_json['name'] = []
    multi_stats_json['imagewise_stats'] = [] 
    multi_stats_json['imagewise_classdist'] = []
    multi_stats_json['is_video'] = []


    print('\n\nEvaluating Detections...')

    
    for it in multi_pred_results_json['iter_no']:

        name = multi_pred_results_json['name'][it]
        pred_results_json = multi_pred_results_json['pred_results_json'][it]
        eval_fps = multi_pred_results_json['eval_fps'][it]
        is_video = multi_pred_results_json['is_video'][it]

        print(f'\nIteration : {it}, Name : {name}, is_video : {is_video}')

        imagewise_stats, imagewise_classdist = evaluate_predictions(name, pred_results_json, eval_fps,
                                                            nms_conf_thresh = nms_conf_thresh,
                                                            aux_conf_thresh = aux_conf_thresh,
                                                            nms_iou_thresh = nms_iou_thresh,
                                                            eval_iou_thresh = eval_iou_thresh,
                                                            eval_acc_thresh = eval_acc_thresh,
                                                            eval_low_conf_thresh = eval_low_conf_thresh,
                                                            eval_overlap_iou_thresh = eval_overlap_iou_thresh,
                                                            class_names_list = class_names_list,
                                                            ignore_classes_from_aux_det = ignore_classes_from_aux_det,
                                                            save_outputs=save_outputs, 
                                                            is_video=is_video)
        
        
        multi_stats_json['iter_no'].append(it)
        multi_stats_json['name'].append(name)
        multi_stats_json['imagewise_stats'].append(imagewise_stats)
        multi_stats_json['imagewise_classdist'].append(imagewise_classdist)
        multi_stats_json['is_video'].append(is_video)

    it = multi_pred_results_json['iter_no'][-1]+1
    print(f'\nIteration : {it}, Name : overall, is_video : False')

    if save_outputs:

        overall_imagewise_stats_keys = ['raw_save_path', 'ndets', 'ACC', 'PREC', 'RECL', 'TP', 'FP', 'FN', 
                                        'nLowConf', 'pLowConf', 'nOverLap', 'dets_save_path', 'aux_dets_save_path', 
                                        'tp_fp_fn_save_path', 'low_conf_save_path', 'overlap_save_path']

    else:

        overall_imagewise_stats_keys = ['raw_save_path', 'ndets', 'ACC', 'PREC', 'RECL', 'TP', 'FP', 'FN', 
                                        'nLowConf', 'pLowConf', 'nOverLap']


    overall_imagewise_classdist_keys = ['alldets', 'TP', 'FP', 'FN', 'LowConf', 'OverLap']

    overall_imagewise_stats = {}
    overall_imagewise_classdist = {}

    overall_imagewise_stats['idx'] = [i for i in range(len(np.concatenate([i['idx'] for i in multi_stats_json['imagewise_stats']])))]
    for key in overall_imagewise_stats_keys:
        overall_imagewise_stats[key] =[item for sublist in [i[key] for i in multi_stats_json['imagewise_stats']]  for item in sublist]

    overall_imagewise_classdist['idx'] = [i for i in range(len(np.concatenate([i['idx'] for i in multi_stats_json['imagewise_classdist']])))]
    for key in overall_imagewise_classdist_keys:
        overall_imagewise_classdist[key] = [item for sublist in [i[key] for i in multi_stats_json['imagewise_classdist']]  for item in sublist]


    multi_stats_json['iter_no'].append(it)
    multi_stats_json['name'].append('overall')
    multi_stats_json['imagewise_stats'].append(overall_imagewise_stats)
    multi_stats_json['imagewise_classdist'].append(overall_imagewise_classdist)
    multi_stats_json['is_video'].append(False)
    
    return multi_stats_json