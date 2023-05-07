from init import *
from eval_utils import *
from detect_pipelines import *
from evaluate_pipelines import *
from eval_stats_report import *
from eval_vis import *
from summary_selection_pipelines import *


import argparse

parser = argparse.ArgumentParser(
                    prog='AutoEval',
                    description='Automated Evaluation Project')

parser.add_argument('--config', required=True) 
args = parser.parse_args()


args_cfg = json.load(open(args.config, 'r'))

print(args_cfg)

given_model = get_model_from_path(args_cfg['main_model_path'], device)
aux_model = get_model_from_path(args_cfg['aux_model_path'], device)

class_names = args_cfg['classnames']

eval_fps = args_cfg['eval_options']['eval_fps'] if args_cfg['eval_options']['eval_fps'] != None else default_cfg['eval_fps']
nms_conf_thresh = args_cfg['eval_options']['nms_conf_thresh'] if args_cfg['eval_options']['nms_conf_thresh'] != None else default_cfg['nms_conf_thresh']
aux_conf_thresh = args_cfg['eval_options']['aux_conf_thresh'] if args_cfg['eval_options']['aux_conf_thresh'] != None else default_cfg['aux_conf_thresh']
nms_iou_thresh = args_cfg['eval_options']['nms_iou_thresh'] if args_cfg['eval_options']['nms_iou_thresh'] != None else default_cfg['nms_iou_thresh']
eval_iou_thresh = args_cfg['eval_options']['eval_iou_thresh'] if args_cfg['eval_options']['eval_iou_thresh'] != None else default_cfg['eval_iou_thresh']
eval_acc_thresh = args_cfg['eval_options']['eval_acc_thresh'] if args_cfg['eval_options']['eval_acc_thresh'] != None else default_cfg['eval_acc_thresh']
eval_low_conf_thresh = args_cfg['eval_options']['eval_low_conf_thresh'] if args_cfg['eval_options']['eval_low_conf_thresh'] != None else default_cfg['eval_low_conf_thresh']
eval_overlap_iou_thresh = args_cfg['eval_options']['eval_overlap_iou_thresh'] if args_cfg['eval_options']['eval_overlap_iou_thresh'] != None else default_cfg['eval_overlap_iou_thresh']
class_names_list = class_names
ignore_classes_from_aux_det = args_cfg['eval_options']['ignore_classes_from_aux_det']

acc_filter_thresh=args_cfg['summary_filters']['acc_filter_thresh']
prec_filter_thresh=args_cfg['summary_filters']['prec_filter_thresh'] 
recl_filter_thresh=args_cfg['summary_filters']['recl_filter_thresh']
low_conf_filter_thresh=args_cfg['summary_filters']['low_conf_filter_thresh'] 
n_overlap_thresh=args_cfg['summary_filters']['n_overlap_thresh']
n_select=args_cfg['summary_filters']['n_select']

if args_cfg['eval_on'] in ['video', 'image_folder']:

    if args_cfg['eval_on'] == 'video':
        name, pred_results_json, eval_fps, is_video = detect_on_video(vid_path = args_cfg['data_path'], 
                                                                eval_fps = eval_fps,
                                                                given_model = given_model, 
                                                                aux_model = aux_model)
    else:
        name, pred_results_json, eval_fps, is_video = detect_on_image_folder(folder_path = args_cfg['data_path'],
                                                                given_model = given_model, 
                                                                aux_model = aux_model)


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
                                                            save_outputs=True, 
                                                            is_video=is_video)

    report = generate_report(imagewise_stats, imagewise_classdist, tp_threshold = 0, alldets_threshold = 1, overlap_count_threshold = 1, print_report = True)

    generate_charts_from_report(name, report, imagewise_stats, is_video=is_video, show=True)

    summary_images = get_summary_images(imagewise_stats, 
                                        acc_filter_thresh=acc_filter_thresh, 
                                        prec_filter_thresh=prec_filter_thresh, 
                                        recl_filter_thresh=recl_filter_thresh, 
                                        low_conf_filter_thresh=low_conf_filter_thresh, 
                                        n_overlap_thresh=n_overlap_thresh, 
                                        n_select=n_select)
    
else :

    multi_pred_results_json = detect_on_multi_videos_or_folders(folder_path = args_cfg['data_path'], eval_fps = eval_fps, given_model = given_model, aux_model = aux_model)

    multi_stats_json = evaluate_multi_predictions(multi_pred_results_json, 
                                    nms_conf_thresh = nms_conf_thresh,
                                    aux_conf_thresh = aux_conf_thresh,
                                    nms_iou_thresh = nms_iou_thresh,
                                    eval_iou_thresh = eval_iou_thresh,
                                    eval_acc_thresh = eval_acc_thresh,
                                    eval_low_conf_thresh = eval_low_conf_thresh,
                                    eval_overlap_iou_thresh = eval_overlap_iou_thresh,
                                    class_names_list = class_names_list,
                                    ignore_classes_from_aux_det = ignore_classes_from_aux_det,
                                    save_outputs=True)

    multi_report_json = multi_generate_report(multi_stats_json, print_report = True, show_vis = True, show_only_final = True)

    summary_images = get_overall_summary_from_all_images_and_videos(multi_stats_json, 
                                                       acc_filter_thresh=acc_filter_thresh, 
                                                       prec_filter_thresh=prec_filter_thresh, 
                                                       recl_filter_thresh=recl_filter_thresh, 
                                                       low_conf_filter_thresh=low_conf_filter_thresh, 
                                                       n_overlap_thresh=n_overlap_thresh, 
                                                       n_select=n_select)