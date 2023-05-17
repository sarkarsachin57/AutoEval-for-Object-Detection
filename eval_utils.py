
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


def get_entropy(p):
    if p==0:
        p = p+1e-10
    elif p==1:
        p = p-1e-10
    entropy = -((p * np.log2(p)) + ((1-p) * np.log2(1-p)))
    return entropy

def get_margin(probs):
    max1 = np.sort(probs)[-1]
    max2 = np.sort(probs)[-2]
    return max1-max2

def get_uncertainty(probs):
    return np.mean([get_entropy(p) for p in probs])


def get_all_classes_from_detections(given_model_dets):
    
    all_classes = []
    for idx, det in enumerate(given_model_dets):
        x1, y1, x2, y2, conf, class_name = det
        all_classes.append(class_name)

    ndets = len(all_classes)

    return all_classes, ndets




def get_low_conf_detections(given_model_dets, eval_low_conf_thresh = cfg['eval_low_conf_thresh']):
    # format_input - given_model_dets : [[x1, y1, x2, y2, conf, class]] 
    
    all_confs = []
    low_conf_ids = []
    low_conf_classes = []
    for idx, det in enumerate(given_model_dets):
        x1, y1, x2, y2, conf, class_name = det
        if conf < eval_low_conf_thresh:
            low_conf_ids.append(idx)
            low_conf_classes.append(class_name)
        all_confs.append(conf)

    avg_conf = round(100*np.mean(all_confs), 2)

    return low_conf_ids , low_conf_classes, all_confs, avg_conf


def get_uncertainty_detections(given_model_dets, given_model_class_probs, uncertainty_thresh):

    entropies = []
    uncertainty_ids = []
    for idx, (det, class_probs) in enumerate(zip(given_model_dets, given_model_class_probs)):
        entropy = get_uncertainty(class_probs)
        entropies.append(entropy)
        if entropy > uncertainty_thresh:
            uncertainty_ids.append(idx)

    avg_entropy = round(100*np.mean(entropies), 2)

    return uncertainty_ids , entropies, avg_entropy
    
def get_low_margin_detections(given_model_dets, given_model_class_probs, class_name_list, margin_thresh):

    margins = []
    low_margin_ids = []
    margin_class_pairs = []
    for idx, (det, class_probs) in enumerate(zip(given_model_dets, given_model_class_probs)):
        margin = get_margin(class_probs)
        margins.append(margin)
        class1 = class_name_list[np.argsort(class_probs)[-1]]
        class2 = class_name_list[np.argsort(class_probs)[-2]]
        margin_class_pairs.append(class1 + ', ' + class2)
        if margin < margin_thresh:
            low_margin_ids.append(idx)

    avg_margin = round(100*np.mean(margins), 2)

    return low_margin_ids , margin_class_pairs, margins, avg_margin



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

    image = add_box_color_legends_to_image(image, text_list=['True Positive', 'False Positive', 'False Negetive'], color_list=[(100, 255, 100), (100, 100, 255), (255, 100, 100)], pad=180)
    image = draw_text_center_top(image, 'Detections with True Positives, False Positives and Negetives', cv2.FONT_HERSHEY_DUPLEX,   0.85, (0, 0, 0), 2, (255, 200, 200))
    
    image[:,0] = image[:,0] * [0,0,0]
    image[:,-1] = image[:,-1] * [0,0,0]
    image[0,:] = image[0,:] * [0,0,0]
    image[-1,:] = image[-1,:] * [0,0,0]
    
    return image
                    


def show_low_conf_detections(image, given_model_dets, low_conf_ids):

    image = image.copy()

    for i,det in enumerate(given_model_dets):

        if i in low_conf_ids:
            x1, y1, x2, y2, conf, class_name = det
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 100, 255), 2)
            draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (100, 100, 255))
                    
        # else:
        #     x1, y1, x2, y2, conf, class_name = det
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 100), 2)
        #     draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (255, 100, 100))
        
    image = add_box_color_legends_to_image(image, text_list=['Low Confidence Detection'], color_list=[(100, 100, 255)], pad=500)
    image = draw_text_center_top(image, 'Detections with low Confidence', cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2, (255, 200, 200))

    image[:,0] = image[:,0] * [0,0,0]
    image[:,-1] = image[:,-1] * [0,0,0]
    image[0,:] = image[0,:] * [0,0,0]
    image[-1,:] = image[-1,:] * [0,0,0]

    return image





def show_detections_with_high_overlap(image, given_model_dets, overlap_det_ids):

    image = image.copy()

    overlap_det_ids = np.unique(np.ravel(overlap_det_ids))

    for i,det in enumerate(given_model_dets):

        if i in overlap_det_ids:
            x1, y1, x2, y2, conf, class_name = det
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 100, 255), 2)
            draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (100, 100, 255))
                    
    
    image = add_box_color_legends_to_image(image, text_list=['Multi Class Overlap Detection'], color_list=[(100, 100, 255)], pad=500)
    image = draw_text_center_top(image, 'Detections with high overlap', cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2, (255, 200, 200))

    image[:,0] = image[:,0] * [0,0,0]
    image[:,-1] = image[:,-1] * [0,0,0]
    image[0,:] = image[0,:] * [0,0,0]
    image[-1,:] = image[-1,:] * [0,0,0]
    
    return image






def show_only_detections(image, given_model_dets, class_names, aux=False ):

    image = image.copy()

    for i,det in enumerate(given_model_dets):
        x1, y1, x2, y2, conf, class_name = det
        color = get_color(class_names.index(class_name)+1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, color)
                    
    
    text = 'auxiliary Model Detection Results' if aux else 'Given Model Detection Results'
    image = add_box_color_legends_to_image(image, text_list=[], color_list=[], pad=480)
    image = draw_text_center_top(image, text, cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2, (255, 200, 200))

    image[:,0] = image[:,0] * [0,0,0]
    image[:,-1] = image[:,-1] * [0,0,0]
    image[0,:] = image[0,:] * [0,0,0]
    image[-1,:] = image[-1,:] * [0,0,0]

    return image



def show_uncertainty_detections(image, given_model_dets, low_conf_ids):

    image = image.copy()

    for i,det in enumerate(given_model_dets):

        if i in low_conf_ids:
            x1, y1, x2, y2, conf, class_name = det
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 100, 255), 2)
            draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (100, 100, 255))
                    
    image = add_box_color_legends_to_image(image, text_list=['High Uncertainty Detection'], color_list=[(100, 100, 255)], pad=500)
    image = draw_text_center_top(image, 'Detections with High Uncertainty', cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2, (255, 200, 200))

    image[:,0] = image[:,0] * [0,0,0]
    image[:,-1] = image[:,-1] * [0,0,0]
    image[0,:] = image[0,:] * [0,0,0]
    image[-1,:] = image[-1,:] * [0,0,0]

    return image



def show_low_margin_detections(image, given_model_dets, low_conf_ids):

    image = image.copy()

    for i,det in enumerate(given_model_dets):

        if i in low_conf_ids:
            x1, y1, x2, y2, conf, class_name = det
            cv2.rectangle(image, (x1, y1), (x2, y2), (100, 100, 255), 2)
            draw_bb_text(image, f' {class_name}, cf : {str(round(conf, 2))}', (x1, y1, x2, y2),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, (100, 100, 255))
                  
    image = add_box_color_legends_to_image(image, text_list=['Low Margin Detection'], color_list=[(100, 100, 255)], pad=500)
    image = draw_text_center_top(image, 'Detections with low Margin', cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2, (255, 200, 200))

    image[:,0] = image[:,0] * [0,0,0]
    image[:,-1] = image[:,-1] * [0,0,0]
    image[0,:] = image[0,:] * [0,0,0]
    image[-1,:] = image[-1,:] * [0,0,0]

    return image
