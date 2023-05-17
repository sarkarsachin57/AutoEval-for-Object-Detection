from init import *
from eval_vis import *



def describe(df, stats):
    d = df.describe()
    return pd.concat([d,df.reindex(d.columns, axis = 1).agg(stats)])


def get_class_based_stats(class_specific_stats, tp_threshold):

    alldets_counts = np.unique(np.concatenate(class_specific_stats['all_classes']), return_counts=True)
    tp_counts = np.unique(np.concatenate(class_specific_stats['TP']), return_counts=True)
    fp_counts = np.unique(np.concatenate(class_specific_stats['FP']), return_counts=True)
    fn_counts = np.unique(np.concatenate(class_specific_stats['FN']), return_counts=True)

    class_based_values = {}
    class_based_values['class_name'] = []
    class_based_values['TP'] = []
    class_based_values['FP'] = []
    class_based_values['FN'] = []


    for class_name in np.unique(np.concatenate([tp_counts[0],fp_counts[0],fn_counts[0]])):

        class_based_values['class_name'].append(class_name)

        try:
            class_based_values['TP'].append(int(tp_counts[1][list(tp_counts[0]).index(class_name)]))
        except:
            class_based_values['TP'].append(0)

        try:
            class_based_values['FP'].append(int(fp_counts[1][list(fp_counts[0]).index(class_name)]))
        except:
            class_based_values['FP'].append(0)    
        
        try:
            class_based_values['FN'].append(int(fn_counts[1][list(fn_counts[0]).index(class_name)]))
        except:
            class_based_values['FN'].append(0)

        
    class_based_values = pd.DataFrame(class_based_values) 
    class_based_values['ACC'] = round(((2*class_based_values['TP']+1e-10) * 100) / (2*class_based_values['TP'] + class_based_values['FP'] + class_based_values['FN'] + 1e-10), 2)
    class_based_values['PREC'] = round(((class_based_values['TP']+1e-10) * 100) / (class_based_values['TP'] + class_based_values['FP'] + 1e-10), 2)
    class_based_values['RECL'] = round(((class_based_values['TP']+1e-10) * 100) / (class_based_values['TP'] + class_based_values['FN'] + 1e-10), 2)

    class_based_values = class_based_values[class_based_values['TP'] >= tp_threshold]

    overall_agg = class_based_values.agg({'TP': 'sum', 'FP': 'sum', 'FN': 'sum', 'ACC' : 'mean', 'PREC': 'mean', 'RECL': 'mean'})
    class_based_values = pd.concat([class_based_values, pd.DataFrame(pd.concat([pd.Series({'class_name':'All Class Aggregation'}),overall_agg])).T]).round(2).sort_values(by='ACC', ascending=False).reset_index().drop(['index'], axis=1)
    class_based_values = pd.concat([class_based_values.drop(class_based_values[class_based_values['class_name'] == 'All Class Aggregation'].index[0]), class_based_values[class_based_values['class_name'] == 'All Class Aggregation']])

    return class_based_values.to_dict()



def get_low_conf_stats(class_specific_stats, alldets_threshold):

    alldets_counts = np.unique(np.concatenate(class_specific_stats['all_classes']), return_counts=True)
    lowconf_counts = np.unique(np.concatenate(class_specific_stats['LowConf']), return_counts=True)

    class_based_values = {}
    class_based_values['class_name'] = []
    class_based_values['all_classes'] = [] 
    class_based_values['lowconf'] = []


    for class_name in np.unique(alldets_counts[0]):

        class_based_values['class_name'].append(class_name)

        try:
            class_based_values['all_classes'].append(int(alldets_counts[1][list(alldets_counts[0]).index(class_name)]))
        except:
            class_based_values['all_classes'].append(0)

        try:
            class_based_values['lowconf'].append(int(lowconf_counts[1][list(lowconf_counts[0]).index(class_name)]))
        except:
            class_based_values['lowconf'].append(0)
            
        
    class_based_values = pd.DataFrame(class_based_values) 
    class_based_values['LowConfPercent'] = round((class_based_values['lowconf'] * 100) / (class_based_values['all_classes']+1e-10), 2)

    class_based_values = class_based_values[class_based_values['all_classes'] >= alldets_threshold]

    overall_agg = class_based_values.agg({'all_classes': 'sum', 'lowconf' : 'sum', 'LowConfPercent': 'mean'})
    class_based_values = pd.concat([class_based_values, pd.DataFrame(pd.concat([pd.Series({'class_name':'All Class Aggregation'}),overall_agg])).T]).round(2).sort_values(by='all_classes', ascending=False).reset_index().drop(['index'], axis=1)
    class_based_values = pd.concat([class_based_values.drop(class_based_values[class_based_values['class_name'] == 'All Class Aggregation'].index[0]), class_based_values[class_based_values['class_name'] == 'All Class Aggregation']])

    return class_based_values.to_dict()


def get_avg_conf_stats(imagewise_classlist):
    imagewise_classlist_df = pd.DataFrame({"all_classes":list_concat(imagewise_classlist['all_classes']),
        "all_confs": list_concat(imagewise_classlist['all_confs'])})
    return imagewise_classlist_df.groupby('all_classes')['all_confs'].mean().apply(lambda x: x*100).sort_values().reset_index().to_dict()

    
def get_avg_uncertainty_stats(imagewise_classlist):
    imagewise_classlist_df = pd.DataFrame({"all_classes":list_concat(imagewise_classlist['all_classes']),
    "all_entropies": list_concat(imagewise_classlist['all_entropies'])})
    return imagewise_classlist_df.groupby('all_classes')['all_entropies'].mean().sort_values(ascending=False).reset_index().to_dict()



def get_margin_stats(imagewise_classlist, count_threshold):
    imagewise_classlist_df = pd.DataFrame({"margin_class_pairs": list_concat(imagewise_classlist['margin_class_pairs']),
    "all_margins": list_concat(imagewise_classlist['all_margins'])})

    margin_class_pairs_df = pd.DataFrame([imagewise_classlist_df.groupby('margin_class_pairs')['all_margins'].count(), 
                                          imagewise_classlist_df.groupby('margin_class_pairs')['all_margins'].mean()]).T 
    margin_class_pairs_df.columns = ['count', 'avg_margin']
    margin_class_pairs_df = margin_class_pairs_df.sort_values(by='count', ascending=False)
    margin_class_pairs_df = margin_class_pairs_df[margin_class_pairs_df['count'] >= count_threshold]

    return margin_class_pairs_df.reset_index().to_dict()


def get_overlap_stats(imagewise_classdist, count_threshold):

    overlap_pair = []
    for pairs in imagewise_classdist['OverLap']:
        if len(pairs) != 0:
            for pair in pairs:
                overlap_pair.append(pair)

    alldets_uni_counts = np.unique(np.concatenate(imagewise_classdist['all_classes']), return_counts=True)
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


def generate_report(imagewise_stats, imagewise_classdist, tp_threshold, alldets_threshold, margin_count_threshold, overlap_count_threshold, print_report = False):

    report_json = {}

    report_json['overall'] = {}
        
    abs_values = describe(pd.DataFrame(imagewise_stats).loc[:,['ndets', 'TP', 'FP', 'FN', 'nLowConf', 'nOverLap']], ['sum']).iloc[1:].round(2)
    abs_values.index = ['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'Total']
    report_json['overall']['absolute'] = abs_values.to_dict()
    report_json['overall']['relative'] = pd.DataFrame(imagewise_stats).loc[:,['ACC', 'PREC', 'RECL', 'pLowConf', 'avg_conf', 'uncertainty', 'margin']].describe().iloc[1:].round(2).to_dict()

    report_json['overall']['relative']['ACC']['Overall'] = round(100*(2*abs_values['TP']['Total']+1e-10) / (2*abs_values['TP']['Total']+abs_values['FP']['Total']+abs_values['FN']['Total']+1e-10), 2)
    report_json['overall']['relative']['PREC']['Overall'] = round(100*(abs_values['TP']['Total']+1e-10) / (abs_values['TP']['Total']+abs_values['FP']['Total']+1e-10), 2)
    report_json['overall']['relative']['RECL']['Overall'] = round(100*(abs_values['TP']['Total']+1e-10) / (abs_values['TP']['Total']+abs_values['FN']['Total']+1e-10), 2)
    report_json['overall']['relative']['pLowConf']['Overall'] = round(100*abs_values['nLowConf']['Total'] / (abs_values['nLowConf']['Total']+abs_values['ndets']['Total']+1e-10), 2)

    imagewise_total_conf = np.array(imagewise_stats['avg_conf']) * np.array(imagewise_stats['ndets'])
    avg_conf = np.sum(imagewise_total_conf[np.logical_not(np.isnan(imagewise_total_conf))]) / np.sum(imagewise_stats['ndets'])

    imagewise_total_un = np.array(imagewise_stats['uncertainty']) * np.array(imagewise_stats['ndets'])
    avg_un = np.sum(imagewise_total_un[np.logical_not(np.isnan(imagewise_total_un))]) / np.sum(imagewise_stats['ndets'])

    imagewise_total_margin = np.array(imagewise_stats['margin']) * np.array(imagewise_stats['ndets'])
    avg_margin = np.sum(imagewise_total_margin[np.logical_not(np.isnan(imagewise_total_margin))]) / np.sum(imagewise_stats['ndets'])
    
    report_json['overall']['relative']['avg_conf']['Overall'] = round(avg_conf, 2)
    report_json['overall']['relative']['uncertainty']['Overall'] = round(avg_un, 2)
    report_json['overall']['relative']['margin']['Overall'] = round(avg_margin, 2)

    report_json['class_based'] = get_class_based_stats(imagewise_classdist, tp_threshold)
    report_json['low_conf_stats'] = get_low_conf_stats(imagewise_classdist, alldets_threshold)
    report_json['avg_conf_stats'] = get_avg_conf_stats(imagewise_classdist)
    report_json['uncertainty_stats'] = get_avg_uncertainty_stats(imagewise_classdist)
    report_json['margin_stats'] = get_margin_stats(imagewise_classdist, margin_count_threshold)
    report_json['overlaps_stats_percentage'], report_json['overlaps_stats_count'] = get_overlap_stats(imagewise_classdist, overlap_count_threshold)

    if print_report:
        print("\nOverall Absolute Results :-\n")
        print(pd.DataFrame(report_json['overall']['absolute']))
        print("\nOverall Relative Results :-\n")
        print(pd.DataFrame(report_json['overall']['relative']))
        print('\nClass Based Results :-\n')
        print(pd.DataFrame(report_json['class_based']))
        print('\nLow Conf results :-\n')
        print(pd.DataFrame(report_json['low_conf_stats']))
        print('\nAverage Conf results :-\n')
        print(pd.DataFrame(report_json['avg_conf_stats']))
        print('\nUncertainty results :-\n')
        print(pd.DataFrame(report_json['uncertainty_stats']))
        print('\nMargin results :-\n')
        print(pd.DataFrame(report_json['margin_stats']))
        print('\nOverlap detection classwise counts :-\n')
        print(pd.DataFrame(report_json['overlaps_stats_count']))

    return report_json






def multi_generate_report(multi_stats_json, print_report = True, show_vis = True, show_only_final = True):

    multi_report_json = {}

    multi_report_json['iter_no'] = []
    multi_report_json['name'] = []
    multi_report_json['report'] = []
    multi_report_json['is_video'] = []

    print('\n\nGenerating Report and Charts...')

    for it in multi_stats_json['iter_no']:

        name = multi_stats_json['name'][it]
        imagewise_stats = multi_stats_json['imagewise_stats'][it]
        imagewise_classdist = multi_stats_json['imagewise_classdist'][it]
        is_video = multi_stats_json['is_video'][it]

        print(f'\nIteration : {it}, Name : {name}, is_video : {is_video}')

        print_report_temp = print_report if not show_only_final or it == multi_stats_json['iter_no'][-1] else False
        report = generate_report(imagewise_stats, imagewise_classdist, tp_threshold = 0, alldets_threshold = 1, overlap_count_threshold = 1, print_report = print_report_temp)
        
        show_vis_temp = show_vis if not show_only_final or it == multi_stats_json['iter_no'][-1] else False
        generate_charts_from_report(name, report, imagewise_stats, is_video=is_video, show=show_vis_temp)

        multi_report_json['iter_no'].append(it)
        multi_report_json['name'].append(name)
        multi_report_json['report'].append(report)
        multi_report_json['is_video'].append(is_video)

    return multi_report_json