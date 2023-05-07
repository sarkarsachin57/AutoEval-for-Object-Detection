from init import *

def RollingPositiveAverage(listA, window=3):
     s = pd.Series(listA)
     s[s < 0] = np.nan
     result = s.rolling(window, center=True, min_periods=1).mean()
     result.iloc[:window // 2] = np.nan
     result.iloc[-(window // 2):] = np.nan
     return list(result)  # or result.values or list(result) if you prefer array or list

def generate_charts_from_report(name, report, imagewise_stats, is_video = True, show = False):


    save_dir = os.path.join('process_out', 'report_charts', name)
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

    try:
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
            
    except:
        pass

    # overlaps_stats_percentage = pd.DataFrame(report['overlaps_stats_percentage'])
    # fig = px.bar(x=overlaps_stats_percentage['class_pairs'], y=overlaps_stats_percentage['percentage'], color=overlaps_stats_percentage['percentage'],
    #             labels={"x":"Classes","y":"Percentages of Overlaps"})
    # fig.update_layout(
    #     title={'text':"Class Wise Bar Chart Visualization of estimated probablity of overlap of one class with another",
    #                         'x':0.5,'y':0.97},
    #     xaxis_title="Class Pairs",
    #     yaxis_title="Scores (%) -->",
    #     font=dict(
    #         family="Georgia",
    #         size=18
    #     )
    # )
    
    # fig.write_html(os.path.join(save_dir, 'class_wise_overlap_percent_bar.html'))
    # if show:
    #     fig.show()
    #     print('\n\n')


    if is_video:

        rolling_window = 5

        fig = go.Figure(data=[
            go.Scatter(name='Precision', x=imagewise_stats['timestamp'], y=RollingPositiveAverage(imagewise_stats['PREC'], window=rolling_window)),
            go.Scatter(name='Recall', x=imagewise_stats['timestamp'], y=RollingPositiveAverage(imagewise_stats['RECL'], window=rolling_window)),
            go.Scatter(name='Accuracy', x=imagewise_stats['timestamp'], y=RollingPositiveAverage(imagewise_stats['ACC'], window=rolling_window)),
            go.Scatter(name=str(75)+'%', x=imagewise_stats['timestamp'], y=[75 for i in range(len(imagewise_stats['ACC']))], line = dict(shape = 'linear', color = 'rgb(100, 100, 100)', dash = 'dashdot')),
            go.Scatter(name=str(50)+'%', x=imagewise_stats['timestamp'], y=[50 for i in range(len(imagewise_stats['ACC']))], line = dict(shape = 'linear', color = 'rgb(100, 100, 100)', dash = 'dash')),
            go.Scatter(name=str(25)+'%', x=imagewise_stats['timestamp'], y=[25 for i in range(len(imagewise_stats['ACC']))], line = dict(shape = 'linear', color = 'rgb(100, 100, 100)', dash = 'dot')),
            
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
            go.Scatter(name='FP', x=imagewise_stats['timestamp'], y=RollingPositiveAverage(imagewise_stats['FP'], window=rolling_window)),
            go.Scatter(name='FN', x=imagewise_stats['timestamp'], y=RollingPositiveAverage(imagewise_stats['FN'], window=rolling_window)),
            go.Scatter(name='TP', x=imagewise_stats['timestamp'], y=RollingPositiveAverage(imagewise_stats['TP'], window=rolling_window)),
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
            go.Scatter(name='Alldets', x=imagewise_stats['timestamp'], y=RollingPositiveAverage(imagewise_stats['ndets'], window=rolling_window)),
            go.Scatter(name='LowConfDets', x=imagewise_stats['timestamp'], y=RollingPositiveAverage(imagewise_stats['nLowConf'], window=rolling_window)),
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
            go.Scatter(name='LowConfDets', x=imagewise_stats['timestamp'], y=RollingPositiveAverage(imagewise_stats['pLowConf'], window=rolling_window)),
            go.Scatter(name=str(75)+'%', x=imagewise_stats['timestamp'], y=[75 for i in range(len(imagewise_stats['ACC']))], line = dict(shape = 'linear', color = 'rgb(100, 100, 100)', dash = 'dashdot')),
            go.Scatter(name=str(50)+'%', x=imagewise_stats['timestamp'], y=[50 for i in range(len(imagewise_stats['ACC']))], line = dict(shape = 'linear', color = 'rgb(100, 100, 100)', dash = 'dash')),
            go.Scatter(name=str(25)+'%', x=imagewise_stats['timestamp'], y=[25 for i in range(len(imagewise_stats['ACC']))], line = dict(shape = 'linear', color = 'rgb(100, 100, 100)', dash = 'dot')),
            
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
            go.Scatter(name='No. of OverLap', x=imagewise_stats['timestamp'], y=RollingPositiveAverage(imagewise_stats['nOverLap'], window=rolling_window)),
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
