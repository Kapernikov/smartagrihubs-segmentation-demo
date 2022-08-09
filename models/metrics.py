from keras.metrics import OneHotMeanIoU


def get_metrics_by_name(metrics_names, num_classes):
    '''
    Returns a metric function by name.
        - OneHotMeanIoU
    '''    
    # Check if provided metric(s) is an array
    if not isinstance(metrics_names, list):
        metrics_names = [metrics_names]
        
    metrics = []

    for metric in metrics_names:
        if metric =='OneHotMeanIoU':
            metrics.append(OneHotMeanIoU(num_classes=num_classes,name='1Hot_Mean_iou'))

    return metrics 