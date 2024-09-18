import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from tqdm.auto import tqdm


def anomaly_threshold(train_data, model, weights, device='cpu'):
    model.netG.eval(); model.netE.eval(); model.netD.eval()
    errors = []

    for data in train_data:
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_recon = model.netG(model.netE(x, edge_index, batch), edge_index)
        error = torch.mean(weights * ((x_recon.detach() - x) ** 2), dim=1)
        errors.append(error.numpy())

    errors = np.stack(errors)

    return np.max(errors, axis=0)


def calculate_tpr_fpr(anomaly_preds, anomaly_labels):
    tps = 0; fps = 0; tns = 0; num_anom = 0
    anomaly_labels = anomaly_labels.reshape(*anomaly_preds.shape)
    
    for i in range(0, len(anomaly_preds)):
        pred = np.any((anomaly_preds[i]==1))
        label = np.all((anomaly_labels[i]==1))

        if np.any(anomaly_labels[i]==-1):
            continue

        if pred and label:
            tps += 1
        elif pred and not label:
            fps += 1
        elif not pred and not label:
            tns += 1
        
        if label:
            num_anom += 1

    tpr = tps / num_anom
    fpr = fps / (fps + tns)
    
    return tpr, fpr


def find_thresholds(thresh, errors, anomaly_labels):
    results = []
    
    for i in np.linspace(-0.1, 0.2, 1000):
        anomaly_preds = (errors > (thresh + i)).astype('int')
        tpr, fpr = calculate_tpr_fpr(anomaly_preds, anomaly_labels)
        results.append([fpr, i])
        
    return np.array(results)


def crash_detection_delay(anomaly_preds, crash_reported):
    anomaly_detected = np.any((anomaly_preds==1), axis=1)
    delays = []
    detects = []
    
    reported_indices = np.where(crash_reported==1)[0]
    
    for i in reported_indices:
        detected = False
        
        for j in range(i-30, i+30):
            if anomaly_detected[j] == 1:
                delays.append(j - i)
                detected = True
                break
            
        detects.append(detected)
    
    return delays, detects


def find_delays(thresh, errors, anomaly_labels, crash_reported):
    results = []
    thresholds = find_thresholds(thresh, errors, anomaly_labels)
    one_fpr_idx = np.where(thresholds[:, 0] == 1)[0][-1]
    zero_fpr_idx = np.where(thresholds[:, 0] == 0)[0][-1]
    val_range = np.linspace(0.01, 0.99, 98)
    
    anomaly_preds = (errors > (thresh + thresholds[zero_fpr_idx, 1])).astype('int')
    delays, detects = crash_detection_delay(anomaly_preds, crash_reported) 
    results.append([0, np.mean(delays), np.std(delays), np.sum(detects)/12])

    for i in tqdm(val_range):
        offset_idx = thresholds.shape[0] - 1 - np.argmin(np.abs(thresholds[:, 0][::-1] - i))
        offset = thresholds[offset_idx, 1]
        anomaly_preds = (errors > (thresh + offset)).astype('int')
        delays, detects = crash_detection_delay(anomaly_preds, crash_reported) 
        if np.sum(detects) == 0:
            delays = [30]
        results.append([thresholds[offset_idx, 0], np.mean(delays), np.std(delays), np.sum(detects)/12])
        
    anomaly_preds = (errors > (thresh + thresholds[one_fpr_idx, 1])).astype(int)
    delays, detects = crash_detection_delay(anomaly_preds, crash_reported) 
    results.append([1, np.mean(delays), np.std(delays), np.sum(detects)/12])

    return results


def calculate_delays(thresh, errors, anomaly_labels, crash_reported):
    thresholds = find_thresholds(thresh, errors, anomaly_labels)
    fprs = [.01, .025, .05, .1]
    new_thresholds = []
    anomaly_instances = []
    
    for fpr in fprs:
        closest_idx = thresholds.shape[0] - 1 - np.argmin(np.abs(thresholds[:,0][::-1] - fpr))
        new_thresholds.append(thresholds[closest_idx][1])

    for t in new_thresholds:
        anomaly_instances.append((errors > thresh + t).astype('int'))

    for i, fpr in enumerate(fprs):
        delays, detects = crash_detection_delay(anomaly_instances[i], crash_reported)
        mu = np.mean(delays) / 2
        std = np.std(delays) / 2
        miss_percent = (1 - (np.sum(detects) / len(detects))) * 100
        print(f'{fpr*100}% FPR -- Reporting Delay: {mu:.2f} +/- {std:.2f}, Miss Percentage: {miss_percent:.2f}%.')


def calculate_auc(errors, anomaly_labels):
    def anomaly_score(errors):
        return np.max(errors, axis=1)
    
    def remove_unknowable(anomaly_labels, scores):
        anomaly_reported = anomaly_labels.reshape(-1,196)[:, 0]
        known = anomaly_reported != -1
        return anomaly_reported[known], scores[known]
    
    scores = anomaly_score(errors)
    labels, scores = remove_unknowable(anomaly_labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return roc_auc

