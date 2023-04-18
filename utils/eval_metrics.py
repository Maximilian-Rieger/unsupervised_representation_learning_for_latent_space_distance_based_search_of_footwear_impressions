"""Utility methods for computing evaluating metrics. All methods assumes greater
scores for better matches, and assumes label == 1 means match.

Extracted from:
https://github.com/hanxf/matchnet/blob/master/eval_metrics.py
"""
import operator
import numpy as np
import tqdm


def ErrorRateAt95Recall(labels, scores):
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    sorted_scores = zip(labels, scores)
    sorted_scores = sorted(sorted_scores, key=operator.itemgetter(1), reverse=False)

    # Compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match
    count = 0
    tp = 0
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break

    return float(count - tp) / count


def ROC(labels, scores):
    # Sort label-score tuples by the score in descending order.
    sorted_scores = zip(labels, scores)
    sorted_scores = sorted(sorted_scores, key=operator.itemgetter(1), reverse=False)

    # Compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    count = 0
    tp = 0

    thresh = []

    res = []
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
            thresh.append(score)
            fp = n_match-tp
            tn = count-tp
            fn = len(sorted_scores)-n_match-count-tp
            precision = tp / (tp + fp + np.finfo(float).eps)
            recall = tp / (tp + fn + np.finfo(float).eps)

            res.append({'threshold': thresh, 'precision': precision, 'recall': recall,
                        'true_positives': tp,
                        'true_negatives': tn, 'false_positives': fp,
                        'false_negatives': fn})

    return res


def generate_pr_curve(labels, distances, num_thresholds=100):
    # sort_dist = np.sort(distances)
    # thresholds = np.geomspace(1, len(sort_dist) - 1, self.options.get('number_of_thresholds', 100), dtype=int)
    # thresholds = np.logspace(1, np.log2(len(sort_dist) - 1), num_thresholds,
    #                          dtype=int, base=2)
    norm_dists = distances/np.max(distances)
    threshold = np.linspace(0, 1, num_thresholds)
    res = []
    for t in tqdm.tqdm(threshold, desc='generating PR curve'):
        true_positives = np.sum(np.logical_and(norm_dists < t, labels == 1))
        false_positives = np.sum(np.logical_and(norm_dists < t, labels == 0))
        true_negatives = np.sum(np.logical_and(norm_dists >= t, labels == 0))
        false_negatives = np.sum(np.logical_and(norm_dists >= t, labels == 1))
        precision = true_positives / (true_positives + false_positives + np.finfo(float).eps)
        recall = true_positives / (true_positives + false_negatives + np.finfo(float).eps)

        res.append({'threshold': t*np.max(distances), 'norm_threshold': t, 'precision': precision, 'recall': recall,
                    'true_positives': true_positives,
                    'true_negatives': true_negatives, 'false_positives': false_positives,
                    'false_negatives': false_negatives})

    return res
