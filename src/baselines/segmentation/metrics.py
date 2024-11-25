import numpy as np
import nibabel as nib
import cc3d
import scipy
import os
import pandas as pd
import src.baselines.segmentation.surface_distance as surface_distance
import sys
import math


def dice(im1, im2):
    """
    Computes Dice score for two images

    Parameters
    ==========
    im1: Numpy Array/Matrix; Predicted segmentation in matrix form 
    im2: Numpy Array/Matrix; Ground truth segmentation in matrix form

    Output
    ======
    dice_score: Dice score between two images
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * (intersection.sum()) / (im1.sum() + im2.sum())


def get_sensitivity_and_specificity(result_array, target_array):
    """
    This function is extracted from GaNDLF from mlcommons

    You can find the documentation here - 

    https://github.com/mlcommons/GaNDLF/blob/master/GANDLF/metrics/segmentation.py#L196

    """
    iC = np.sum(result_array)
    rC = np.sum(target_array)

    overlap = np.where((result_array == target_array), 1, 0)

    # Where they agree are both equal to that value
    TP = overlap[result_array == 1].sum()
    FP = iC - TP
    FN = rC - TP
    TN = np.count_nonzero((result_array != 1) & (target_array != 1))

    Sens = 1.0 * TP / (TP + FN + sys.float_info.min)
    Spec = 1.0 * TN / (TN + FP + sys.float_info.min)

    # Make Changes if both input and reference are 0 for the tissue type
    if (iC == 0) and (rC == 0):
        Sens = 1.0

    return Sens, Spec


def get_Predseg_combinedByDilation(pred_dilated_cc_mat, pred_label_cc):
    """
    Computes the Corrected Connected Components after combing lesions
    together with respect to their dilation extent

    Parameters
    ==========
    pred_dilated_cc_mat: Numpy Array/Matrix; Ground Truth Dilated Segmentation 
                       after CC Analysis
    pred_label_cc: Numpy Array/Matrix; Ground Truth Segmentation after 
                       CC Analysis

    Output
    ======
    pred_seg_combinedByDilation_mat: Numpy Array/Matrix; Ground Truth 
                                   Segmentation after CC Analysis and 
                                   combining lesions
    """

    pred_seg_combinedByDilation_mat = np.zeros_like(pred_dilated_cc_mat)

    for comp in range(np.max(pred_dilated_cc_mat)):
        comp += 1

        pred_d_tmp = np.zeros_like(pred_dilated_cc_mat)
        pred_d_tmp[pred_dilated_cc_mat == comp] = 1
        pred_d_tmp = (pred_label_cc*pred_d_tmp)

        np.place(pred_d_tmp, pred_d_tmp > 0, comp)
        pred_seg_combinedByDilation_mat += pred_d_tmp

    return pred_seg_combinedByDilation_mat


def get_GTseg_combinedByDilation(gt_dilated_cc_mat, gt_label_cc):
    """
    Computes the Corrected Connected Components after combing lesions
    together with respect to their dilation extent

    Parameters
    ==========
    gt_dilated_cc_mat: Numpy Array/Matrix; Ground Truth Dilated Segmentation 
                       after CC Analysis
    gt_label_cc: Numpy Array/Matrix; Ground Truth Segmentation after 
                       CC Analysis

    Output
    ======
    gt_seg_combinedByDilation_mat: Numpy Array/Matrix; Ground Truth 
                                   Segmentation after CC Analysis and 
                                   combining lesions
    """

    gt_seg_combinedByDilation_mat = np.zeros_like(gt_dilated_cc_mat)

    for comp in range(np.max(gt_dilated_cc_mat)):
        comp += 1

        gt_d_tmp = np.zeros_like(gt_dilated_cc_mat)
        gt_d_tmp[gt_dilated_cc_mat == comp] = 1
        gt_d_tmp = (gt_label_cc*gt_d_tmp)

        np.place(gt_d_tmp, gt_d_tmp > 0, comp)
        gt_seg_combinedByDilation_mat += gt_d_tmp

    return gt_seg_combinedByDilation_mat


def process_segmentation(pred_mat, gt_mat, dil_factor):
    """
    Processes the segmentation matrices and returns the combined connected components arrays.

    Parameters
    ==========
    pred_mat: Numpy Array/Matrix; Predicted segmentation matrix
    gt_mat: Numpy Array/Matrix; Ground truth segmentation matrix
    dil_factor: int; Dilation factor for processing

    Output
    ======
    pred_mat_combinedByDilation: Numpy Array/Matrix; Processed predicted segmentation matrix
    gt_mat_combinedByDilation: Numpy Array/Matrix; Processed ground truth segmentation matrix
    """

    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)

    # Process ground truth segmentation
    gt_mat_cc = cc3d.connected_components(gt_mat, connectivity=26)
    gt_mat_dilation = scipy.ndimage.binary_dilation(
        gt_mat, structure=dilation_struct, iterations=dil_factor)
    gt_mat_dilation_cc = cc3d.connected_components(
        gt_mat_dilation, connectivity=26)

    gt_mat_combinedByDilation = get_GTseg_combinedByDilation(
        gt_dilated_cc_mat=gt_mat_dilation_cc,
        gt_label_cc=gt_mat_cc)

    # Process predicted segmentation
    pred_mat_cc = cc3d.connected_components(pred_mat, connectivity=26)
    pred_mat_dilation = scipy.ndimage.binary_dilation(
        pred_mat, structure=dilation_struct, iterations=dil_factor)
    pred_mat_dilation_cc = cc3d.connected_components(
        pred_mat_dilation, connectivity=26)

    pred_mat_combinedByDilation = get_Predseg_combinedByDilation(
        pred_dilated_cc_mat=pred_mat_dilation_cc,
        pred_label_cc=pred_mat_cc)

    # Remove small lesions
    lesion_volume_thresh = 5
    labels, counts = np.unique(pred_mat_combinedByDilation, return_counts=True)
    labels_to_remove = labels[counts <= lesion_volume_thresh]
    mask = np.isin(pred_mat_combinedByDilation, labels_to_remove, invert=True)
    pred_mat_combinedByDilation = np.where(
        mask, pred_mat_combinedByDilation, 0)

    return pred_mat_combinedByDilation, gt_mat_combinedByDilation


def get_LesionWiseScores(pred_mat, gt_mat, dil_factor, lesion_volume_thresh=None):
    # Assuming isotropic voxels with a default size of 1.0 for each dimension
    sx, sy, sz = 1.0, 1.0, 1.0

    # Get Dice score for the full image
    if np.all(gt_mat == 0) and np.all(pred_mat == 0):
        full_dice = 1.0
    else:
        full_dice = dice(
            pred_mat,
            gt_mat
        )

    # Get HD95 score for the full image
    if np.all(gt_mat == 0) and np.all(pred_mat == 0):
        full_hd95 = 0.0
    else:
        full_sd = surface_distance.compute_surface_distances(gt_mat.astype(int),
                                                             pred_mat.astype(
                                                                 int),
                                                             (sx, sy, sz))
        full_hd95 = surface_distance.compute_robust_hausdorff(full_sd, 95)

    # Get Sensitivity and Specificity
    full_sens, full_specs = get_sensitivity_and_specificity(result_array=pred_mat,
                                                            target_array=gt_mat)

    dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)

    # Get GT Volume and Pred Volume for the full image
    full_gt_vol = np.sum(gt_mat > 0)*sx*sy*sz
    full_pred_vol = np.sum(pred_mat > 0)*sx*sy*sz

    pred_mat_cc = pred_mat

    gt_label_cc = gt_mat.astype(
        np.int32)
    pred_label_cc = pred_mat_cc.astype(
        np.int32)

    gt_tp = []
    tp = []
    fn = []
    fp = []
    metric_pairs = []
    for gtcomp in range(np.max(gt_label_cc)):
        gtcomp += 1

        # Extracting current lesion
        gt_tmp = np.zeros_like(gt_label_cc)
        gt_tmp[gt_label_cc == gtcomp] = 1

        # Extracting ROI GT lesion component
        gt_tmp_dilation = scipy.ndimage.binary_dilation(
            gt_tmp, structure=dilation_struct, iterations=dil_factor)

        # Volume of lesion
        gt_vol = np.sum(gt_tmp > 0)*sx*sy*sz

        # Extracting Predicted true positive lesions
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp = pred_tmp*gt_tmp_dilation
        intersecting_cc = np.unique(pred_tmp)
        intersecting_cc = intersecting_cc[intersecting_cc != 0]
        for cc in intersecting_cc:
            tp.append(cc)

        # Isolating Predited Lesions to calulcate Metrics
        pred_tmp = np.copy(pred_label_cc)
        pred_tmp[np.isin(pred_tmp, intersecting_cc, invert=True)] = 0
        pred_tmp[np.isin(pred_tmp, intersecting_cc)] = 1

        # Calculating Lesion-wise Dice and HD95
        dice_score = dice(pred_tmp, gt_tmp)
        surface_distances = surface_distance.compute_surface_distances(
            gt_tmp, pred_tmp, (sx, sy, sz))
        hd = surface_distance.compute_robust_hausdorff(surface_distances, 95)

        metric_pairs.append((intersecting_cc,
                            gtcomp, gt_vol, dice_score, hd))

        # Extracting Number of TP/FP/FN and other data
        if len(intersecting_cc) > 0:
            gt_tp.append(gtcomp)
        else:
            fn.append(gtcomp)

    fp = np.unique(
        pred_label_cc[np.isin(
            pred_label_cc, tp+[0], invert=True)])
    
    if lesion_volume_thresh is not None:
        fp = [f for f in fp if np.sum(pred_label_cc == f)*sx*sy*sz > lesion_volume_thresh]

    return tp, fn, fp, gt_tp, metric_pairs, full_dice, full_hd95, full_gt_vol, full_pred_vol, full_sens, full_specs


def get_LesionWiseResults(pred, label):
    """
    Computes the Lesion-wise scores for pair of prediction and ground truth segmentations
    """

    dilation_factor = 1
    lesion_volume_thresh = 100

    pred_mat, gt_mat = process_segmentation(pred, label, dilation_factor)

    tp, fn, fp, gt_tp, metric_pairs, full_dice, full_hd95, full_gt_vol, full_pred_vol, full_sens, full_specs = get_LesionWiseScores(
        pred_mat=pred_mat,
        gt_mat=gt_mat,
        dil_factor=dilation_factor,
        lesion_volume_thresh=lesion_volume_thresh
    )

    metric_df = pd.DataFrame(
        metric_pairs, columns=['predicted_lesion_numbers', 'gt_lesion_numbers',
                                'gt_lesion_vol', 'dice_lesionwise', 'hd95_lesionwise']
    ).sort_values(by=['gt_lesion_numbers'], ascending=True).reset_index(drop=True)

    metric_df['_len'] = metric_df['predicted_lesion_numbers'].map(len)

    # Removing <= 50 lesions from analysis
    fn_sub = (metric_df[(metric_df['_len'] == 0) &
                (metric_df['gt_lesion_vol'] <= lesion_volume_thresh)
    ]).shape[0]

    gt_tp_sub = (metric_df[(metric_df['_len'] != 0) &
                            (metric_df['gt_lesion_vol']
                            <= lesion_volume_thresh)
                            ]).shape[0]
    
    metric_df_thresh = metric_df[metric_df['gt_lesion_vol'] > lesion_volume_thresh]
    metric_df_thresh.loc[:, ['dice_lesionwise', 'hd95_lesionwise']] = metric_df_thresh[['dice_lesionwise', 'hd95_lesionwise']].replace(np.inf, 374)
    if len(metric_df_thresh) == 0:
        return None

    try:
        lesion_wise_dice = np.sum(metric_df_thresh['dice_lesionwise'])/(len(metric_df_thresh) + len(fp))
    except:
        lesion_wise_dice = np.nan

    try:
        lesion_wise_hd95 = (np.sum(metric_df_thresh['hd95_lesionwise']) + len(fp)*374)/(len(metric_df_thresh) + len(fp))
    except:
        lesion_wise_hd95 = np.nan

    if math.isnan(lesion_wise_dice):
        lesion_wise_dice = 1

    if math.isnan(lesion_wise_hd95):
        lesion_wise_hd95 = 0
        
    if full_dice == np.inf:
        full_dice = 0
    if full_hd95 == np.inf:
        full_hd95 = 374

    metrics_dict = {
        # 'Num_TP': len(gt_tp) - gt_tp_sub,  # GT_TP
        # 'Num_FP': len(fp),
        # 'Num_FN': len(fn) - fn_sub,
        # 'Sensitivity': full_sens,
        # 'Specificity': full_specs,
        'Legacy_Dice': full_dice,
        'Legacy_HD95': full_hd95,
        # 'GT_Complete_Volume': full_gt_vol,
        'LesionWise_Score_Dice': lesion_wise_dice,
        'LesionWise_Score_HD95': lesion_wise_hd95
    }
    return metrics_dict
