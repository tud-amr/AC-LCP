import cv2
import numpy as np
import time
from PIL import Image
from torchreid import models
from torchreid.utils import load_pretrained_weights
from torchvision.transforms import (
    Resize,Compose, ToTensor, Normalize)
from torch.nn import functional as F
from scipy.spatial import distance
from tracking_system.Utilities.geometryCam import cutImage


class ReId:
    def __init__(self):
        """
        Load re-identification model to obtain appearance information
        """
        self.model = models.build_model(name='osnet_x0_25', num_classes=1000)
        load_pretrained_weights(self.model, '/home/scasao/pytorch/deep-person-reid/configs/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth')
        self.model.cuda()
        self.model.eval()
        
        height=256
        width=128
        
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
        normalize = Normalize(mean=norm_mean, std=norm_std)
        
        self.transform_te = Compose([Resize((height, width)), ToTensor(), normalize])
        
    def extractFeatures(self, img):
        """
        Parameters
        ----------
        img : numpy array. 
            Cropped image of the person to extract features          

        Returns
        -------
        features : numpy array 
            Output of re-identification network
        """
        img = Image.fromarray(img)
        img = self.transform_te(img)
        img = img.cuda()
        img = img.unsqueeze(0)

        features = self.model(img)        
        features = F.normalize(features, p=2, dim=1)
        features = features.data.cpu()
        features = features.numpy()

        return features
        
    def distance_list(self, imgA, featuresB):
        featuresA = self.extractFeatures(imgA)
        scores = distance.cdist(featuresA, featuresB, 'cosine')    

        return scores

    def distance(self, imgA, imgB):
        featuresA = self.extractFeatures(imgA)
        featuresB = self.extractFeatures(imgB)
        score = distance.cosine(featuresA, featuresB)

        return score


def check_appearance(track):
    if len(track.appearance) > 1: 
        appearance = track.appearance
    else:         
        appearance = None
    return appearance   


def compute_appearance_score(reid, trackers, frame, bbox):
    """One bbox, several trackers. Return the nearest tracker in appearance"""
    img_detect = cutImage(frame, bbox)
    scores, bestPatches_ind = [], []
    for track in trackers:
        gallery = track.appearance
        if len(gallery) > 0:
            feat_detect = np.array(reid.extractFeatures(img_detect))
            scores_p = distance.cdist(feat_detect, gallery, 'cosine')[0]
            scores.append(np.amin(scores_p))
            bestPatches_ind.append(scores_p.tolist().index(np.amin(scores_p)))
        else:
            scores.append(1)
            bestPatches_ind.append(np.nan)
    return scores, bestPatches_ind


def applyAppThreshold(scores_app, scores_geom, trackers_candidates, threshold, best_patches = None):
    app_selected, geom_selected, track_selected, best_p = [],[],[],[]
    for i, f in enumerate(scores_app):
        if f <= threshold:
            app_selected.append(f)
            geom_selected.append(scores_geom[i])
            track_selected.append(trackers_candidates[i])
            if best_patches is not None:
                best_p.append(best_patches[i])
    if best_patches is None:
        return app_selected, geom_selected, track_selected 
    else:
        return app_selected, geom_selected, track_selected, best_p


def checkAPP_crossTrackers(aux_trackers, unrelated_trackers1, unrelated_trackers2, camera, camera2):
    trackers_related = []
    for unrelated_track1 in unrelated_trackers1:
        app1 = check_appearance(unrelated_track1)
        for unrelated_track2 in unrelated_trackers2:
            app2 = check_appearance(unrelated_track2)
            if app2 is not None and app1 is not None and unrelated_track2 not in trackers_related:
                distances = distance.cdist(app1, app2, 'cosine')
                min_dist = np.round(np.amin(distances), 2)
                # same person
                if min_dist == 0.00:
                    unrelated_track1.LUT[camera2] = [unrelated_track2.id, unrelated_track2.color]
    
                    index_track = aux_trackers[camera].index(unrelated_track1)
                    aux_trackers[camera][index_track] = unrelated_track1
                    trackers_related.append(unrelated_trackers2)
                    break
    return aux_trackers

