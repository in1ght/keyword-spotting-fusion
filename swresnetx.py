# Imports
import gc
import warnings
import argparse
import torch
import torch

import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Ignore warnings
warnings.filterwarnings('ignore')

# Imports from other local python files
from models import load_models

class Dataset_frames(torch.utils.data.Dataset):
    """
    Sliding-window processing dataset.

    Generates frames of fixed-size using a sliding window.
    allows switching between mel-only or combined (mel + MFCC) inputs. 
    (two different models further down the pipeline used for different data)

    Args:
        path - path to numpy with spectrogram data (Path)
        encoder - label encoder (object)
        frame_step - step size for sliding window (int, default=1)
        data_requsted - controls input type (0=mel, 1=mel+mfcc) (int, default=0)

    Returns:
        torch.utils.data.Dataset:
            iterable dataset of framed spectrogram inputs
    """
    def __init__(self, path: Path, encoder: object, frame_step: int = 1, data_requsted: int = 0):
        if frame_step < 1 or not isinstance(frame_step, int):
            raise ValueError("frame step ought to be more than 0 and int")
        data_file = np.load(path)
        samples_list_wrt_frame_step = []

        if len(data_file.shape) == 2:
            data_file = data_file.reshape((-1,data_file.shape[0],data_file.shape[1]))

        length = data_file.shape[2]
        frame = 0
        while 44+frame < length:
            samples_list_wrt_frame_step.append(data_file[0,:,frame:44+frame])
            frame += frame_step
    
        self.data_file = np.array(samples_list_wrt_frame_step)
        self.data_requsted = data_requsted
        self.frame_step = frame_step
        self.encoder = encoder
            

    def __len__(self):
        return self.data_file.shape[0]

    def __getitem__(self, idx: int):
        item = self.data_file[idx]
        melspect = item[12:76, :]
        melspect_torch = self.transform_data(melspect)

        if self.data_requsted == 1:
            mfss = np.concatenate((item[:12, :], item[76:, :]), axis=0)
            mmfss_torch = self.transform_data(mfss)
            return melspect_torch, mmfss_torch

        return melspect_torch

    # Helpers

    def set_to_mels_and_mfss(self, flag: int = 0):
        """
        Changes data mode (0=mel, 1=mel+mfcc).

        Args:
            flag - mode selector (int)
        """
        self.data_requsted = flag

    def transform_data(self, data: np.ndarray):
        """
        Reshapes raw data into tensor format.

        Args:
            data - input array (np.ndarray)

        Returns:
            torch.Tensor
        """
        data_reshaped = data.reshape((1, data.shape[0], data.shape[1]))
        return torch.Tensor(data_reshaped)
    

    def get_predictions(self, model: torch.nn.Module):
        """
        Generates predictions for each frame using a single model.

        Args:
            model - trained model (nn.Module)

        Returns:
            np.ndarray:
                Predicted class indices per frame
        """
        model.eval().to('cuda')
        preds = []
        for _, data in enumerate(self):
            data = data.reshape((1,1, data.shape[1], data.shape[2]))
            data = data.to('cuda')
            outputs = model(data)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            preds.append(outputs[0])

        return np.array(preds)
    

    def get_predictions_3class(self, model_mfcc: torch.nn.Module, model_melspect: torch.nn.Module, network_clas: torch.nn.Module):
        """
        Generates predictions using two feature extractors models and a classifier.

        Use this to processes MFCC and mel-spectrogram inputs separately.
        Then it concatenates outputs and feeds into a classifier.

        Args:
            model_mfcc - model for MFCC input (nn.Module)
            model_melspect - model for mel-spectrogram input (nn.Module)
            network_clas - classifier combining both outputs (nn.Module)

        Returns:
            np.ndarray:
                Predicted class indices per frame
        """
        melspect = self.data_file[:,12:76, :]
        melspect = melspect.reshape(melspect.shape[0],1,melspect.shape[1],melspect.shape[2])

        mfss = np.concatenate((self.data_file[:,:12, :], self.data_file[:,76:, :]), axis=1)
        mfss = mfss.reshape(mfss.shape[0],1,mfss.shape[1],mfss.shape[2])

        mfss = DataLoader(torch.Tensor(mfss), batch_size=30, shuffle=False)
        melspect = DataLoader(torch.Tensor(melspect), batch_size=30, shuffle=False)

        predictions = []
        for mfc_c, mel_c in zip(mfss,melspect):
            mfc_c = mfc_c.to(torch.device("cuda"))
            mel_c = mel_c.to(torch.device("cuda"))
            output_1 = model_mfcc(mfc_c)
            output_2 = model_melspect(mel_c)

            output_1_2 = torch.cat((output_1, output_2), 1)

            outputs = network_clas(output_1_2)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            predictions.append(outputs)

        del mfc_c
        del mel_c
        gc.collect()
        torch.cuda.empty_cache()

        predictions = np.concatenate((predictions), axis=0)

        return predictions
    
    def eval_print(self,model: torch.nn.Module = None, predictions: list = None):
        """
        Prints detected words with duration and timestamps.

        Args:
            model - model for prediction (optional)
            predictions - precomputed predictions (optional)
        """
        last, count_frames, start_frame = 10, 0, 0
        if predictions is None:
            predictions = self.get_predictions(model)
        for i, item in enumerate(predictions):
            if item == 10: # i.e. == other
                if last != 10:
                    print(f"Word \{self.encoder.classes_[last]}\ was spotted for \{count_frames}\ frames starting from \{start_frame}\ frame, i.e. \{round(1.1*start_frame*(self.frame_step/44),2)}\ second")
                    last = 10       
            else:
                if item == last:
                    count_frames += 1
                else:
                    if last != 10:
                        print(f"Word \{self.encoder.classes_[last]}\ was spotted for \{count_frames}\ frames starting from \{start_frame}\ frame, i.e. \{round(1.1*start_frame*(self.frame_step/44),2)}\ second")
                    count_frames,start_frame,last = 1, i, item  

    
    def predict_with_key_word_system(
            self,
            model: torch.nn.Module,
            min_freq: int = 2, 
            included_frames: int = 15, 
            key_frames: tuple = (3,2),
            predictions: list = None
        ):
        """
        Detects keyword phrases using temporal voting and key-frame logic.

        Identifies frequent words within a sliding window. Combines detected words with commands and 
        outputs detected phrases with timestamps, given enough confidence that a specific word occured.

        Args:
            model - prediction model (nn.Module)
            min_freq - minimum frequency for detection (int)
            included_frames - window size for word detection (int)
            key_frames - parameters controlling key-word detection (tuple)
            predictions - optional precomputed predictions (list)

        Returns:
            list:
                Detected (phrase, timestamp) pairs
        """
        if predictions is None:
            predictions = self.get_predictions(model)
        included_frames += 1
        calls = []

        key_frames, key_frames_2 = key_frames
        
        for i, item in enumerate(predictions[key_frames:]):

            key_check = predictions[i:key_frames+i]
            unique_k, counts_k = np.unique(key_check, return_counts=True)

            if (8 in unique_k and counts_k[np.where(unique_k == 8)]>=key_frames_2) or \
                (9 in unique_k and counts_k[np.where(unique_k == 9)]>=key_frames_2):
                word_check = list(predictions[i-included_frames:key_frames+i])
                word_check = np.array(list(filter(lambda x: x not in [8,9,10], word_check))) # exclude an / aus / other

                unique, counts = np.unique(word_check, return_counts=True)

                if len(counts) != 0:
                    max_occ = counts.max()
                    if max_occ >= min_freq:
                        word = unique[np.where(counts == max_occ)][0]
                        
                        
                        key_word = unique_k[np.where(counts_k == counts_k.max())][0]
                        
                        if not any(((self.encoder.classes_[word] + ' ' + self.encoder.classes_[key_word]) == call[0]) for call in calls):
                            # meaning check:
                            timestamp = self.get_time_stamp(predictions,included_frames,word,i,key_frames)
                            calls.append((self.encoder.classes_[word] + ' ' + self.encoder.classes_[key_word],timestamp))
        return calls
    
    def get_time_stamp(
            self,
            predictions: list,
            included_frames: int,
            word: int,
            cur_step: int,
            key_frames: int
        ):
        """
        Computes timestamp for a detected word occurrence.

        Args:
            predictions - predicted classes (list)
            included_frames - window size (int)
            word - detected word index (int)
            cur_step - current frame index (int)
            key_frames - key-frame parameter (int)

        Returns:
            float:
                Estimated timestamp in seconds
        """
        cur_step = cur_step-included_frames-key_frames
        count = 0
        for i in predictions[cur_step:]:
            if i == word:
                return (cur_step+count+2)*(self.frame_step/40)
            count += 1
            
        return (cur_step+0.3)*(self.frame_step/40)


############################################################

##############      Execute Python Code       ##############

############################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute model on a single sample - refer to the readme for more infromation on the arguments")
    # ======== 1 Required     ========
    parser.add_argument("--model_name", type=str, required=True, help="Model name to load")

    # ======== 2 General      ========
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--data_path", type=str, default=r"data\scene\5_Lukas_Staubsauger_an_Licht_aus.npy", help="Path to input sample")
    parser.add_argument("--print_thinking", action="store_true", help="Print debug output")

    # ======== 3 Parameters   ========
    parser.add_argument("--included_frames", type=int, default=46, help="Number of frames to include")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum frequency threshold")
    parser.add_argument("--key_frames", type=int, nargs=2, default=(3, 2), help="Key frame configuration")

    args = parser.parse_args()

    # ======== Load data, create sample, Run the model ========
    model_mfcc, model_melspect, classifier, encoder, config = load_models(args.model_name, True, args.device)
    sample = Dataset_frames(args.data_path, encoder, 1)
    predictions = sample.get_predictions_3class(model_mfcc, model_melspect, classifier)

    pred_np = sample.predict_with_key_word_system(
        model=model_melspect,min_freq=2,included_frames=46,
        key_frames=(3, 2), predictions=predictions)

    if args.print_thinking:
        sample.eval_print(predictions=predictions)

    print(f"\nThe following commands were detected:\n\n {pred_np} \n\n")
