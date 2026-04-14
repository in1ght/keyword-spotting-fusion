# Imports
import os
import warnings
import torch
import argparse
import json

from tqdm import tqdm
import pandas as pd

# Imports from other local python files
from swresnetx import Dataset_frames
from models import load_models

# Ignore warnings
warnings.filterwarnings('ignore')

def display_evaluation(
        configs: list,
        TP_: list, 
        FP_: list, 
        FN_: list
    ):
    """
    Displays recall and precision for configurations.

    Note: No returns, just a print.

    Computes and prints recall and precision per configuration -> then rints it.

    Args:
        TP_ - True Positives per configuration (list of int)
        FP_ - False Positives per configuration (list of int)
        FN_ - False Negatives per configuration (list of int)
    """
    for i, (TP_i, FP_i, FN_i) in enumerate(zip(TP_,FP_,FN_)):
        print(f"{configs[i][0]} ({configs[i][0]}, {configs[i][1][1]}): Recall: {TP_i/(TP_i+FN_i):.3f} | Precision: {TP_i/(TP_i+FP_i):.3f} | TP: {TP_i}, FP: {FP_i}")

def save_evaluation(
        result_strs: list,
        configs: list,
        included_frames:int, 
        frame_step: int, 
        model_name: str
    ):
    """
    Saves prediction results to CSV files.

    Note: No returns, just a file.write (save).

    Args:
        result_strs - CSV-formatted strings (list of str)
        configs - configurations of view [min_freq, key_frames] (list of tuples)
        included_frames - frames number to include in window (int, default=28)
        frame_step - a step the model does during the online execution (int, default = 2)
        model_name - model name (str)
    """
    base_path = os.path.join("models", model_name, "results")
    os.makedirs(base_path, exist_ok=True)
    for i in range(0,len(result_strs)):
        name = (
            f"freq_{configs[i][0]}"
            f"_num_{configs[i][1][0]}"
            f"_from_{configs[i][1][1]}"
            f"_frames_{included_frames}"
            f"_step_{frame_step}.csv"
        )
        path = os.path.join(base_path, name)
        with open(path, "w") as f:
            f.write(result_strs[i])

def perform_evaluation(
        model_name: str, 
        configs: list, 
        path: str, 
        model_mfcc: torch.nn.Module,
        model_melspect: torch.nn.Module,
        classifier: torch.nn.Module,
        encoder: object,
        dv_scenes: pd.DataFrame,
        dv_scenes_annotations: pd.DataFrame,
        included_frames: int = 28, 
        frame_step: int = 2,
        display: bool = True, 
        save: bool = True
    ):
    """
    Runs evaluation of keyword detection models across given configurations.

    Purpose:
        - Generates predictions using MFCC and mel-spectrogram models.
        - Applies keyword detection logic with different parameter settings.
        - Compares predictions against ground truth annotations.
        - Computes TP, FP, FN and optionally displays and saves results.

    Args:
        model_name - model name - used for saving (str)
        configs - configurations of view [min_freq, key_frames] (list of tuples)
        path - path to input data files (str)
        model_mfcc - model for MFCC-based predictions (nn.Module)
        model_melspect - model for mel-spectrogram predictions (nn.Module)
        classifier - final classifier (nn.Module)
        encoder - label encoder (object)
        dv_scenes - file metadata (pd.DataFrame)
        dv_scenes_annotations - ground truth annotations (pd.DataFrame)
        included_frames - frames number to include in window (int, default=28)
        frame_step - a step the model does during the online execution (int, default = 2)
        display - if True, prints evaluation results (bool, default=True)
        save - if True, saves results to CSV (bool, default=True)

    Returns:
        Tuple[list, list, list, list]:
            - TP_, FP_, FN_ counts per configuration
            - result strings for each configuration
    """
    TP_,FP_,FN_ = [0] * len(configs),[0] * len(configs),[0] * len(configs)

    result_strs = ['filename,command,timestamp\n' for _ in range(len(configs))]

    for _, file in tqdm(dv_scenes.iterrows()):
        current_path = os.path.join(path, (file.filename + '.npy'))
        current_sample = Dataset_frames(current_path, encoder, frame_step)

        predictions = current_sample.get_predictions_3class(model_mfcc, model_melspect, classifier)

        # Run all configurations
        results_models = []
        for min_freq, key_frames in configs:
            results = current_sample.predict_with_key_word_system(
                model=model_melspect,
                min_freq=min_freq,
                included_frames=included_frames,
                key_frames=key_frames,
                predictions=predictions
            )
            results_models.append(results)

        # Ground truth
        annotations = dv_scenes_annotations[dv_scenes_annotations['filename'] == file.filename]['command']
        true_commands = set(annotations.to_list())

        for i, result_time in enumerate(results_models):
            predicted_commands = set()

            for cmd, timestamp in result_time:
                predicted_commands.add(cmd)

                result_strs[i] += (
                    f"{os.path.basename(current_path)[:-4]},"
                    f"{cmd},{round(timestamp + 0.6, 5)}\n"
                )

                if cmd in true_commands:
                    TP_[i] += 1
                else:
                    FP_[i] += 1

            # FN calculation
            FN_[i] += len(true_commands - predicted_commands)
    
    if display:
        display_evaluation(configs, TP_, FP_, FN_)

    if save:
        save_evaluation(result_strs, configs, included_frames, frame_step, model_name)

    return TP_, FP_, FN_, result_strs


############################################################

##############      Execute Python Code       ##############

############################################################


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Perform evaluation over multiple configurations")
    
    # ======== 1 Required       ========
    parser.add_argument("--model_name", type=str, required=True, help="Model name to load")

    # ======== 2 General        ========
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dv_scenes", type=str, default=os.path.join(base, "data", "development_scenes.csv"), help="CSV with evaluation scene metadata")
    parser.add_argument("--dv_scenes_annotations", type=str, default=os.path.join(base, "data", "development_scene_annotations.csv"), help="CSV with evaluation scene annotations")
    parser.add_argument("--dv_folder", type=str, default=os.path.join(base, "data", "development_scenes"), help="Folder with evaluation scene files")
    parser.add_argument("--json_path", type=str, default=os.path.join(base, "models", "configurations.json"), help="Path to configuration JSON file")

    # ======== 3 Checkboxes     ========
    parser.add_argument("--display", action="store_true", help="Enable evaluation display output")
    parser.add_argument("--save", action="store_true", help="Enable saving evaluation results") 

    args = parser.parse_args()

    # ======== Load data, load models ========
    model_mfcc, model_melspect, classifier, encoder, config = load_models(args.model_name, True, args.device)
    
    dv_scenes = pd.read_csv(args.dv_scenes)
    dv_scenes_annotations = pd.read_csv(args.dv_scenes_annotations)

    # ======== Load configurations from the JSON ========
    json_path = args.json_path

    if not os.path.isfile(json_path):
        raise ValueError(f"File {json_path} not found")
    
    with open(json_path, "r", encoding="utf-8") as f:
        conf_json = json.load(f)  

    required_keys = ["configs", "included_frames", "frame_step"]
    if not all(key in conf_json for key in required_keys):
        raise ValueError("Must contain all hyperparameters, these inslude: configs, included_frames, frame_step, some are missing.")
    
    configs = [tuple([a, tuple(b)]) for a, b in conf_json["configs"]]

    # ======== Perform evaluation ========
    TP_, FP_, FN_, result_strs = perform_evaluation(
        args.model_name,
        configs, 
        args.dv_folder, 
        model_mfcc, 
        model_melspect, 
        classifier, 
        encoder, 
        dv_scenes, 
        dv_scenes_annotations,
        conf_json["included_frames"],
        conf_json["frame_step"],
        args.display,
        args.save
    )