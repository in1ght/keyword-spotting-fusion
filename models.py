# Imports
import warnings
import os
import torch
import pickle
import torch
import torch.nn as nn

# Ignore warnings
warnings.filterwarnings('ignore')

class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network for feature extraction and classification.

    Processes mel-spectrograms or MFCCs, maps features to class predictions.
    Conv2D -> BatchNorm (optional) -> Activation blocks.

    Args:
        input_channels - input channels number (int)
        hidden_channels - convolutional layers sizes (list of int)
        num_classes - output classes number (int)
        use_batch_normalization - if True, applies batch normalization (bool, default=True)
        kernel_size - convolution kernel size (int, default=3)
        activation_function - activation function applied after conv layer (nn.Module)

    Returns:
        torch.Tensor:
            class logits of shape (batch_size, num_classes)
    """
    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            num_classes: int,
            use_batch_normalization: bool = True,
            kernel_size: int = 3,
            activation_function: nn.Module = nn.ReLU()
        ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_hidden_layers = len(hidden_channels)
        self.use_batch_normalization = use_batch_normalization
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.activation_function = activation_function

        self.conv_layers = nn.ModuleList()
        
        self.conv_layers.append(nn.Conv2d(
            input_channels,
            hidden_channels[0],
            3,
            padding="same",
            padding_mode="zeros"
        ))

        for num in range(1, self.num_hidden_layers):
            self.conv_layers.append(nn.Conv2d(
                hidden_channels[num-1],
                hidden_channels[num],
                kernel_size,
                padding="same",
                padding_mode="zeros"
            ))
        if self.use_batch_normalization:
            self.batch_norm_layers = nn.ModuleList()
            for i in range(self.num_hidden_layers):
                self.batch_norm_layers.append(nn.BatchNorm2d(hidden_channels[i]))
        self.output_layer = nn.Conv2d(hidden_channels[-1], self.num_classes, kernel_size=1, stride=1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4884, 512), # 2816  if melspect 4884 if all
            nn.ReLU(),
            nn.Linear(512, 11)
        )
    
    def forward(self, input_images: torch.Tensor) -> torch.Tensor:

        for i in range(self.num_hidden_layers):
            input_images = self.conv_layers[i](input_images)

            if self.use_batch_normalization:
                input_images = self.batch_norm_layers[i](input_images)
            input_images = self.activation_function(input_images)
        input_images = self.output_layer(input_images)
        input_images = self.fc(input_images)

        return input_images
    

# class FNN_Simple(nn.Module):
#     """
#     Feed-forward network for final classification.

#     Args:
#         inter_dim - hidden layer size (int)
#         num_classes - number of output classes (int)

#     Returns:
#         torch.Tensor:
#             class logits of shape (batch_size, num_classes)
#     """
#     def __init__(self, inter_dim, num_classes):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.LazyLinear(inter_dim), # 2816  if melspect 4884 if all
#             nn.ReLU(),
#             nn.Linear(inter_dim, num_classes))
    
#     def forward(self, x):
#         return self.fc(x)

class FNN_Simple(nn.Module):
    """
    Feed-forward network for final classification.

    Args:
        input_dim - additional input feature size (int)
        inter_dim - hidden layer size (int)
        num_classes - number of output classes (int)

    Returns:
        torch.Tensor:
            class logits of shape (batch_size, num_classes)
    """
    def __init__(self, input_dim, inter_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear((4884+input_dim), inter_dim), # 2816  if melspect 4884 if all
            nn.ReLU(),
            nn.Linear(inter_dim, num_classes))
    
    def forward(self, x):
        return self.fc(x)

    

def construct_resnet_x(
        repo_or_dir: str = 'NVIDIA/DeepLearningExamples:torchhub',
        name: str = 'nvidia_resneXt',
        inter_dim: int = 512,
        out_dim: int = 11
    ):
    """
    Loads and adapts a ResNeXt model for single-channel input and custom classification.

    Args:
        repo_or_dir - Torch Hub repository (str)
        name - model name to load (str)
        inter_dim - hidden dimension for the classifier (int)
        out_dim - out dimension for the classifier (int)

    Returns:
        nn.Module:
            Modified ResNeXt model ready for training
    """
    model = torch.hub.load(repo_or_dir, name)

    n_inputs = model.fc.in_features
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias = False)
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, inter_dim),
        nn.ReLU(),
        nn.Linear(inter_dim, out_dim)
    )
    for param in model.parameters():
        param.requires_grad = True

    return model, n_inputs



def load_models(model_name: str, mode_fusion: bool = True, device="cuda"):
    """
    Loads trained: mfcc model, mel-spectrogram model, and classifier and LabelEncoder.

    Args:
        model_name - saved model directory name (str)
        device - models device ("cuda" or "cpu", default="cuda")

    Returns:
        tuple:
            model_mfcc (torch.nn.Module),
            model_melspect (torch.nn.Module),
            classifier (torch.nn.Module),
            encoder (object),
            config (dict)
    """
    base_path = os.path.join("models", model_name)
    config = torch.load(os.path.join(base_path, "config.pt"))

    # Load Models
    model_melspect, n_inputs = construct_resnet_x()
    model_mfcc = SimpleCNN(
        1,config["cnn_hidden_channels"],
        1,True,config["cnn_kernel_size"])
    classifier = FNN_Simple(
        n_inputs,config["fnn_hidden_dim"],
        len(config["useful_words"]))

    if mode_fusion:
        # Restore Architecture (for 3-class classifier)
        model_mfcc.fc = torch.nn.Flatten()
        model_melspect.fc = torch.nn.Identity()

    # Load Weights
    model_mfcc.load_state_dict(torch.load(os.path.join(base_path, "model_mfcc.pt"), map_location=device))
    model_melspect.load_state_dict(torch.load(os.path.join(base_path, "model_melspect.pt"), map_location=device))
    classifier.load_state_dict(torch.load(os.path.join(base_path, "classifier.pt"), map_location=device))

    # Load Encoder
    with open(os.path.join(base_path, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    # Other
    model_mfcc.to(device)
    model_melspect.to(device)
    classifier.to(device)

    model_mfcc.eval()
    model_melspect.eval()
    classifier.eval()

    return model_mfcc, model_melspect, classifier, encoder, config


def save_models_encoder(
        model_mfcc: torch.nn.Module, 
        model_melspect: torch.nn.Module, 
        classifier: torch.nn.Module, 
        encoder: object,
        args
    ):
    """
    Saves trained: mfcc model, mel-spectrogram model, and classifier and LabelEncoder.

    Args:
        model_mfcc - trained mfcc model - Custom CNN  (nn.Module)
        model_melspect - trained mel-spectrogram model - ResNetX (nn.Module)
        classifier - trained classifier combining both models (nn.Module)
        encoder - label encoder (object) 
        args - configuration - argparser given
    """
    base_path = os.path.join("models", args.model_name)
    os.makedirs(base_path, exist_ok=True)

    torch.save(model_mfcc.state_dict(), os.path.join(base_path, "model_mfcc.pt"))
    torch.save(model_melspect.state_dict(), os.path.join(base_path, "model_melspect.pt"))
    torch.save(classifier.state_dict(), os.path.join(base_path, "classifier.pt"))

    # config for reproducibility
    torch.save(vars(args), os.path.join(base_path, "config.pt"))

    with open(os.path.join(base_path, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    print(f"Models (and the encoder) saved to {base_path}")
