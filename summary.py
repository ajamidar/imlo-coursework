import torch
import torch.nn as nn
from model import CustomPetNet

def analyze_network():
    model = CustomPetNet(num_classes=37)
    
    # parsing the model to answer Q1-Q4 for the form
    num_layers = 0
    layer_types = []
    layer_units = []
    activation_functions = []
    
    def traverse_model(module):
        nonlocal num_layers
        for name, child in module.named_children():
            # gotta dig into the sequential containers
            if isinstance(child, nn.Sequential):
                traverse_model(child)
            else:
                if isinstance(child, nn.Conv2d):
                    num_layers += 1
                    layer_types.append('Conv2d')
                    layer_units.append(str(child.out_channels))
                    # I put a relu after every conv layer so hardcoding it here
                    activation_functions.append('ReLU')
                elif isinstance(child, nn.Linear):
                    num_layers += 1
                    layer_types.append('Linear')
                    layer_units.append(str(child.out_features))
                    # the last linear layer just outputs raw logits
                    activation_functions.append('None')
                elif isinstance(child, (nn.MaxPool2d, nn.AdaptiveMaxPool2d)):
                    layer_types.append('MaxPool2d')
                    layer_units.append('None')
                    activation_functions.append('None')
    
    traverse_model(model)
    
    print("\n--- Submission Point Information ---")
    print(f"Q1: How many layers in your network? (Conv & Linear) -> {num_layers}")
    print(f"Q2: Type of each layer -> {','.join(layer_types)}")
    print(f"Q3: Units/kernels in each layer -> {','.join(layer_units)}")
    print(f"Q4: Activation functions -> {','.join(activation_functions)}")
    
    # counting up the parameters for Q5
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Q5: Total weights/biases -> {total_params}")
    
    print(f"Q6: Loss function -> CrossEntropyLoss")
    print(f"Q7: Optimisation algorithm -> AdamW (with OneCycleLR)")
    print(f"Q8: Learning rate -> 5e-3")
    print(f"Q9: Epochs -> 30")
    print(f"Q10: Batch size -> 32")
    
    # hardcoding some of the data stats since they don't change
    print(f"Q11: Augmentations/transforms -> RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize")
    print(f"Q12: Total training images -> 3312") 
    print(f"Q13: Total validation images -> 368")
    
if __name__ == '__main__':
    analyze_network()
