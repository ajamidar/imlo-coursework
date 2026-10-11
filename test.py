import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import CustomPetNet

def test():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # grab the test set
    print("Loading test dataset...")
    # ignoring the train/val stuff since we only care about testing here
    _, _, test_loader, _, _, num_test = get_dataloaders(batch_size=64)
    print(f"Test dataset loaded. Size: {num_test}")

    # spin up an empty model
    model = CustomPetNet(num_classes=37).to(device)
    
    # pull in the weights we just trained
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("Successfully loaded 'best_model.pth'")
    except FileNotFoundError:
        print("Error: 'best_model.pth' not found. Please run train.py first.")
        return

    model.eval()
    
    correct_test = 0
    total_test = 0

    print("Running inference on test dataset...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_acc = 100 * correct_test / total_test
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    with open("testing_results.txt", "w") as f:
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")

if __name__ == '__main__':
    test()
