import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import os

from dataset import get_dataloaders
from model import CustomPetNet

# coursework limits - gotta stick to 30 epochs
MAX_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 5e-3 # pushing the LR pretty high for onecycle
WEIGHT_DECAY = 1e-4

def train():
    # figure out if we're on gpu or stuck with cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # fixing the seed so the markers don't penalize me for random results
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # grab the data
    print("Loading data...")
    train_loader, val_loader, test_loader, num_train, num_val, num_test = get_dataloaders(batch_size=BATCH_SIZE)
    print(f"Dataset loaded. Train: {num_train}, Val: {num_val}, Test: {num_test}")

    # setup the model and training gear
    model = CustomPetNet(num_classes=37).to(device)
    
    # standard cross entropy, nothing fancy
    criterion = nn.CrossEntropyLoss()
    
    # adamw just seems to work better here
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # using onecyclelr so it actually converges before the 30 epoch cutoff
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        epochs=MAX_EPOCHS, 
        steps_per_epoch=steps_per_epoch
    )

    best_val_acc = 0.0
    start_time = time.time()

    print(f"\nStarting training for {MAX_EPOCHS} epochs...")
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # main training loop
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{MAX_EPOCHS}]", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # clear out old grads
            optimizer.zero_grad()

            # push data through
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backprop
            loss.backward()
            
            # clipping gradients because the high LR makes it unstable sometimes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # don't forget to step the scheduler every batch for onecycle
            scheduler.step()

            # keep track of how this is going
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # update the tqdm bar
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct_train / total_train

        # check how it does on the validation set
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct_val / total_val

        print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # if this epoch was the best so far, save the weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"--> Saving new best model with Val Acc: {val_acc:.2f}%")
            torch.save(model.state_dict(), 'best_model.pth')

    # done training
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # need to get the final train acc for the submission form
    print("\nCalculating final training accuracy for submission...")
    # reload the best weights just to be sure
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # evaluate on training images WITHOUT augmentations, as per coursework guidance
    # random crops/flips can affect the score, so we use clean transforms here
    # we replicate the exact same seed-42 split used during training to get the right indices
    clean_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    clean_full = datasets.OxfordIIITPet(
        root='./data', split='trainval', target_types='category',
        download=False, transform=clean_transforms
    )
    val_size_eval = int(0.1 * len(clean_full))
    train_size_eval = len(clean_full) - val_size_eval
    gen_eval = torch.Generator().manual_seed(42)
    train_clean_subset, _ = torch.utils.data.random_split(
        clean_full, [train_size_eval, val_size_eval], generator=gen_eval
    )
    clean_train_loader = torch.utils.data.DataLoader(
        train_clean_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    correct_train_final = 0
    total_train_final = 0
    with torch.no_grad():
        for images, labels in clean_train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train_final += labels.size(0)
            correct_train_final += (predicted == labels).sum().item()

    final_train_acc = 100 * correct_train_final / total_train_final
    print(f"Final Train Accuracy: {final_train_acc:.2f}%")

    # dump the final stats to a file so I don't lose them
    with open("training_results.txt", "w") as f:
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Final Training Accuracy: {final_train_acc:.2f}%\n")

if __name__ == '__main__':
    train()
