import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Hyper Parameters
DEVICE = torch.device('cuda')
EPOCHS = 50
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 64
MODEL_VERSION = 2
TRAINED_VERSION = ""
from Models.model_v1.build_model import Model
print(torch.cuda.get_device_name(DEVICE))

# Load Data
file_train = np.load(".\\Data\\file_train.npy")
target_train = np.load(".\\Data\\target_train.npy")
file_val = np.load(".\\Data\\file_validation.npy")
target_val = np.load(".\\Data\\target_validation.npy")
file_test = np.load(".\\Data\\file_test.npy")
target_test = np.load(".\\Data\\target_test.npy")

pipeline = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5961, 0.4564, 0.3906], std=[0.2178, 0.1938, 0.1845])
])

def default_loader(path):
    img = Image.open(path)
    img = img.resize((128, 128))
    img = pipeline(img)
    return img

class DataSet(Dataset):
    def __init__(self, imgs, target, loader=default_loader):
        self.imgs = imgs
        self.target = target
        self.loader = loader
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        file_name = self.imgs[index]
        img = self.loader(file_name)
        target_age = torch.tensor([self.target[0][index]])
        target_gender = torch.tensor([self.target[1][index]], dtype=torch.float32)
        target_race = torch.tensor(self.target[2][index], dtype=torch.int64)
        return img, target_age, target_gender, target_race
    
train_data = DataSet(file_train, target_train)
val_data = DataSet(file_val, target_val)
test_data = DataSet(file_test, target_test)
train_loader = DataLoader(train_data, TRAIN_BATCH_SIZE, True)
val_loader = DataLoader(val_data, VAL_BATCH_SIZE, True)
test_loader = DataLoader(test_data, 1, True)

# Train Function
def train_model(model, device, train_loader, val_loader, optimizer, history):
    #Training
    model.train()
    correct_gender = 0.
    correct_race = 0.
    for data, target_age, target_gender, target_race in tqdm(train_loader, desc="Training"):
        data, target_age, target_gender, target_race = data.to(device), target_age.to(device), target_gender.to(device), target_race.to(device)
        target_race_onehot = F.one_hot(target_race, 5)
        target_race_onehot = target_race_onehot.float()

        optimizer.zero_grad()

        age_pred, gender_pred, race_pred = model(data)
        
        loss1 = torch.sqrt(F.mse_loss(age_pred.float(), target_age.float()))
        loss2 = F.binary_cross_entropy(gender_pred, target_gender)
        loss3 = F.cross_entropy(race_pred, target_race_onehot)
        Loss = loss1.float() + loss2.float() + loss3.float()
        Loss.float().backward()

        threshold = torch.tensor([0.5]).to(device)
        gender_pred = (gender_pred>threshold).float()
        correct_gender += gender_pred.eq(target_gender.view_as(gender_pred)).sum().item()
        race_pred = race_pred.argmax(dim=1)
        correct_race += race_pred.eq(target_race.view_as(race_pred)).sum().item()

        optimizer.step()

    acc_gender = 100 * correct_gender / len(train_loader.dataset)
    acc_race = 100 * correct_race / len(train_loader.dataset)
    print("TotalLoss:{:.4f}  AgeLoss:{:.4f}  GenderAcc:{:.2f}%  RaceAcc:{:.2f}%".format(Loss.item(), loss1.item(), acc_gender, acc_race))
    history['train_total_loss'].append(Loss.item())
    history['train_age_loss'].append(loss1.item())
    history['train_gender_loss'].append(loss2.item())
    history['train_race_loss'].append(loss3.item())
    history['train_gender_acc'].append(acc_gender)
    history['train_race_acc'].append(acc_race)

    #Validation
    model.eval()
    correct_gender = 0.
    correct_race = 0.
    with torch.no_grad():
        for data, target_age, target_gender, target_race in tqdm(val_loader, desc="Validation"):
            data, target_age, target_gender, target_race = data.to(device), target_age.to(device), target_gender.to(device), target_race.to(device)
            age_pred, gender_pred, race_pred = model(data)
            target_race_onehot = F.one_hot(target_race, 5)
            target_race_onehot = target_race_onehot.float()
            loss1 = torch.sqrt(F.mse_loss(age_pred.float(), target_age.float()))
            loss2 = F.binary_cross_entropy(gender_pred, target_gender)
            loss3 = F.cross_entropy(race_pred, target_race_onehot)
            Loss = loss1.float() + loss2.float() + loss3.float()
            threshold = torch.tensor([0.5]).to(device)
            gender_pred = (gender_pred>threshold).float()
            correct_gender += gender_pred.eq(target_gender.view_as(gender_pred)).sum().item()
            race_pred = race_pred.argmax(dim=1)
            correct_race += race_pred.eq(target_race.view_as(race_pred)).sum().item()
    acc_gender = 100 * correct_gender / len(val_loader.dataset)
    acc_race = 100 * correct_race / len(val_loader.dataset)
    print("TotalLoss:{:.4f}  AgeLoss:{:.4f}  GenderAcc:{:.2f}%  RaceAcc:{:.2f}%".format(Loss.item(), loss1.item(), acc_gender, acc_race))
    history['val_total_loss'].append(Loss.item())
    history['val_age_loss'].append(loss1.item())
    history['val_gender_loss'].append(loss2.item())
    history['val_race_loss'].append(loss3.item())
    history['val_gender_acc'].append(acc_gender)
    history['val_race_acc'].append(acc_race)

# Test Function
def test_model(model, device, test_loader):
    model.eval()
    correct_gender = 0.
    correct_race = 0.
    with torch.no_grad():
        for data, target_age, target_gender, target_race in tqdm(test_loader, desc="Testing"):
            data, target_age, target_gender, target_race = data.to(device), target_age.to(device), target_gender.to(device), target_race.to(device)
            age_pred, gender_pred, race_pred = model(data)
            loss1 = torch.sqrt(F.mse_loss(age_pred.float(), target_age.float()))
            threshold = torch.tensor([0.5]).to(device)
            gender_pred = (gender_pred>threshold).float()
            correct_gender += gender_pred.eq(target_gender.view_as(gender_pred)).sum().item()
            race_pred = race_pred.argmax(dim=1)
            correct_race += race_pred.eq(target_race.view_as(race_pred)).sum().item()
    acc_gender = 100 * correct_gender / len(test_loader.dataset)
    acc_race = 100 * correct_race / len(test_loader.dataset)
    print("AgeLoss:{:.4f}  GenderAcc:{:.2f}%  RaceAcc:{:.2f}%".format(loss1.item(), acc_gender, acc_race))



model = torch.load(f".\\Models\\model_v{MODEL_VERSION}\\model_v{MODEL_VERSION}.pt").to(DEVICE)
optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

history = {'train_total_loss': [], 'train_age_loss': [], 'train_gender_loss': [],
           'train_race_loss': [], 'train_gender_acc': [], 'train_race_acc': [],
           'val_total_loss': [], 'val_age_loss': [], 'val_gender_loss': [],
           'val_race_loss': [], 'val_gender_acc': [], 'val_race_acc': [],}

for epoch in range(1, EPOCHS+1):
    print(f"=====Epoch[{epoch}/{EPOCHS}]=====")
    train_model(model, DEVICE, train_loader, val_loader, optimizer, history)
    scheduler.step(sum(history['train_total_loss'])/epoch)
test_model(model, DEVICE, test_loader)

torch.save(model, f".\\Models\\model_v{MODEL_VERSION}\\model_v{MODEL_VERSION}_trained{TRAINED_VERSION}.pt")

x = list(range(1, EPOCHS+1))
x_tick = [1] + [i for i in range(10, EPOCHS+1, 10)]
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0][0].plot(x, history['train_total_loss'], label='train')
ax[0][0].plot(x, history['val_total_loss'], label='val')
ax[0][0].set_title("Total Loss")
ax[0][0].set_xlabel("epoch")
ax[0][0].set_ylabel("loss")
ax[0][0].set_xticks(x_tick)
ax[0][0].legend()

ax[0][1].plot(x, history['train_age_loss'], label='train')
ax[0][1].plot(x, history['val_age_loss'], label='val')
ax[0][1].set_title("Age Loss")
ax[0][1].set_xlabel("epoch")
ax[0][1].set_ylabel("loss")
ax[0][1].set_xticks(x_tick)
ax[0][1].legend()

ax[0][2].plot(x, history['train_gender_acc'], label='train')
ax[0][2].plot(x, history['val_gender_acc'], label='val')
ax[0][2].set_title("Gender Accuracy")
ax[0][2].set_xlabel("epoch")
ax[0][2].set_ylabel("accuracy(%)")
ax[0][2].set_xticks(x_tick)
ax[0][2].legend()

ax[1][0].plot(x, history['train_gender_loss'], label='train')
ax[1][0].plot(x, history['val_gender_loss'], label='val')
ax[1][0].set_title("Gender Loss")
ax[1][0].set_xlabel("epoch")
ax[1][0].set_ylabel("loss")
ax[1][0].set_xticks(x_tick)
ax[1][0].legend()

ax[1][1].plot(x, history['train_race_loss'], label='train')
ax[1][1].plot(x, history['val_race_loss'], label='val')
ax[1][1].set_title("Race Loss")
ax[1][1].set_xlabel("epoch")
ax[1][1].set_ylabel("loss")
ax[1][1].set_xticks(x_tick)
ax[1][1].legend()

ax[1][2].plot(x, history['train_race_acc'], label='train')
ax[1][2].plot(x, history['val_race_acc'], label='val')
ax[1][2].set_title("Race Accuracy")
ax[1][2].set_xlabel("epoch")
ax[1][2].set_ylabel("accuracy(%)")
ax[1][2].set_xticks(x_tick)
ax[1][2].legend()

fig.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.25)

plt.savefig(f".\\Models\\model_v{MODEL_VERSION}\\train_info.jpg")
plt.show()