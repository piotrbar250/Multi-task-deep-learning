# %% Imports, seed
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F

torch.manual_seed(1)

# %% Global config / mappings

# shape indices:
# 0: square, 1: circle, 2: up, 3: right, 4: down, 5: left
PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6)]  # 15 unordered pairs
PAIR_TO_IDX = {p: k for k, p in enumerate(PAIRS)}
N_CONFIGS = len(PAIRS) * 9  # 135

def class_id_to_pair_and_split(class_id: int):
    pair_idx = class_id // 9
    split_idx = class_id % 9  # 0..8 -> counts 1..9
    ca = split_idx + 1
    cb = 10 - ca
    i, j = PAIRS[pair_idx]
    return (i, j), (ca, cb)

def class_id_to_pair(class_id: int):
    pair_idx = class_id // 9
    return PAIRS[pair_idx]

# %% Dataset + augmentation

class GSN(Dataset):
    def __init__(self, root, transform=None, transform_relabel=None):
        self.data_dir = os.path.join(root, "data")
        self.transform = transform          # image-only transforms
        self.transform_relabel = transform_relabel  # image + label transforms

        df = pd.read_csv(os.path.join(self.data_dir, "labels.csv"))
        self.names = df["name"].tolist()
        cols = ["squares", "circles", "up", "right", "down", "left"]
        self.labels = torch.tensor(df[cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        name = self.names[index]
        img_path = os.path.join(self.data_dir, name)

        img = Image.open(img_path).convert("L")
        img = transforms.ToTensor()(img)  # (1,28,28)
        
        if self.transform:
            img = self.transform(img)

        cnt = self.labels[index]

        if self.transform_relabel:
            img, cnt = self.transform_relabel(img, cnt)
        
        cls = self.counts_to_class_id(cnt)

        return img, cls, cnt
    
    def counts_to_class_id(self, counts):
        if isinstance(counts, torch.Tensor):
            c = counts.detach().cpu().tolist()
        else:
            c = list(counts)

        nz = [i for i, v in enumerate(c) if v > 0]
        if len(nz) != 2:
            raise ValueError(f"Expected exactly 2 nonzero counts, got {len(nz)}: {c}")
        if sum(c) != 10:
            raise ValueError(f"Expected counts to sum to 10, got {sum(c)}: {c}")

        a, b = sorted(nz)
        ca = int(c[a])
        pair_index = PAIR_TO_IDX[(a, b)]

        class_id = pair_index * 9 + (ca - 1)
        return class_id


class Augmentation:
    def __init__(self, p_hflip=0.5, p_vflip=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, img, cnt):
        cnt = cnt.clone()

        # random 0/90/180/270 deg clockwise rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            img = torch.rot90(img, k=-k, dims=[1, 2])  # clockwise
            dirs = cnt[2:6]           # [up, right, down, left]
            dirs = torch.roll(dirs, shifts=k)  # cycle orientations
            cnt[2:6] = dirs

        # random horizontal flip: swap right <-> left
        if torch.rand(1).item() < self.p_hflip:
            img = torch.flip(img, dims=[2])    # flip width
            cnt[[3, 5]] = cnt[[5, 3]]

        # random vertical flip: swap up <-> down
        if torch.rand(1).item() < self.p_vflip:
            img = torch.flip(img, dims=[1])    # flip height
            cnt[[2, 4]] = cnt[[4, 2]]

        return img, cnt

# %% Model

class NeuralNetwork(nn.Module):
    def __init__(self, cls_hidden=256, dropout=0.3):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 28 * 28, 256), nn.ReLU()
        )

        self.head_cls = nn.Sequential(
            nn.Linear(256, cls_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(cls_hidden, N_CONFIGS),
            nn.LogSoftmax(dim=1)
        )

        self.head_cnt = nn.Linear(256, 6)
    
    def forward(self, x):
        feats = self.backbone(x)
        cls = self.head_cls(feats)
        cnt = self.head_cnt(feats)
        return cls, cnt

# %% Train / eval epochs with mode

def train_epoch(
    net: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    log_interval: int,
    lambda_cnt: float,
    mode: str = "multitask",
    verbose: bool = False,
):
    net.train()
    total_loss = total_cls_loss = total_cnt_loss = 0.0
    n_total = 0
    
    for batch_idx, (img, cls_target, cnt_target) in enumerate(train_loader):
        img = img.to(device)
        cls_target = cls_target.long().to(device)
        cnt_target = cnt_target.to(device)

        optimizer.zero_grad()

        cls_pred, cnt_pred = net(img)
        
        cls_loss = F.nll_loss(cls_pred, cls_target)
        cnt_loss = F.smooth_l1_loss(cnt_pred, cnt_target)

        if mode == "cls_only":
            loss = cls_loss
        elif mode == "reg_only":
            loss = lambda_cnt * cnt_loss
        elif mode == "multitask":
            loss = cls_loss + lambda_cnt * cnt_loss
        else:
            raise ValueError(f"Unknown mode: {mode}")

        loss.backward()
        optimizer.step()

        B = len(img)
        total_loss += loss.item() * B
        total_cls_loss += cls_loss.item() * B
        total_cnt_loss += cnt_loss.item() * B
        n_total += B

        if verbose and batch_idx % log_interval == 0:
            done = batch_idx * B
            total = len(train_loader.dataset)
            print(
                f"Train Epoch: {epoch} [{done}/{total} ({done / total:.0%})] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = total_loss / n_total
    epoch_cls_loss = total_cls_loss / n_total
    epoch_cnt_loss = total_cnt_loss / n_total
    return epoch_loss, epoch_cls_loss, epoch_cnt_loss


def eval_epoch(
    net: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    epoch: int,
    lambda_cnt: float,
    mode: str = "multitask",
    verbose: bool = False,
):
    net.eval()
    total_loss = total_cls_loss = total_cnt_loss = 0.0
    n_total = 0
    correct = 0

    with torch.no_grad():
        for img, cls_target, cnt_target in test_loader:
            img = img.to(device)
            cls_target = cls_target.long().to(device)
            cnt_target = cnt_target.to(device)

            cls_pred, cnt_pred = net(img)
            cls_loss = F.nll_loss(cls_pred, cls_target)
            cnt_loss = F.smooth_l1_loss(cnt_pred, cnt_target)

            if mode == "cls_only":
                loss = cls_loss
            elif mode == "reg_only":
                loss = lambda_cnt * cnt_loss
            elif mode == "multitask":
                loss = cls_loss + lambda_cnt * cnt_loss
            else:
                raise ValueError(f"Unknown mode: {mode}")

            B = len(img)
            total_loss += loss.item() * B
            total_cls_loss += cls_loss.item() * B
            total_cnt_loss += cnt_loss.item() * B
            n_total += B

            pred_cls = cls_pred.argmax(dim=1)
            correct += (pred_cls == cls_target).sum().item()

    epoch_loss = total_loss / n_total
    epoch_cls_loss = total_cls_loss / n_total
    epoch_cnt_loss = total_cnt_loss / n_total
    epoch_acc = correct / n_total

    if verbose:
        print(
            f"Eval Epoch: {epoch} | "
            f"acc: {epoch_acc:.4f} | "
            f"loss: {epoch_loss:.4f} | "
            f"cls_loss: {epoch_cls_loss:.4f} | "
            f"cnt_loss: {epoch_cnt_loss:.4f}"
        )
    return epoch_loss, epoch_cls_loss, epoch_cnt_loss, epoch_acc

# %% Helper: dataloaders

def make_loaders(root=".", batch_size=64, test_batch_size=1000, device=None):
    num_workers = min(8, os.cpu_count() or 2)
    pin = (device is not None and device.type == "cuda")
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_aug = Augmentation()
    train_full = GSN(root=root, transform_relabel=train_aug)
    test_full = GSN(root=root)

    train_dataset = Subset(train_full, range(0, 9000))
    test_dataset = Subset(test_full, range(9000, 10000))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **loader_kwargs)
    return train_loader, test_loader

# %% Training wrapper (for scenarios)

def train_model(
    mode: str,
    lambda_cnt: float,
    epochs: int = 100,
    lr: float = 1e-3,
    dropout: float = 0.3,
    patience: int = 10,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch, "mps", None) and torch.mps.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader, test_loader = make_loaders(
        root=".",
        batch_size=64,
        test_batch_size=1000,
        device=device,
    )

    net = NeuralNetwork(cls_hidden=256, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_cls_loss": [],
        "train_cnt_loss": [],
        "val_loss": [],
        "val_cls_loss": [],
        "val_cnt_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_acc = 0.0
    best_state = None
    bad_epochs = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_cls_loss, train_cnt_loss = train_epoch(
            net, device, train_loader, optimizer,
            epoch, log_interval=10,
            lambda_cnt=lambda_cnt,
            mode=mode,
            verbose=False,
        )

        val_loss, val_cls_loss, val_cnt_loss, val_acc = eval_epoch(
            net, device, test_loader,
            epoch,
            lambda_cnt=lambda_cnt,
            mode=mode,
            verbose=True,
        )

        history["train_loss"].append(train_loss)
        history["train_cls_loss"].append(train_cls_loss)
        history["train_cnt_loss"].append(train_cnt_loss)
        history["val_loss"].append(val_loss)
        history["val_cls_loss"].append(val_cls_loss)
        history["val_cnt_loss"].append(val_cnt_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
            best_epoch = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(
                    f"Early stop at epoch {epoch}. "
                    f"Best val loss: {best_val_loss:.4f}, "
                    f"best acc: {best_acc:.4f}, "
                    f"best epoch: {best_epoch}"
                )
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    return net, history

# %% Example: run a multitask training (you'll later call with other modes / λ)

# multitask: NLL + λ_cnt * SmoothL1
net, history = train_model(
    mode="multitask",
    lambda_cnt=1.0,   # you can try 0.3, 0.5, etc later
    epochs=100,
    lr=1e-3,
    dropout=0.3,
    patience=10,
)
