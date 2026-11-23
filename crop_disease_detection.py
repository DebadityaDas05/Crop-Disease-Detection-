"""
Crop Disease Detection using Transfer Learning (PyTorch)
SDG 2 - Zero Hunger  |  PCCCS582 Neural Network & Deep Learning

Dataset: PlantVillage (Kaggle: "emmarex/plantdisease" or "abdallahalidev/plantvillage-dataset")
Expected folder layout (ImageFolder):
    data/PlantVillage/<class_name>/*.jpg

Install:
    pip install torch torchvision scikit-learn matplotlib seaborn gradio pillow

Usage:
    python crop_disease_detection.py train   --data data/PlantVillage --epochs 8
    python crop_disease_detection.py predict --image leaf.jpg
    python crop_disease_detection.py demo                      # Gradio web app
"""
import argparse, json, os, time
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH, CLASSES_PATH = "crop_model.pt", "classes.json"
IMG = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# Dataset wrapper to apply split-specific transforms
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, y = self.subset[idx]
        return self.transform(img), y


def build_model(n_classes, pretrained=True):
    m = models.mobilenet_v2(weights="DEFAULT" if pretrained else None)
    for p in m.features.parameters():          # freeze backbone first
        p.requires_grad = False
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, n_classes))
    return m


def run_epoch(model, loader, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def train(args):
    full = datasets.ImageFolder(args.data)
    classes = full.classes
    n = len(full)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    g = torch.Generator().manual_seed(42)
    tr, va, te = random_split(full, [n_tr, n_va, n - n_tr - n_va], generator=g)

    dl = lambda ds, sh: DataLoader(DatasetWrapper(ds, train_tf if sh else eval_tf),
                                   batch_size=args.bs, shuffle=sh, num_workers=args.workers,
                                   pin_memory=(DEVICE == "cuda"))
    tr_l, va_l, te_l = dl(tr, True), dl(va, False), dl(te, False)
    print(f"Classes: {len(classes)} | train {len(tr)} val {len(va)} test {len(te)} | device {DEVICE}")

    model = build_model(len(classes)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    best, hist = 0.0, {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}

    for ep in range(1, args.epochs + 1):
        if ep == args.epochs // 2 + 1:          # fine-tune whole network
            for p in model.features.parameters(): p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            print("Unfroze backbone for fine-tuning")
        t0 = time.time()
        tl, ta = run_epoch(model, tr_l, loss_fn, opt)
        vl, vacc = run_epoch(model, va_l, loss_fn)
        for k, v in zip(hist, (tl, vl, ta, vacc)): hist[k].append(v)
        print(f"Epoch {ep}/{args.epochs}  train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{vacc:.3f}  ({time.time()-t0:.0f}s)")
        if vacc > best:
            best = vacc
            torch.save(model.state_dict(), MODEL_PATH)
    json.dump(classes, open(CLASSES_PATH, "w"))

    # ---- final evaluation on held-out test set ----
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)); model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in te_l:
            ps += model(x.to(DEVICE)).argmax(1).cpu().tolist(); ys += y.tolist()
    print("\nTest report:\n", classification_report(ys, ps, target_names=classes, digits=3))
    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(12, 10)); sns.heatmap(cm, cmap="Blues", cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["tr_loss"], label="train"); ax[0].plot(hist["va_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["tr_acc"], label="train"); ax[1].plot(hist["va_acc"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
    plt.tight_layout(); plt.savefig("training_curves.png", dpi=150)
    print("Saved: crop_model.pt, classes.json, confusion_matrix.png, training_curves.png")


def load_trained():
    classes = json.load(open(CLASSES_PATH))
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval(), classes


def predict_image(img, model=None, classes=None, k=3):
    if model is None: model, classes = load_trained()
    x = eval_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    top = p.topk(min(k, len(classes)))
    return {classes[i]: float(v) for v, i in zip(top.values, top.indices)}


def demo():
    import gradio as gr
    model, classes = load_trained()
    fn = lambda img: predict_image(img, model, classes, k=5)
    gr.Interface(fn, gr.Image(type="pil", label="Upload crop leaf"),
                 gr.Label(num_top_classes=5, label="Predicted disease"),
                 title="Crop Disease Detection (SDG 2 - Zero Hunger)",
                 description="Upload a leaf photo to identify the crop and disease.").launch()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict", "demo"])
    ap.add_argument("--data", default="data/PlantVillage")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--image")
    a = ap.parse_args()
    if a.mode == "train": train(a)
    elif a.mode == "predict":
        for k, v in predict_image(Image.open(a.image)).items(): print(f"{k}: {v:.1%}")
    else: demo()
