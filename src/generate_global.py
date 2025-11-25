#!/usr/bin/env python
"""
Builds explanation_global.pdf — the 6-page handbook for nurses/volunteers.
Usage:
    python src/generate_global.py --model model.h5 --train data/train.csv \
            --test  data/test.csv  --out explanation_global.pdf
"""
import argparse, numpy as np, pandas as pd, tensorflow as tf, io, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

# ─── helpers ──────────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3,3))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    buf = io.BytesIO(); fig.savefig(buf, bbox_inches="tight"); plt.close(fig)
    buf.seek(0); return ImageReader(buf)

def plot_fairness(df, y_pred):
    df = df.copy(); df["pred"] = y_pred
    groups = {"Women":df.sex==0, "Men":df.sex==1,
              "<40":df.age<40, "≥40":df.age>=40}
    fig, ax = plt.subplots(figsize=(4,2))
    bars = []
    for i,(g,mask) in enumerate(groups.items()):
        acc = (df.loc[mask,"diagnosis"]==df.loc[mask,"pred"]).mean()
        ax.bar(i, acc, color="#4c72b0"); ax.text(i, acc+0.02, f"{acc:.2f}", ha='center')
        bars.append(acc)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups.keys())
    ax.set_ylim(0,1); ax.set_ylabel("Accuracy")
    buf = io.BytesIO(); fig.savefig(buf, bbox_inches="tight"); plt.close(fig)
    buf.seek(0); return ImageReader(buf)

# ─── main ─────────────────────────────────────────────────────────────────
def main(args):
    train = pd.read_csv(args.train_csv)
    test  = pd.read_csv(args.test_csv)
    model = tf.keras.models.load_model(args.model, compile=False)

    Xtest = np.stack([plt.imread(f"{args.test_dir}/{i}.png")/255.0
                      for i in test.id_code])
    y_true = test.diagnosis.values
    y_pred = np.argmax(model.predict(Xtest, verbose=0),1)

    # build PDF
    pdf = Canvas(args.out, pagesize=letter)
    w,h = letter

    # page 1 – accuracy
    pdf.setFont("Helvetica-Bold", 14); pdf.drawString(40,h-60,"How often is the app right?")
    img = plot_confusion(y_true,y_pred)
    pdf.drawImage(img, 80, h-350, width=300, preserveAspectRatio=True)
    pdf.drawString(80, h-380, f"Overall accuracy: {(y_true==y_pred).mean():.2%}")
    pdf.showPage()

    # page 2 – confidence bar legend
    pdf.setFont("Helvetica-Bold", 14); pdf.drawString(40,h-60,"Confidence bar (what nurses see)")
    pdf.setFillColor(colors.green); pdf.rect(80,h-120,150,20,fill=1)
    pdf.setFillColor(colors.yellow); pdf.rect(230,h-120,80,20,fill=1)
    pdf.setFillColor(colors.red);   pdf.rect(310,h-120,80,20,fill=1)
    pdf.setFillColor(colors.black)
    pdf.drawString(80,h-145,"Green: very sure   Yellow: unsure   Red: high-risk")
    pdf.drawString(80,h-165,"Gray bar means poor photo → retake.")
    pdf.showPage()

    # page-3/4 SHAP overlays (pre-saved PNGs)
    for fname,title in [(args.sample_high,"High-risk example (red spots)"),
                        (args.sample_low ,"Low-risk example (few red spots)")]:
        pdf.setFont("Helvetica-Bold", 14); pdf.drawString(40,h-60,title)
        pdf.drawImage(fname, 60, h-400, width=430, preserveAspectRatio=True)
        pdf.drawString(60, h-420, "Red = parts app used; blue = less important.")
        pdf.showPage()

    # page 5 – fairness bars
    pdf.setFont("Helvetica-Bold", 14); pdf.drawString(40,h-60,"Fairness check")
    pdf.drawImage(plot_fairness(test,y_pred), 80, h-350, width=300,
                  preserveAspectRatio=True)
    pdf.drawString(80, h-380, "Bar height shows accuracy for each group.")
    pdf.showPage()

    # page 6 – data facts
    pdf.setFont("Helvetica-Bold", 14); pdf.drawString(40,h-60,"Data in brief")
    n = len(train)+len(test)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(60,h-100, f"Total eye images used to train the app: {n}")
    pdf.drawString(60,h-120, "Age range of people in the data: 18–83 years")
    pdf.drawString(60,h-140, "Images came from public hospital datasets")
    pdf.save()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv",  required=True)
    p.add_argument("--test_dir",  default="data/test_images")
    p.add_argument("--sample_high", default="output/9859e2a6cc24_high_explain.png")
    p.add_argument("--sample_low",  default="output/e8ddfc9709ce_low_explain.png")
    p.add_argument("--out",        default="explanation_global.pdf")
    main(p.parse_args())
