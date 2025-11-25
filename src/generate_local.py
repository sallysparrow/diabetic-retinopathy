#!/usr/bin/env python
"""
Generate a one-page PDF for a single patient and (optionally) merge two files
into explanation_local.pdf.

Examples
========
# Build two PDFs
python src/generate_local.py --id 9859e2a6cc24
python src/generate_local.py --id e8ddfc9709ce

# Merge into assignment deliverable
python src/generate_local.py --merge 9859e2a6cc24.pdf e8ddfc9709ce.pdf
"""
import argparse, matplotlib.pyplot as plt, numpy as np, skimage.segmentation as seg
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.pagesizes import letter
from PyPDF2 import PdfMerger
import tensorflow as tf, pandas as pd, os, shap, textstat, io

IMG_DIR = "data/test_images"
MODEL   = "model.h5"
masker  = shap.maskers.Image("inpaint_telea", (320, 320, 3))
explainer = shap.Explainer(tf.keras.models.load_model(MODEL, compile=False),
                           masker, output_names=[str(i) for i in range(5)])

def load_tensor(img_id):
    img = plt.imread(f"{IMG_DIR}/{img_id}.png")/255.0
    return np.expand_dims(img,0), img

def make_overlay(img_id):
    X, img = load_tensor(img_id)
    sv = explainer(X, outputs=shap.Explanation.argsort.flip[:1],
                   max_evals=150).values.squeeze().mean(-1)
    segs = seg.slic(img, n_segments=250, compactness=10, sigma=1)
    sv_smooth = np.zeros_like(sv)
    for s in np.unique(segs):
        sv_smooth[segs==s] = sv[segs==s].mean()
    vmax = np.percentile(np.abs(sv_smooth),95)
    heat = plt.cm.seismic(np.clip(sv_smooth/vmax,-1,1))[...,:3]
    blend = 0.55*heat + 0.45*img
    buf = io.BytesIO(); plt.imsave(buf, blend); buf.seek(0)
    return blend, buf

def build_pdf(img_id, out_path):
    blend, buf = make_overlay(img_id)
    # simple result text
    pred = int(np.argmax(explainer.model(load_tensor(img_id)[0]),1))
    risk = ["Very low","Low","Medium","High","Very high"][pred]
    pdf = Canvas(out_path, pagesize=letter)
    w,h = letter
    pdf.setFont("Helvetica-Bold",16); pdf.drawString(40,h-60,
        f"Screening result: {risk} risk")
    pdf.drawImage(buf, 60, h-400, width=430, preserveAspectRatio=True)
    pdf.setFont("Helvetica",11)
    pdf.drawString(60,h-420,"Red shows parts the app thinks may be damaged.")
    pdf.drawString(60,h-440,"Only an eye doctor can confirm. Please book a visit "
                            f"{'within 1 month' if pred>=3 else 'within 12 months'}.")
    pdf.save()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="patient image id (no .png)")
    ap.add_argument("--out", help="output pdf name")
    ap.add_argument("--merge", nargs="+", help="pdfs to merge into explanation_local.pdf")
    args = ap.parse_args()

    if args.id:
        out = args.out or f"{args.id}.pdf"
        build_pdf(args.id, out)
        print("wrote", out)

    if args.merge:
        merger = PdfMerger()
        for f in args.merge: merger.append(f)
        merger.write("explanation_local.pdf"); merger.close()
        print("merged → explanation_local.pdf")

if __name__ == "__main__":
    main()
