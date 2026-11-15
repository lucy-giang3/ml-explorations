from __future__ import annotations
import argparse, copy, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, BertForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup
from torch.optim import AdamW


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_data(csv_path: str | None, sample_count: int | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if csv_path:
        source = Path(csv_path)
    else:
        import kagglehub
        source = Path(kagglehub.dataset_download("danofer/sarcasm")) / "train-balanced-sarcasm.csv"
    if not source.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {source}")
    cols = ["label", "comment", "parent_comment", "subreddit", "score", "ups", "downs"]
    df = pd.read_csv(source, usecols=lambda c: c in cols)
    required = [c for c in ["label", "comment", "parent_comment"] if c not in df]
    if required:
        raise ValueError(f"Dataset is missing required columns: {required}")
    df = df.dropna(subset=["comment"]).drop_duplicates(subset=["comment"]).reset_index(drop=True)
    df["parent_comment"] = df["parent_comment"].fillna("")
    if sample_count is not None and sample_count < len(df):
        df, _ = train_test_split(df, train_size=sample_count, stratify=df.label, random_state=seed)
        df = df.reset_index(drop=True)
    train, temp = train_test_split(df, test_size=.2, stratify=df.label, random_state=seed)
    val, test = train_test_split(temp, test_size=.5, stratify=temp.label, random_state=seed)
    return tuple(x.reset_index(drop=True) for x in (train, val, test))


def scores(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p),
            "recall": float(r), "f1": float(f1)}


class CommentDataset(Dataset):
    def __init__(self, frame, tokenizer, max_len, use_parent):
        if use_parent:
            self.enc = tokenizer(frame.parent_comment.tolist(), frame.comment.tolist(), truncation=True, max_length=max_len)
        else:
            self.enc = tokenizer(frame.comment.tolist(), truncation=True, max_length=max_len)
        self.labels = frame.label.astype(int).tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        row = {k: v[i] for k,v in self.enc.items()}
        row["labels"] = self.labels[i]
        return row


def train_transformer(train_df, val_df, test_df, *, model_name, out_dir, use_parent,
                      max_len, batch_size, epochs, lr, weight_decay, warmup_ratio, seed, device):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_ds = CommentDataset(train_df, tokenizer, max_len, use_parent)
    val_ds = CommentDataset(val_df, tokenizer, max_len, use_parent)
    test_ds = CommentDataset(test_df, tokenizer, max_len, use_parent)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, collate_fn=collator)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n,p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(groups, lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio*total_steps), total_steps)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def evaluate(loader, get_preds=False):
        model.eval(); losses=[]; preds=[]; labels=[]
        with torch.no_grad():
            for batch in loader:
                batch = {k:v.to(device) for k,v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    out = model(**batch)
                losses.append(float(out.loss.item())*len(batch["labels"]))
                preds.extend(out.logits.argmax(-1).cpu().tolist()); labels.extend(batch["labels"].cpu().tolist())
        result=scores(labels,preds); result["loss"]=sum(losses)/len(labels)
        return (result,np.asarray(preds),np.asarray(labels)) if get_preds else result

    best_f1=-1.; best_state=None
    for epoch in range(1,epochs+1):
        model.train(); total_loss=0.; seen=0; start=time.perf_counter()
        for batch in train_loader:
            batch={k:v.to(device) for k,v in batch.items()}; optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out=model(**batch); loss=out.loss
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            size=len(batch["labels"]); total_loss+=loss.item()*size; seen+=size
        val=evaluate(val_loader)
        print(f"parent={use_parent} epoch={epoch}/{epochs} train_loss={total_loss/seen:.4f} val_acc={val['accuracy']:.4f} val_f1={val['f1']:.4f} elapsed_s={time.perf_counter()-start:.0f}",flush=True)
        if val["f1"]>best_f1:
            best_f1=val["f1"]; best_state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
    model.load_state_dict(best_state); model.to(device)
    if device.type=="cuda": torch.cuda.synchronize()
    start=time.perf_counter(); test_metrics,preds,labels=evaluate(test_loader,True)
    # The datasets are tokenized before timing; measurement includes batch collation and model evaluation.
    if device.type=="cuda": torch.cuda.synchronize()
    latency_ms=(time.perf_counter()-start)*1000/len(test_df)
    test_metrics["inference_ms_per_comment"] = latency_ms
    test_metrics["best_validation_f1"] = best_f1
    test_metrics["test_classification_report"] = classification_report(labels,preds,target_names=["not_sarcastic","sarcastic"],output_dict=True,zero_division=0)
    out=Path(out_dir)/("bert_with_parent" if use_parent else "bert_without_parent")
    out.mkdir(parents=True,exist_ok=True); model.save_pretrained(out); tokenizer.save_pretrained(out)
    pred_frame=test_df.copy(); pred_frame["prediction"]=preds; pred_frame["error"]=(preds!=labels).astype(int)
    pred_frame.to_csv(out/"test_predictions.csv",index=False)
    return test_metrics,pred_frame


def metadata_slices(pred_frame, min_group=50):
    result={}
    if "subreddit" in pred_frame:
        result["subreddit"]=(pred_frame.groupby("subreddit",dropna=False).agg(n=("error","size"),error_rate=("error","mean"),accuracy=("error",lambda x:1-x.mean())).query("n >= @min_group").sort_values("error_rate",ascending=False).head(30).reset_index().to_dict("records"))
    for col in ["score","ups","downs"]:
        if col in pred_frame:
            buckets=pd.qcut(pd.to_numeric(pred_frame[col],errors="coerce"),q=5,duplicates="drop")
            result[col]=(pred_frame.assign(bucket=buckets).groupby("bucket",observed=True).agg(n=("error","size"),error_rate=("error","mean"),accuracy=("error",lambda x:1-x.mean())).reset_index().astype({"bucket":str}).to_dict("records"))
    return result


def main():
    ap=argparse.ArgumentParser(description="Train baseline and BERT sarcasm classifiers with context ablation.")
    ap.add_argument("--csv", help="Path to train-balanced-sarcasm.csv (downloads with KaggleHub when omitted)")
    ap.add_argument("--output-dir",default="outputs")
    ap.add_argument("--samples",type=int,default=100_000,help="Stratified sample count; use 0 for all cleaned rows")
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--model",default="bert-base-uncased")
    ap.add_argument("--max-len",type=int,default=64); ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--epochs",type=int,default=3); ap.add_argument("--lr",type=float,default=2e-5)
    ap.add_argument("--weight-decay",type=float,default=.01); ap.add_argument("--warmup-ratio",type=float,default=.1)
    ap.add_argument("--contexts",choices=["both","without","with"],default="both")
    args=ap.parse_args(); seed=args.seed; set_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; seed={seed}",flush=True)
    sample_count=None if args.samples==0 else args.samples
    train_df,val_df,test_df=get_data(args.csv,sample_count,seed)
    print(f"Split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}",flush=True)
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    baseline=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=300_000,sublinear_tf=True,strip_accents="unicode")),("clf",LogisticRegression(max_iter=1000,solver="liblinear",random_state=seed))])
    baseline.fit(train_df.comment.fillna(""),train_df.label)
    start=time.perf_counter(); bpred=baseline.predict(test_df.comment.fillna("")); latency=(time.perf_counter()-start)*1000/len(test_df)
    bmetrics=scores(test_df.label,bpred); bmetrics["inference_ms_per_comment"]=latency
    import joblib
    joblib.dump(baseline,outdir/"tfidf_logistic_regression.joblib")
    results={"seed":seed,"model":args.model,"n_train":len(train_df),"n_validation":len(val_df),"n_test":len(test_df),"baseline":bmetrics,"bert":{},"metadata_slices":{}}
    pd.DataFrame([{"model":"TF-IDF + Logistic Regression",**bmetrics}]).to_csv(outdir/"baseline_metrics.csv",index=False)
    contexts=[False,True] if args.contexts=="both" else [args.contexts=="with"]
    pred_frames=[]
    for context in contexts:
        metrics,pf=train_transformer(train_df,val_df,test_df,model_name=args.model,out_dir=outdir,use_parent=context,max_len=args.max_len,batch_size=args.batch_size,epochs=args.epochs,lr=args.lr,weight_decay=args.weight_decay,warmup_ratio=args.warmup_ratio,seed=seed,device=device)
        key="with_parent" if context else "without_parent"
        results["bert"][key]=metrics; results["metadata_slices"][key]=metadata_slices(pf); pred_frames.append((key,pf))
    rows=[{"model":"TF-IDF + Logistic Regression",**bmetrics}]
    for key,metrics in results["bert"].items(): rows.append({"model":f"BERT {key}",**{k:v for k,v in metrics.items() if isinstance(v,(int,float))}})
    pd.DataFrame(rows).to_csv(outdir/"comparison_metrics.csv",index=False)
    with (outdir/"results.json").open("w",encoding="utf-8") as f: json.dump(results,f,indent=2)
    print(json.dumps({"baseline":bmetrics,"bert":{k:{m:v for m,v in x.items() if isinstance(v,(int,float))} for k,x in results["bert"].items()}},indent=2),flush=True)
    print(f"Saved metrics and artifacts under {outdir.resolve()}")

if __name__=="__main__": main()
