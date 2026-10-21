import os, json, re, torch, warnings
from PIL import Image
warnings.filterwarnings("ignore")
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-VL-32B-Instruct"
WORK_DIR   = "/root/autodl-tmp/work"
QUERIES    = os.path.join(WORK_DIR, "queries.json")
OUT_FILE   = os.path.join(WORK_DIR, "v32b_preds.json")

PROMPT = 'Find the target object described by: "{query}"\n\nOutput a TIGHT bounding box around ONLY the target. Reply ONLY with JSON: {{"bbox_2d": [x1,y1,x2,y2]}} in 0-1000 coordinates.'
BN_PAT = re.compile(r'\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]')

def main():
    print("loading Qwen3-VL-32B fp16, forcing all on GPU...")
    # Force everything on GPU, use flash attention to save memory
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        
    )
    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    queries = json.load(open(QUERIES, encoding="utf-8"))
    preds = {}
    if os.path.exists(OUT_FILE):
        preds = json.load(open(OUT_FILE, encoding="utf-8"))
    print(f"total {len(queries)}, done {len(preds)}")

    import time
    t0 = time.time()
    for i, (key, qv) in enumerate(queries.items()):
        if key in preds:
            continue
        vpath = os.path.join(WORK_DIR, qv["visible"].replace(".png", ".jpg"))
        if not os.path.exists(vpath):
            preds[key] = [0.0, 0.0, 1.0, 1.0]
            continue
        v = Image.open(vpath).convert("RGB")
        prompt = PROMPT.format(query=qv["query"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": v},
            {"type": "text", "text": prompt}
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[v], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        resp = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        m = BN_PAT.search(resp)
        if m:
            c4 = [float(m.group(j))/1000.0 for j in range(1,5)]
            x1,y1,x2,y2 = c4
            x1=max(0,min(1,x1)); y1=max(0,min(1,y1))
            x2=max(0,min(1,x2)); y2=max(0,min(1,y2))
            if x1>x2: x1,x2=x2,x1
            if y1>y2: y1,y2=y2,y1
            if x2-x1<0.01 or y2-y1<0.01:
                preds[key]=[0.0,0.0,1.0,1.0]
            else:
                preds[key]=[round(x1,4),round(y1,4),round(x2,4),round(y2,4)]
        else:
            preds[key]=[0.0,0.0,1.0,1.0]
        if (i+1) % 10 == 0:
            json.dump(preds, open(OUT_FILE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
            elapsed = time.time() - t0
            speed = len(preds) / elapsed
            eta = (len(queries) - len(preds)) / speed / 60
            print(f"done {len(preds)}/{len(queries)} ({speed:.2f}/s) ETA {eta:.0f}min", flush=True)

    json.dump(preds, open(OUT_FILE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("finished!", OUT_FILE)

if __name__ == "__main__":
    main()
