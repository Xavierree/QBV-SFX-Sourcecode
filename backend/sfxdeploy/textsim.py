"""Sentence embeddings for the describe approach: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, commit e8f8c21122; multilingual because SFX contains Indonesian / Portuguese / Spanish speech clips),
mean pooling over the attention mask + L2 normalisation (the model's own sentence-transformers recipe). Fixed before any description was scored."""
import numpy as np, torch
from . import config as C
_M = {}
def embed(texts, device='cpu', bs=128):
    from transformers import AutoTokenizer, AutoModel
    if 'm' not in _M:
        p = str(C.ROOT / 'models' / 'text' / 'paraphrase-multilingual-MiniLM-L12-v2'); _M['tok'] = AutoTokenizer.from_pretrained(p); _M['m'] = AutoModel.from_pretrained(p).to(device).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), bs):
            b = _M['tok']([t if t else ' ' for t in texts[s:s + bs]], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device); h = _M['m'](**b).last_hidden_state
            m = b['attention_mask'].unsqueeze(-1).to(h.dtype); e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9); out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
    return np.concatenate(out)
