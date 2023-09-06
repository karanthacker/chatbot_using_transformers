import gradio as gr
import torch
from torchtext.data.utils import get_tokenizer
import numpy as np
import subprocess

from huggingface_hub import hf_hub_download
from transformer import Transformer

model_url = "https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl"
subprocess.run(["pip", "install", model_url])

MAX_LEN = 350

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
vocab = torch.load(hf_hub_download(repo_id="karanthacker/chat_ai",
                                   filename="vocab.pth"))
vocab_token_dict = vocab.get_stoi()
indices_to_tokens = vocab.get_itos()
pad_token = vocab_token_dict['<pad>']
unknown_token = vocab_token_dict['<unk>']
sos_token = vocab_token_dict['<sos>']
eos_token = vocab_token_dict['<eos>']
text_pipeline = lambda x: vocab(tokenizer(x))

d_model = 512
heads = 8
N = 6
src_vocab = len(vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(len(vocab), len(vocab), d_model, N, heads).to(device)
model.load_state_dict(torch.load(hf_hub_download(repo_id="karanthacker/chat_ai",
                                      filename="alpaca_weights.pt"), map_location=device))
model.eval()

def respond(input):
    model.eval()
    src = torch.tensor(text_pipeline(input), dtype=torch.int64).unsqueeze(0).to(device)
    src_mask = ((src != pad_token) & (src != unknown_token)).unsqueeze(-2).to(device)
    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(MAX_LEN).type_as(src.data).to(device)
    outputs[0] = torch.tensor([vocab.get_stoi()['<sos>']])
    for i in range(1, MAX_LEN):
        trg_mask = np.triu(np.ones([1, i, i]), k=1).astype('uint8')
        trg_mask = torch.autograd.Variable(torch.from_numpy(trg_mask) == 0).to(device)

        out = model.out(model.decoder(outputs[:i].unsqueeze(0), e_outputs, src_mask, trg_mask))
        
        out = torch.nn.functional.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == vocab_token_dict['<eos>']:
            break
        
    return ' '.join([indices_to_tokens[ix] for ix in outputs[1:i]])

iface = gr.Interface(fn=respond,
                     inputs="text",
                     outputs="text")
iface.launch()