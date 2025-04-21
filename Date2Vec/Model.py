import sys
# Module injection: tell unpickler that "Model" refers to this module.
import Date2Vec.Model as d2v_module
sys.modules["Model"] = d2v_module

import torch
from torch import nn

# --------------------------
# 1. Define Date2Vec first.
# --------------------------
class Date2Vec(nn.Module):
    def __init__(self, k=32, act="sin"):
        super(Date2Vec, self).__init__()
        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1

        self.fc1 = nn.Linear(6, k1)
        self.fc2 = nn.Linear(6, k2)
        self.d2 = nn.Dropout(0.3)
 
        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

        self.fc3 = nn.Linear(k, k // 2)
        self.d3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(k // 2, 6)
        self.fc5 = nn.Linear(6, 6)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.d2(self.activation(self.fc2(x)))
        out = torch.cat([out1, out2], 1)
        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

    def encode(self, x):
        out1 = self.fc1(x)
        out2 = self.activation(self.fc2(x))
        out = torch.cat([out1, out2], 1)
        return out

# -----------------------------------------------------
# 2. Add safe globals AFTER Date2Vec is defined.
# -----------------------------------------------------
torch.serialization.add_safe_globals({'Date2Vec': Date2Vec})

# -----------------------------------------------------
# 3. Define Date2VecConvert to load the pretrained model.
# -----------------------------------------------------
class Date2VecConvert:
    def __init__(self, model_path="./d2v_model/d2v_98291_17.169918439404636.pth"):
        # Load the full checkpoint by setting weights_only=False.
        self.model = torch.load(model_path, map_location='cpu', weights_only=False).eval()
    
    def __call__(self, x):
        with torch.no_grad():
            # Ensure x is a torch.Tensor.
            return self.model.encode(torch.Tensor(x).unsqueeze(0)).squeeze(0).cpu()

if __name__ == "__main__":
    model = Date2Vec()
    inp = torch.randn(1, 6)
    out = model(inp)
    print(out)
    print(out.shape)
