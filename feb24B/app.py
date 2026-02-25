import torch
import torch.nn as nn
import gradio as gr 

model_data = torch.load('model.pth')

fm = model_data['fm']
fs = model_data['fs']
parameters = model_data['parameters']

linear = model = nn.Linear(1,1)
linear.load_state_dict(parameters)

model = nn.Sequential(
    linear,
    nn.Sigmoid()
)

def f(size):
    features = torch.tensor([
        [size]
    ]).float()
    X = (features - fm)/fs
    classification = model(X)

    if classification > .5:
        return 'Malignant'
    else:
        return 'Benign'

with gr.Blocks() as iface:
    tumor_box = gr.Number(label = 'Provide Size of Tumor (cm)')
    diag_box = gr.Text(label = 'Diagnosis')
    tumor_box.change(fn = f, inputs = [tumor_box], outputs = [diag_box])

iface.launch()