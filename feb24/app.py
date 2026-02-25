import torch
import torch.nn as nn
import gradio as gr

model_data = torch.load('model.pth')
fm = model_data['fm']
fs = model_data['fs']
tm = model_data['tm']
ts = model_data['ts']
parameters = model_data['parameters']

model = nn.Linear(2,1)
model.load_state_dict(model_data['parameters'])

def f(weight, engine_size):
    features = torch.tensor([
        [weight, engine_size]
    ]).float()
    X = (features - fm)/fs
    Yhat = model(X)
    prediction = Yhat * ts + tm
    return(prediction.item())

with gr.Blocks() as iface:
    weight_box = gr.Number(label = 'Provide Weight of Vehicle')
    engine_box = gr.Number(label = 'Provide Engine Size')
    MPG_box = gr.Number(label = 'MPG Prediction')
    weight_box.change(fn = f, inputs = [weight_box, engine_box], outputs = [MPG_box])
    engine_box.change(fn = f, inputs = [weight_box, engine_box], outputs = [MPG_box])

iface.launch()
