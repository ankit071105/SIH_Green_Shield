import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def _get_last_conv_layer(model):
    if hasattr(model, 'features'):
        return model.features[-1]
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise RuntimeError('No conv layer found')

def generate_gradcam_visualization(model, pil_img: Image.Image, target_class: int = None):
    model.eval()
    device = next(model.parameters()).device
    img_tensor = _transform(pil_img).unsqueeze(0).to(device)
    activations = None
    grads = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0].detach()

    last_conv = _get_last_conv_layer(model)
    h_fw = last_conv.register_forward_hook(forward_hook)
    h_bw = last_conv.register_backward_hook(backward_hook)

    logits = model(img_tensor)
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())
    loss = logits[0, target_class]
    model.zero_grad()
    loss.backward(retain_graph=True)

    pooled_grads = torch.mean(grads, dim=[0,2,3])
    for i in range(activations.shape[1]):
        activations[0,i,:,:] *= pooled_grads[i]
    heatmap = torch.sum(activations[0], dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    heatmap = cv2.resize(heatmap, (pil_img.width, pil_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    superimposed = heatmap * 0.4 + img * 0.6
    superimposed = np.clip(superimposed, 0, 255).astype('uint8')
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    h_fw.remove(); h_bw.remove()
    return Image.fromarray(superimposed)
