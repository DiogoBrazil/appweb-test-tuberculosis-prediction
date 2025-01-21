from fasthtml.common import *
from model import load_model, val_transforms, classes
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from fastapi import Request

app, rt = fast_app()

# Carregar o modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device=device)

@rt('/')
def get():
    return Titled(
        "Predição de Tuberculose Pulmonar",
        Div(  
            Form(
                Div(
                    Div(
                        Input(
                            type="file", 
                            name="file", 
                            accept="image/*"
                        ),
                        style="display: flex; justify-content: center; align-items: center; width: 100%;"
                    ),
                    style="margin-top: 50px;"
                ),
                Div(
                    Button(
                        "Enviar", 
                        type="submit",
                        style="padding: 10px 155px; font-size: 16px;"
                    ),
                    style="margin-top: 50px; text-align: center;"
                ),
                method="post",
                enctype="multipart/form-data",
                hx_post="/upload",
                hx_target="#result",
                hx_swap="innerHTML",
                style="display: flex; flex-direction: column; align-items: center;"
            ),
            Div(id="result"),
            style="width: 100%;"
        ),
        style="text-align: center; padding-top: 50px;"  # Adicionei padding-top ao título
    )

@rt('/upload')
async def post(file: UploadFile):
    # Leia a imagem enviada
    image_data = await file.read()
    img_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Aplique as transformações
    img_tensor = val_transforms(img_pil).unsqueeze(0).to(device)
    
    # Faça a predição
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().flatten()
    
    # Prepare a imagem para exibição
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_html = f'<img src="data:image/jpeg;base64,{img_str}" alt="Uploaded Image" width="450" height="450" style="margin-right: 20px;"/>'
    
    # Prepare o gráfico de barras
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(classes, probs, color=['green', 'red'])
    ax.set_xlabel('Classes')
    ax.set_ylabel('Probabilidades (%)')
    ax.set_title('Probabilidades por Classe')
    
    # Aumenta o limite y para acomodar os rótulos
    ax.set_ylim([0, 1.1])  # Aumenta um pouco o limite superior
    
    # Adicione os valores acima de cada barra com porcentagem
    ax.bar_label(bars, fmt='%.1f%%', padding=3, labels=[f'{p*100:.1f}%' for p in probs])
    
    # Ajuste das margens e layout
    plt.tight_layout()  # Ajusta automaticamente o layout
    
    # Desabilita a exibição de valores que ultrapassem os limites
    ax.set_clip_on(True)
    
    # Salve o gráfico em uma string base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)  # Fecha a figura para liberar memória
    buf.seek(0)
    graph_str = base64.b64encode(buf.getvalue()).decode()
    graph_html = f'<img src="data:image/png;base64,{graph_str}" alt="Probability Graph"/>'
    
    # Retorne a imagem e o gráfico lado a lado
    return NotStr(f'<div style="display: flex; justify-content: center; align-items: center;">{img_html}{graph_html}</div>')

if __name__ == "__main__":
    serve()
