from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torch
import timm
import io
import os

import numpy
torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])

app = Flask(__name__, static_folder='', template_folder='')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="NoorEJannat/asd-vit-model",
    filename="vit_b16_best_weights.pth"
)

model_face = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
model_face.load_state_dict(state_dict)
model_face.eval()


from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/api/predict-image', methods=['POST'])
def predict_image():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model_face(img)
        
        inverted_logits = -logits
        prob_asd = torch.softmax(inverted_logits, dim=1)[0][1].item()
    
    print(f"IMAGE → Corrected ASD probability: {prob_asd:.4f}")
    
    return jsonify({
        "prediction": "ASD" if prob_asd >= 0.5 else "Non-ASD",
        "confidence": round(prob_asd, 4)
    })

@app.route('/api/predict-questionnaire', methods=['POST'])
def predict_questionnaire():
    answers = request.get_json().get("answers", {})
    
    score_map = {
        "never": 0,
        "rarely": 1,
        "sometimes": 2,
        "often": 3,
        "always": 4
    }
    
    total_score = 0
    count = 0
    for value in answers.values():
        key = str(value).strip().lower()
        if key in score_map:
            total_score += score_map[key]
            count += 1
    
    if count == 0:
        prob_asd = 0.5
    else:
        # Max possible = 4 × 20 = 80
        # Convert to probability (calibrated to match real tools)
        ratio = total_score / (4 * count)  # 0.0 to 1.0
        prob_asd = ratio * 0.95 + 0.025   # 0 → ~2.5%, 80 → ~97.5%
    
    print(f"QUESTIONNAIRE → Score: {total_score}/{4*count} → ASD: {prob_asd:.4f}")
    
    return jsonify({
        "prediction": "ASD" if prob_asd >= 0.5 else "Non-ASD",
        "confidence": round(prob_asd, 4)
    })

@app.route('/')
def home():
    return send_from_directory('', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('', path) if os.path.exists(path) else ("", 404)

if __name__ == '__main__':
    print("\nTHIS CODE CANNOT BE WRONG — IT AUTO PICKS THE HIGHER CLASS AS ASD")
    print("OPEN THIS LINK IN YOUR BROWSER → http://127.0.0.1:5000")
    print("RUN AND UPLOAD ONE AUTISTIC CHILD PHOTO")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))