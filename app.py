import os
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from image_generator import generate_image
from PIL import Image as PILImage
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Generate image from text
@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')
        steps = int(data.get('steps', 50))
        guidance_scale = float(data.get('guidance_scale', 9.0))
        width = int(data.get('width', 768))
        height = int(data.get('height', 768))
        
        # Generate a unique filename
        filename = f"generated_{uuid.uuid4().hex}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Generate the image
        generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            output_dir=app.config['UPLOAD_FOLDER'],
            filename=filename
        )
        
        return jsonify({
            'success': True,
            'image_url': f"/static/uploads/{filename}",
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Serve uploaded files
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
