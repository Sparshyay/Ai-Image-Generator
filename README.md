# AI Image Generator

A powerful web-based AI image generator using Stable Diffusion 2.1. Create stunning images from text prompts with advanced controls.

## Features

- 🖼️ Generate high-quality images from text prompts
- 🎨 Supports negative prompts to exclude unwanted elements
- ⚡ Fast generation with GPU acceleration
- 🌐 Web interface for easy use
- 🎛️ Advanced controls for fine-tuning results
- 📱 Responsive design works on all devices

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/Sparshyay/Ai-Image-Generator.git
cd Ai-Image-Generator
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
python app.py
```

4. Open your browser and visit: http://localhost:5000

## Deploy to Vercel

1. Install Vercel CLI (if not already installed):
```bash
npm install -g vercel
```

2. Login to your Vercel account:
```bash
vercel login
```

3. Deploy the application:
```bash
vercel
```

4. Follow the prompts to complete the deployment.

## Environment Variables

For local development, create a `.env` file with the following variables:

```
# Optional: Set a different port (default: 5000)
PORT=5000

# Optional: Enable debug mode (not recommended for production)
FLASK_DEBUG=1
```

## Usage

1. Enter your desired prompt in the text area
2. (Optional) Add a negative prompt to exclude unwanted elements
3. Adjust the settings as needed:
   - **Steps**: More steps = better quality but slower (50-100 recommended)
   - **Guidance Scale**: How closely to follow the prompt (7-12 recommended)
   - **Width/Height**: Image dimensions (must be multiples of 8)
4. Click "Generate Image"
5. Download your generated image when complete

## API Endpoints

The application provides the following API endpoints:

- `POST /generate` - Generate an image
  - Parameters: `prompt`, `negative_prompt` (optional), `steps`, `guidance_scale`, `width`, `height`
  - Returns: JSON with `success`, `image_url`, and `filename`

- `GET /health` - Health check endpoint
  - Returns: `{"status": "ok"}`

## License

This project is open source and available under the [MIT License](LICENSE).
