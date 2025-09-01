# Fabric Defect Detection - Vercel Deployment

🚀 **Live Demo**: [Deploy on Vercel](https://vercel.com/new/clone?repository-url=https://github.com/LOVEPOISON11/fabric-defect-detection1)

A web-based fabric defect detection system optimized for Vercel deployment. This application uses computer vision to identify defects in fabric images through an intuitive web interface.

## ✨ Features

- 🖼️ **Image Upload Detection**: Upload fabric images for defect analysis
- 📷 **Camera Integration**: Real-time detection using device camera
- 🎯 **Multiple Defect Types**: Detects holes, stains, thread breaks, and more
- 📊 **Detailed Results**: Confidence scores and defect counts
- 🌐 **Vercel Optimized**: Lightweight deployment for serverless hosting
- 📱 **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/LOVEPOISON11/fabric-defect-detection1)

## 🛠️ Local Development

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/LOVEPOISON11/fabric-defect-detection1.git
   cd fabric-defect-detection1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally**
   ```bash
   # For local development
   python api/index.py

   # Or using Vercel CLI
   vercel dev
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000` (Vercel) or `http://localhost:5000` (direct Python)

## 📁 Project Structure

```
fabric-defect-detection1/
├── api/
│   └── index.py                 # Main Flask application (Vercel entry point)
├── templates/
│   ├── index.html              # Main upload interface
│   └── camera.html             # Camera detection interface
├── models/
│   └── best.pt                 # YOLOv8 model (optional for full AI)
├── fabric_defect_detector.py   # Full AI detector (PyTorch)
├── fabric_defect_detector_lite.py # Lightweight detector (Vercel-optimized)
├── requirements.txt            # Python dependencies
├── vercel.json                 # Vercel configuration
└── README.md                   # This file
```

## 🔧 Configuration

### Vercel Deployment
The application automatically detects the deployment environment:
- **Vercel**: Uses lightweight pattern-based detection
- **Local with AI**: Uses full YOLOv8 model if available
- **Demo Mode**: Basic pattern analysis fallback

## 🎯 Usage

### Web Interface

1. **Image Upload Mode**
   - Visit the main page
   - Drag & drop or select fabric images
   - Adjust confidence threshold
   - Click "Detect Defects"
   - View results with highlighted defects

2. **Camera Mode**
   - Click "Camera Mode"
   - Allow camera permissions
   - Point camera at fabric
   - Enable real-time detection
   - View live defect analysis

### API Endpoints

- `GET /` - Main interface
- `GET /camera` - Camera interface
- `POST /detect` - Image detection API
- `GET /health` - Health check

## 🔍 Detection Types

The system can identify various fabric defects:
- **Holes** - Physical damage or tears
- **Stains** - Color discoloration
- **Thread Breaks** - Broken or loose threads
- **Color Variation** - Inconsistent coloring
- **Texture Defects** - Surface irregularities
- **Weave Defects** - Pattern inconsistencies

## 🚀 Deployment Options

### Vercel (Recommended)
1. Fork this repository
2. Connect to Vercel
3. Deploy automatically

### Other Platforms
- **Heroku**: Add `Procfile` with `web: python api/index.py`
- **Railway**: Direct deployment supported
- **Render**: Python web service

## 📊 Performance

- **Vercel**: ~2-5 seconds per image
- **Local AI**: ~0.5-2 seconds per image
- **Memory**: <512MB on Vercel
- **File Size**: Up to 16MB images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally and on Vercel
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**Made with ❤️ for fabric quality control**
