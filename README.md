# 🛡️ SafeSphere

SafeSphere is a real-time content moderation backend designed to detect **hate speech** and **misinformation** on the web. It provides a simple REST API built with **FastAPI**, integrating trusted tools like:


---

## ⚙️ Features

- 🔥 **Real-time Hate Speech Detection** using Google’s Perspective API  
- 📚 **Misinformation Detection** via Google's Fact Check Tools API  
- 💨 **FastAPI Backend** – Lightweight, asynchronous, and production-ready  
- 🌐 **RESTful Endpoints** for integration with browser extensions or any frontend  
- 🧠 Built with speed and accuracy in mind – no heavy LLMs involved (yet)  

---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/safesphere.git
cd safesphere
```

### 2. Create & Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file:
```env
PERSPECTIVE_API_KEY=your_perspective_api_key
FACT_CHECK_API_KEY=your_fact_check_api_key
```

### 5. Run the Server
```bash
uvicorn main:app --reload
```

---

## 🔭 Future Development

- 🧱 **Upgrade to Google Guardrails API** for better moderation scalability and context-aware detection  
- 🖼️ **Add OCR Support** to analyze images and detect embedded hate speech or misinformation  
- 🌐 **Deploy as a browser extension** for inline moderation of web content  

---

## 🧑‍💻 Contributing

Pull requests are welcome! Please open an issue first to discuss what you’d like to change or add.

---

## 📄 License

This project is licensed under the MIT Open License. See the LICENSE file for more details.

---

## 🌐 Connect

- 💬 Have questions? Open an [issue](https://github.com/your-username/safesphere/issues)  
- ⭐ Star this repo if you find it useful!
