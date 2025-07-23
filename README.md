# 🤖 PDF QA Chatbot

A simple PDF Question-Answering chatbot built with **Streamlit**, **FAISS**, **SentenceTransformers**, and **DeepSeek-R1** via [OpenRouter.ai](https://openrouter.ai).

![screenshot](.github/demo.png) <!-- optional -->

## 📄 Features

- Upload any PDF
- Ask natural language questions
- Uses vector search + LLM to answer
- Free and deployable on Streamlit Cloud

## 🚀 Demo

👉 [Launch the App](https://pdf-qa-chatbot.streamlit.app) *(coming soon)*

## 🛠 Tech Stack

- Streamlit
- SentenceTransformers (MiniLM-L6-v2)
- FAISS (Vector Search)
- DeepSeek-R1 LLM via OpenRouter
- Python 3.10+

## 🧪 Local Setup

```bash
git clone https://github.com/kashifcodes92/pdf-qa-chatbot.git
cd pdf-qa-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
