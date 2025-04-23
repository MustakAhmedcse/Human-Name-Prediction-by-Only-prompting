# Name Realism Classifier API

This project provides a Flask-based REST API that uses OpenAI's GPT model to determine whether a given name is a realistic human name or not.

## 🔧 Features

- Validates input name format locally.
- Calls OpenAI's GPT model for classification.
- Returns a JSON response indicating whether the name is realistic or not.
- Logs activity for debugging and traceability.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/name-realism-classifier.git
cd name-realism-classifier
```

### 2. Set Up Environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

Replace `your_openai_api_key` with your actual OpenAI API key.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist, generate it with:

```bash
pip freeze > requirements.txt
```

## ▶️ Running the Server

```bash
python app.py
```

The server will start on `http://0.0.0.0:5000`.

## 📬 API Usage

### Endpoint

`POST /predict`

### Request Payload

```json
{
  "name": "John Doe"
}
```

### Successful Response

#### Realistic Name

```json
{
  "name": "John Doe",
  "prediction": "Realistic"
}
```

#### Not Realistic Name

```json
{
  "name": "Table Chair",
  "prediction": "Not Realistic",
  "reason": "common phrase"
}
```

## 🧪 Example Test with Curl

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"name": "John Doe"}'
```

## 📒 Notes

- The OpenAI model used: `gpt-4.1-nano`
- Ensure the `.env` file is not committed to version control.
- Logging is configured at `INFO` level.

## 🛡️ License

MIT License. See [LICENSE](LICENSE) for details.


