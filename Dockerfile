# 1️⃣ Base image (Python environment)
FROM python:3.10

# 2️⃣ Set working directory inside container
WORKDIR /app

# 3️⃣ Copy all project files into container
COPY . .

# 4️⃣ Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Expose API port
EXPOSE 8000

# 6️⃣ Start FastAPI server
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]