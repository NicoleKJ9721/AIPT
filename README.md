# AI Industrial Inspection Platform

## Frontend (React + Vite)
The frontend is located in the `frontend` directory.

### Setup
```bash
cd frontend
npm install
```

### Development
```bash
npm run dev
```

### Build
```bash
npm run build
```

## Backend (FastAPI + YOLOv8)
The backend is located in the `backend` directory.

### Setup
```bash
cd backend
pip install -r requirements.txt
```

### Run Server
```bash
# Run from the backend directory
python -m uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.
Docs available at `http://localhost:8000/docs`.

### Testing
```bash
# Run from the project root or backend directory
pytest backend/tests
```

## Features
- **Annotation Studio**: Responsive canvas, specialized tools, layer management.
- **Project Management**: Local data management, export/import.
- **AI Models**: YOLOv8 integration for object detection.
