# CityTransit Analytics & Reporting Service

Servicio de análisis y reportería construido con Python/FastAPI que proporciona análisis avanzado de datos, predicciones con Machine Learning y análisis de sentimientos.

## 🚀 Características

### Machine Learning
- **LSTM (Long Short-Term Memory)**: Predicción de demanda de pasajeros basada en series temporales
- **DBSCAN**: Segmentación de usuarios y detección de outliers
- **BERT**: Análisis de sentimientos de feedback y comentarios

### Análisis de Datos
- Dashboards de KPIs en tiempo real
- Reportes analíticos personalizados
- Análisis de patrones de uso
- Métricas de rendimiento

### Bases de Datos
- **ClickHouse**: OLAP para queries analíticas rápidas
- **MongoDB**: Almacenamiento flexible de reportes
- **Redis**: Cache de resultados

## 📋 Requisitos

- Python 3.11+
- Docker & Docker Compose
- ClickHouse
- MongoDB
- Redis

## 🛠️ Instalación

### 1. Clonar el repositorio

```bash
cd analytics-service
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Copiar `.env.example` a `.env` y configurar:

```bash
cp .env.example .env
```

### 5. Ejecutar con Docker

```bash
docker-compose up -d
```

O ejecutar directamente:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📚 API Endpoints

### Health Check
```
GET /health
GET /api/v1/health
```

### Análisis de Demanda
```
POST /api/v1/analytics/demand/predict
GET  /api/v1/analytics/demand/forecast/{route_id}
GET  /api/v1/analytics/demand/trends
```

### Segmentación de Usuarios
```
POST /api/v1/analytics/users/segment
GET  /api/v1/analytics/users/clusters
GET  /api/v1/analytics/users/outliers
```

### Análisis de Sentimientos
```
POST /api/v1/analytics/sentiment/analyze
GET  /api/v1/analytics/sentiment/summary
GET  /api/v1/analytics/sentiment/trends
```

### Reportes y KPIs
```
GET  /api/v1/reports/kpis
GET  /api/v1/reports/dashboard
POST /api/v1/reports/generate
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📖 Documentación

Una vez ejecutando el servicio:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 Configuración

### ClickHouse
```bash
# Crear tabla de análisis
CREATE TABLE IF NOT EXISTS analytics_events (
    event_id UUID,
    user_id Int32,
    event_type String,
    timestamp DateTime,
    data String
) ENGINE = MergeTree()
ORDER BY (timestamp, user_id);
```

### MongoDB
```javascript
// Crear índices
db.reports.createIndex({ "created_at": -1 });
db.sentiment_analysis.createIndex({ "analyzed_at": -1 });
```

## 🚀 Integración con Backend Java

El servicio se comunica con el backend Java de CityTransit para:
- Obtener datos de usuarios y transacciones
- Sincronizar resultados de análisis
- Autenticación JWT compartida

## 📊 Modelos de Machine Learning

### LSTM - Predicción de Demanda
- **Input**: Series temporales de uso (hora, día, clima, eventos)
- **Output**: Predicción de demanda para las próximas 24-48 horas
- **Métricas**: RMSE, MAE, R²

### DBSCAN - Segmentación
- **Features**: Frecuencia de uso, rutas, horarios, gasto
- **Output**: Clusters de usuarios y outliers
- **Aplicación**: Marketing personalizado

### BERT - Sentimientos
- **Input**: Texto de feedback o comentarios
- **Output**: Positivo/Neutral/Negativo + Score de confianza
- **Modelo**: BERT fine-tuned en español

## 📝 Licencia

MIT License
"# analytics-service" 
