from fastapi import FastAPI, HTTPException
from .schema import ChurnPredictionInput, ChurnPredictionOutput
from .model import ChurnModel
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn",
    version="1.0.0"
)

# Initialize model
model = ChurnModel()

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the Churn Prediction API"}

@app.post("/predict", response_model=ChurnPredictionOutput)
async def predict_churn(input_data: ChurnPredictionInput):
    """Predict customer churn based on input features"""
    try:
        logger.info(f"Received prediction request")
        start_time = time.time()
        
        # Make prediction
        prediction = model.predict(input_data)
        
        processing_time = time.time() - start_time
        logger.info(f"Prediction made in {processing_time:.4f} seconds: {prediction}")
        
        return prediction
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}