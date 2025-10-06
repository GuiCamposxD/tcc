# SageMaker Training Setup

## Cost-Efficient Architecture

**Your JupyterLab Space:** ml.t3.medium (~$0.05/hour)
- Runs data preparation only
- Can be stopped after launching training job

**Training Job:** ml.g4dn.xlarge (~$0.74/hour) 
- GPU instance for heavy training
- Runs independently - you pay only during training
- Auto-stops when complete

## Files

- `main_sagemaker.ipynb` - Run this in your ml.t3.medium Space
- `train.py` - Training script (uploaded to S3 automatically)
- `requirements.txt` - Dependencies for training container

## Workflow

1. Open `main_sagemaker.ipynb` in your ml.t3.medium Space
2. Run cells to prepare data and launch training job
3. **Close notebook and stop Space** - training continues
4. Monitor job in SageMaker Console > Training > Training jobs
5. Model saved to S3 when complete

## Cost Savings

Traditional approach: Keep GPU instance running = $0.74/hour continuously
This approach: ml.t3.medium + training job = $0.05/hour + $0.74/hour only during training

**Example:** 2-hour training
- Old way: $1.48 (if you keep GPU running)
- New way: $0.10 (prep) + $1.48 (training) = $1.58 total, but Space can be stopped
