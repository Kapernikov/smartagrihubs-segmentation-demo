import datetime

def create_metadata(model_name: str) -> dict:
    """ not necessary if we use mlflow """
    metadata = {}
    metadata['modelname'] = model_name
    metadata['timestamp'] = datetime.datetime.now().isoformat()
    return metadata