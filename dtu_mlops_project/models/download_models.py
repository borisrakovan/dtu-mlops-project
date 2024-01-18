from google.cloud import storage


def download_model(bucket_name, model_name):
    # TODO handle path to model
    source_blob_name = model_name
    destination_file_name = model_name
    print(f"Downloading model to {destination_file_name}")

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
    except Exception as e:
        print(f"Error downloading blob: {e}")
        raise


if __name__ == "__main__":
    bucket_name = "dtu_mlops_project_data"
    model_name = "epoch=5-step=15912.ckpt"
    download_model(bucket_name, model_name)
