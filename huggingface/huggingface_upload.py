import argparse
from huggingface_hub import HfApi

def upload_to_hf(folder_path, repo_id, token):
    api = HfApi(token=token)
    print(f"Uploading {folder_path} to {repo_id}...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
    )
    print("Upload complete!")

def main():
    parser = argparse.ArgumentParser(description="Upload trained model to Hugging Face Hub")
    parser.add_argument('--folder', type=str, required=True, help='Path to the experiment folder to upload')
    parser.add_argument('--repo_id', type=str, default="kritaphatson/thai_snake_image_classifier", help='Hugging Face repo ID')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')
    
    args = parser.parse_args()
    
    upload_to_hf(args.folder, args.repo_id, args.token)

if __name__ == "__main__":
    main()
