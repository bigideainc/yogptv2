import os

from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder
from transformers import PreTrainedModel, PreTrainedTokenizerFast

load_dotenv()

class HuggingFaceModelStore:
    @classmethod
    def assert_access_token_exists(cls) -> str:
        """Ensures that the Hugging Face access token exists in the environment variables."""
        token = os.getenv("HF_ACCESS_TOKEN")
        if not token:
            raise ValueError("No Hugging Face access token found in environment. Make sure HF_ACCESS_TOKEN is set.")
        return token

    def upload_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, job_id: str):
        """Uploads a trained model to Hugging Face, creating a new repository if it does not exist."""
        token = self.assert_access_token_exists()
        repo_name = f"job_{job_id}"  # Dynamic repository name based on job ID
        api = HfApi()
        
        try:
            # Check if the repository exists
            repo_url = api.create_repo(repo_name, token=token, private=True, exist_ok=True)  # `exist_ok=True` will not raise an error if repo exists
            print(f"Repository URL: {repo_url}")

            # Save model and tokenizer locally
            model.save_pretrained(repo_name)
            tokenizer.save_pretrained(repo_name)

            # Upload to Hugging Face Hub
            model.push_to_hub(repo_name, use_auth_token=token)
            tokenizer.push_to_hub(repo_name, use_auth_token=token)

            print(f"Successfully uploaded {repo_name} to Hugging Face Hub.")
            return repo_url
        except Exception as e:
            print(f"Failed to create or upload to repository {repo_name}: {str(e)}")


