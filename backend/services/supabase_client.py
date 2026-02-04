import os
from supabase import create_client, Client
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseService:
    def __init__(self):
        self.url: str = os.environ.get("SUPABASE_URL", "")
        self.key: str = os.environ.get("SUPABASE_KEY", "")
        self.client: Client = None
        
        if self.url and self.key:
            try:
                self.client = create_client(self.url, self.key)
                logger.info("✅ Supabase client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Supabase client: {e}")
        else:
            logger.warning("⚠️ SUPABASE_URL or SUPABASE_KEY not found. Cloud storage disabled.")

    def upload_file(self, bucket: str, file_path: Path, destination_path: str) -> str:
        """
        Upload a file to Supabase Storage.
        Returns the public URL of the uploaded file.
        """
        if not self.client:
            logger.warning("Supabase client not initialized. Skipping upload.")
            return ""

        try:
            with open(file_path, 'rb') as f:
                self.client.storage.from_(bucket).upload(
                    file=f,
                    path=destination_path,
                    file_options={"upsert": "true"}
                )
            
            # Get public URL
            public_url = self.client.storage.from_(bucket).get_public_url(destination_path)
            logger.info(f"Checking public URL type: {type(public_url)} -> {public_url}")

            # Ensure we return a string
            if isinstance(public_url, str):
                return public_url
            # Sometimes the SDK might return a response object depending on version/mocking
            return str(public_url)
            
        except Exception as e:
            logger.error(f"Failed to upload {file_path} to {bucket}/{destination_path}: {e}")
            raise e

    def download_file(self, bucket: str, source_path: str, destination_path: Path) -> bool:
        """
        Download a file from Supabase Storage.
        """
        if not self.client:
            logger.warning("Supabase client not initialized. Skipping download.")
            return False

        try:
            response = self.client.storage.from_(bucket).download(source_path)
            
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            with open(destination_path, 'wb') as f:
                f.write(response)
                
            return True
        except Exception as e:
            logger.error(f"Failed to download {bucket}/{source_path}: {e}")
            return False

    def list_files(self, bucket: str, folder: str = "") -> list:
        """
        List files in a bucket folder.
        """
        if not self.client:
            return []
            
        try:
            return self.client.storage.from_(bucket).list(folder)
        except Exception as e:
            logger.error(f"Failed to list files in {bucket}/{folder}: {e}")
            return []

# Singleton instance
supabase_service = SupabaseService()
