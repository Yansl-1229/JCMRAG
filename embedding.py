import requests
import json
import base64
from PIL import Image
from io import BytesIO

class JinaCLIPEmbeddings:
    def __init__(self, api_key):
        self.url = 'https://aihubmix.com/v1/embeddings'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.model = "jina-clip-v2"

    def _encode_image_to_base64(self, image_path_or_url):
        try:
            if image_path_or_url.startswith(('http://', 'https://')):
                response = requests.get(image_path_or_url)
                response.raise_for_status()
                img_bytes = response.content
            else:
                with open(image_path_or_url, "rb") as image_file:
                    img_bytes = image_file.read()
            
            # Optional: Resize image if it's too large to reduce base64 string length
            try:
                img = Image.open(BytesIO(img_bytes))
                img.thumbnail((512, 512)) # Resize to max 512x512
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
            except Exception as e:
                print(f"Warning: Could not process image for resizing: {e}")

            return base64.b64encode(img_bytes).decode('utf-8')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image from URL: {e}")
            return None
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path_or_url}")
            return None
        except Exception as e:
            print(f"Error encoding image {image_path_or_url}: {e}")
            return None

    def embed_texts(self, texts):
        if not texts:
            return []
        
        payload_input = []
        for text in texts:
            if isinstance(text, str):
                payload_input.append({"text": text})
            else:
                print(f"Warning: Skipping invalid text input: {text}")

        if not payload_input:
            return []

        data = {
            "model": self.model,
            "input": payload_input
        }
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status() # Raise an exception for HTTP errors
            results = response.json()
            if "data" in results:
                return [item["embedding"] for item in results["data"]]
            else:
                print(f"Error in API response (text embedding): {results}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Request failed (text embedding): {e}")
            return []
        except json.JSONDecodeError:
            print(f"Failed to decode JSON response (text embedding): {response.text}")
            return []

    def embed_images(self, image_paths_or_urls):
        if not image_paths_or_urls:
            return []

        payload_input = []
        for img_ref in image_paths_or_urls:
            if isinstance(img_ref, str):
                # Check if it's a base64 string already (simple check)
                if len(img_ref) > 100 and not img_ref.startswith(('http', '/')):
                     payload_input.append({"image": img_ref}) # Assume it's base64
                else:
                    base64_image = self._encode_image_to_base64(img_ref)
                    if base64_image:
                        payload_input.append({"image": base64_image})
            else:
                print(f"Warning: Skipping invalid image input: {img_ref}")

        if not payload_input:
            return []

        data = {
            "model": self.model,
            "input": payload_input
        }
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status()
            results = response.json()
            if "data" in results:
                return [item["embedding"] for item in results["data"]]
            else:
                print(f"Error in API response (image embedding): {results}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Request failed (image embedding): {e}")
            return []
        except json.JSONDecodeError:
            print(f"Failed to decode JSON response (image embedding): {response.text}")
            return []

    def embed_multimodal(self, inputs):
        """
        Embeds a list of multimodal inputs. Each input in the list should be a dictionary
        with either a 'text' key or an 'image' key (path/URL or base64 string).
        Example: [{'text': 'hello'}, {'image': 'path/to/img.jpg'}]
        """
        if not inputs:
            return []

        payload_input = []
        for item in inputs:
            if "text" in item and isinstance(item["text"], str):
                payload_input.append({"text": item["text"]})
            elif "image" in item and isinstance(item["image"], str):
                if len(item["image"]) > 100 and not item["image"].startswith(('http', '/')):
                    payload_input.append({"image": item["image"]}) # Assume it's base64
                else:
                    base64_image = self._encode_image_to_base64(item["image"])
                    if base64_image:
                        payload_input.append({"image": base64_image})
            else:
                print(f"Warning: Skipping invalid multimodal input: {item}")
        
        if not payload_input:
            return []

        data = {
            "model": self.model,
            "input": payload_input
        }
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
            response.raise_for_status()
            results = response.json()
            # The API returns a list of embeddings in the 'data' field
            if "data" in results and isinstance(results["data"], list):
                embeddings = []
                for emb_data in results["data"]:
                    if "embedding" in emb_data:
                        embeddings.append(emb_data["embedding"])
                    else:
                        print(f"Warning: Embedding not found in item: {emb_data}")
                        embeddings.append(None) # Or handle error as appropriate
                return embeddings
            else:
                print(f"Error or unexpected format in API response (multimodal embedding): {results}")
                return [None] * len(payload_input) # Return None for each input on error
        except requests.exceptions.RequestException as e:
            print(f"Request failed (multimodal embedding): {e}")
            return [None] * len(payload_input)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON response (multimodal embedding): {response.text}")
            return [None] * len(payload_input)

# Example Usage (for testing this file directly):
if __name__ == '__main__':
    # 请替换为您的 AiHubMix 密钥
    api_key = "sk-VttLrtXYMsKnEs4CD01eA4D39575463486Ef5d7e2a063095" 
    # 如果您想从环境变量加载API密钥，可以使用以下方式：
    # import os
    # api_key = os.getenv("AIHUBMIX_API_KEY")
    # if not api_key:
    #     raise ValueError("AIHUBMIX_API_KEY environment variable not set.")

    embedder = JinaCLIPEmbeddings(api_key=api_key)

    # Test text embedding
    texts_to_embed = [
        "A beautiful sunset over the beach",
        "海滩上美丽的日落"
    ]
    text_embeddings = embedder.embed_texts(texts_to_embed)
    print("Text Embeddings:")
    if text_embeddings:
        for i, emb in enumerate(text_embeddings):
            print(f"  Text {i+1}: Embedding dim {len(emb)}") # First 5 elements: {emb[:5]}
    else:
        print("  Failed to get text embeddings.")
    print("-"*20)

    # Test image embedding
    images_to_embed = [
        "https://i.ibb.co/nQNGqL0/beach1.jpg", # URL
        # "path/to/your/local/image.jpg" # Example for local file
        # A base64 encoded image string (very small placeholder)
        "R0lGODlhEAAQAMQAAORHHOVSKudfOulrSOp3WOyDZu6QdvCchPGolfO0o/XBs/fNwfjZ0frl3/zy7////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAkAABAALAAAAAAQABAAAAVVICSOZGlCQAosJ6mu7fiyZeKqNKToQGDsM8hBADgUXoGAiqhSvp5QAnQKGIgUhwFUYLCVDFCrKUE1lBavAViFIDlTImbKC5Gm2hB0SlBCBMQiB0UjIQA7"
    ]
    # Create a dummy image file for local testing if you don't have one
    # try:
    #     from PIL import Image
    #     dummy_img = Image.new('RGB', (60, 30), color = 'red')
    #     dummy_img.save("dummy_image.png")
    #     images_to_embed.append("dummy_image.png")
    # except ImportError:
    #     print("Pillow not installed, skipping local dummy image test.")
        
    image_embeddings = embedder.embed_images(images_to_embed)
    print("Image Embeddings:")
    if image_embeddings:
        for i, emb in enumerate(image_embeddings):
            if emb:
                print(f"  Image {i+1}: Embedding dim {len(emb)}") # First 5 elements: {emb[:5]}
            else:
                print(f"  Image {i+1}: Failed to get embedding.")
    else:
        print("  Failed to get any image embeddings.")
    print("-"*20)

    # Test multimodal embedding
    multimodal_inputs = [
        {"text": "A beautiful sunset over the beach"},
        {"image": "https://i.ibb.co/nQNGqL0/beach1.jpg"},
        {"text": "海滩上美丽的日落"},
        {"image": "R0lGODlhEAAQAMQAAORHHOVSKudfOulrSOp3WOyDZu6QdvCchPGolfO0o/XBs/fNwfjZ0frl3/zy7////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAkAABAALAAAAAAQABAAAAVVICSOZGlCQAosJ6mu7fiyZeKqNKToQGDsM8hBADgUXoGAiqhSvp5QAnQKGIgUhwFUYLCVDFCrKUE1lBavAViFIDlTImbKC5Gm2hB0SlBCBMQiB0UjIQA7"}
    ]
    multimodal_embeddings = embedder.embed_multimodal(multimodal_inputs)
    print("Multimodal Embeddings:")
    if multimodal_embeddings:
        for i, emb in enumerate(multimodal_embeddings):
            if emb:
                print(f"  Input {i+1}: Embedding dim {len(emb)}") # First 5 elements: {emb[:5]}
            else:
                print(f"  Input {i+1}: Failed to get embedding.")
    else:
        print("  Failed to get any multimodal embeddings.")

    # Example with a local file (create a dummy file first)
    # with open("test_image.png", "wb") as f:
    #     # A minimal valid PNG
    #     f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="))
    # local_image_test = embedder.embed_multimodal([{"image": "test_image.png"}])
    # print("\nLocal Image Test Embedding:")
    # if local_image_test and local_image_test[0]:
    #     print(f"  Input 1: Embedding dim {len(local_image_test[0])}")
    # else:
    #     print("  Failed to get local image embedding.")
    # import os
    # os.remove("test_image.png") # Clean up dummy file
    # if os.path.exists("dummy_image.png"): os.remove("dummy_image.png")