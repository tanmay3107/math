# document_processor.py
import re

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        """
        Initializes the document processor.
        :param chunk_size: The maximum number of characters/words per chunk.
        :param chunk_overlap: How many characters/words should overlap between chunks to maintain context.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text):
        """
        Removes messy formatting, multiple spaces, and weird newline characters.
        """
        # Replace multiple newlines with a single space
        text = re.sub(r'\n+', ' ', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text):
        """
        Splits a massive string of text into overlapping chunks.
        """
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split(' ')
        
        chunks = []
        
        # Sliding window over the words array
        i = 0
        while i < len(words):
            # Extract the chunk
            chunk_end = min(i + self.chunk_size, len(words))
            chunk = words[i:chunk_end]
            
            # Join words back into a single string
            chunks.append(' '.join(chunk))
            
            # Move the window forward, but step back by the overlap amount
            i += (self.chunk_size - self.chunk_overlap)
            
            # Break condition if the overlap pushes us into an infinite loop at the end
            if i >= len(words):
                break
                
        return chunks

# --- Quick Test ---
if __name__ == "__main__":
    print("⚙️ Initializing Document Chunking Engine...")
    
    # Let's simulate a highly technical document (e.g., an internal company wiki)
    dummy_document = """
    EdgeVision V2.0 Deployment Guide.
    
    The EdgeVision architecture utilizes a decoupled microservice pattern. 
    The camera client runs entirely on low-power IoT devices using OpenCV. 
    When motion is detected, it captures a frame and sends it to the central inference server.
    
    The inference server requires an NVIDIA GPU with at least 4GB of VRAM. 
    It runs a 2-billion parameter Vision-Language Model loaded in FP16 precision. 
    If the system detects unauthorized personnel, it logs the event into the SQLAlchemy database.
    
    Security protocols require all API endpoints to be secured via JWT authentication.
    """
    
    # We will use an aggressively small chunk size just to prove the overlap works
    processor = DocumentProcessor(chunk_size=20, chunk_overlap=5)
    
    print("\n🔪 Slicing document into overlapping chunks...")
    chunks = processor.chunk_text(dummy_document)
    
    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx + 1} (Length: {len(chunk.split())} words) ---")
        print(chunk)
        
    print("\n✅ Notice how the end of Chunk 1 overlaps perfectly with the beginning of Chunk 2 to prevent data loss!")