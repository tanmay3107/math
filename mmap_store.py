import mmap
import struct
import os
from typing import List, Dict, Tuple, Optional

class MMapVectorStore:
    """
    Disk-backed, memory-mapped vector store for scalable zero-dependency vector retrieval.
    Binary layout per record: [4 bytes header length] + [JSON header (id, metadata)] + [dim * 4 bytes float32 vector]
    """

    def __init__(self, filename: str, dim: int):
        self.filename = filename
        self.dim = dim
        self.file_obj = open(self.filename, "a+b")
        self._ensure_file_size()

    def _ensure_file_size(self):
        self.file_obj.seek(0, os.SEEK_END)
        self.file_size = self.file_obj.tell()
        if self.file_size > 0:
            self.mm = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            self.mm = None

    def append(self, doc_id: str, vector: List[float], metadata: Optional[Dict] = None) -> int:
        """Appends a vector record to disk and updates memory map."""
        if len(vector) != self.dim:
            raise ValueError(f"Vector dimension {len(vector)} does not match store dimension {self.dim}")

        import json
        header_bytes = json.dumps({"id": doc_id, "metadata": metadata or {}}).encode("utf-8")
        header_len = len(header_bytes)
        vector_bytes = struct.pack(f"{self.dim}f", *vector)

        # Move to end of file and write record payload
        self.file_obj.seek(0, os.SEEK_END)
        offset = self.file_obj.tell()
        self.file_obj.write(struct.pack("I", header_len))
        self.file_obj.write(header_bytes)
        self.file_obj.write(vector_bytes)
        self.file_obj.flush()

        if self.mm:
            self.mm.close()
        self.mm = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        self.file_size = self.mm.size()
        return offset

    def read_at(self, offset: int) -> Tuple[str, List[float], Dict]:
        """Reads vector record directly from memory-mapped file without loading full dataset into RAM."""
        if not self.mm or offset >= self.file_size:
            raise IndexError("Offset out of bounds.")

        header_len = struct.unpack("I", self.mm[offset:offset+4])[0]
        curr = offset + 4

        import json
        header_data = json.loads(self.mm[curr:curr+header_len].decode("utf-8"))
        curr += header_len

        vector = list(struct.unpack(f"{self.dim}f", self.mm[curr:curr + (self.dim * 4)]))
        return header_data["id"], vector, header_data["metadata"]

    def close(self):
        if self.mm:
            self.mm.close()
        if self.file_obj and not self.file_obj.closed:
            self.file_obj.close()