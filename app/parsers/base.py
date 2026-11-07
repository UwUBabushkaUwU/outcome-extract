from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    location: str  # "[Page 3]" or "[Sheet1!A1:H40]"

@dataclass
class Document:
    chunks: list[Chunk]
    images: list[tuple[int, bytes]] = field(default_factory=list)  # (page no, PNG) for scanned pages