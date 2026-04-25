from pillow_heif import register_heif_opener

register_heif_opener()

from pipeline.preprocessor import preprocess, preprocess_pil
from pipeline.extractor import FeatureExtractor
from pipeline.database import ReferenceDatabase
from pipeline.pdf_extractor import extract_images_from_pdf, extract_all_pdfs
from pipeline.identifier import (
    build_reference_database,
    load_database,
    identify_from_path,
    identify_from_pil,
)

__all__ = [
    "preprocess", "preprocess_pil",
    "FeatureExtractor",
    "ReferenceDatabase",
    "extract_images_from_pdf", "extract_all_pdfs",
    "build_reference_database", "load_database",
    "identify_from_path", "identify_from_pil",
]
