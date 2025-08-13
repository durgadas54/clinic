"""
Production-Grade PDF Parsing System for RAG
Implements strategy pattern for different PDF types with robust multimodal content extraction
"""

import io
import json
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import logging
from pathlib import Path

# Core libraries
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import cv2
import numpy as np

# For content extraction and chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content that can be extracted from PDF"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    SCANNED_PAGE = "scanned_page"


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates"""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class ContentElement:
    """Represents a piece of content extracted from PDF"""
    content_type: ContentType
    content: str
    bbox: BoundingBox
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class PageLayout:
    """Represents the layout structure of a PDF page"""
    page_number: int
    elements: List[ContentElement] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    is_scanned: bool = False
    page_dimensions: Tuple[float, float] = (0, 0)


@dataclass
class DocumentStructure:
    """Represents the complete document structure"""
    pages: List[PageLayout] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_type: str = "generic"


class VisionLLMInterface:
    """Interface for Vision-Language Model API calls"""

    def __init__(self, api_client=None):
        self.api_client = api_client

    def extract_text_from_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Extract text from image using Vision-Language Model
        This is a placeholder - implement based on your specific LLM API
        """
        if prompt is None:
            prompt = (
                "Extract all text from this image. Maintain the original layout and structure. "
                "If this is a table, preserve the tabular format. "
                "Return only the extracted text without any additional commentary."
            )

        # Convert PIL Image to base64 for API call
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Placeholder for actual API call
        # Replace with your specific LLM API implementation
        try:
            # Example API call structure:
            # response = self.api_client.vision_chat({
            #     "messages": [{"role": "user", "content": prompt}],
            #     "image": image_base64
            # })
            # return response.get("content", "")

            logger.warning("VisionLLMInterface not implemented - returning empty text")
            return ""
        except Exception as e:
            logger.error(f"Error in VisionLLM text extraction: {e}")
            return ""

    def analyze_image_content(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image content to determine type and extract metadata"""
        prompt = (
            "Analyze this image and return a JSON response with the following structure:\n"
            "{\n"
            "  'content_type': 'table' | 'chart' | 'diagram' | 'text' | 'mixed',\n"
            "  'description': 'brief description of the content',\n"
            "  'has_text': true/false,\n"
            "  'complexity': 'low' | 'medium' | 'high'\n"
            "}"
        )

        try:
            # Placeholder for actual API call
            return {
                "content_type": "mixed",
                "description": "Image content",
                "has_text": True,
                "complexity": "medium"
            }
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return {"content_type": "unknown", "has_text": False}


class PDFParsingStrategy(ABC):
    """Abstract base class for PDF parsing strategies"""

    def __init__(self, vision_llm: VisionLLMInterface):
        self.vision_llm = vision_llm

    @abstractmethod
    def parse_document(self, pdf_path: str) -> DocumentStructure:
        """Parse the entire PDF document"""
        pass

    @abstractmethod
    def identify_sections(self, page_layout: PageLayout) -> List[Dict[str, Any]]:
        """Identify sections within a page layout"""
        pass

    @abstractmethod
    def should_merge_elements(self, elem1: ContentElement, elem2: ContentElement) -> bool:
        """Determine if two elements should be merged"""
        pass


class GenericPDFParser(PDFParsingStrategy):
    """Generic PDF parsing strategy for standard documents"""

    def __init__(self, vision_llm: VisionLLMInterface):
        super().__init__(vision_llm)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )

    def parse_document(self, pdf_path: str) -> DocumentStructure:
        """Parse PDF document with generic strategy"""
        doc_structure = DocumentStructure()

        try:
            pdf_doc = fitz.open(pdf_path)
            doc_structure.metadata = {
                "title": pdf_doc.metadata.get("title", ""),
                "author": pdf_doc.metadata.get("author", ""),
                "page_count": pdf_doc.page_count,
                "file_path": pdf_path
            }

            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                page_layout = self._parse_page(page, page_num + 1)
                doc_structure.pages.append(page_layout)

            pdf_doc.close()

        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")

        return doc_structure

    def _parse_page(self, page: fitz.Page, page_number: int) -> PageLayout:
        """Parse a single PDF page"""
        page_layout = PageLayout(
            page_number=page_number,
            page_dimensions=(page.rect.width, page.rect.height)
        )

        # Check if page is primarily scanned
        if self._is_scanned_page(page):
            page_layout.is_scanned = True
            page_layout = self._parse_scanned_page(page, page_layout)
        else:
            page_layout = self._parse_digital_page(page, page_layout)

        # Identify sections after content extraction
        page_layout.sections = self.identify_sections(page_layout)

        return page_layout

    def _is_scanned_page(self, page: fitz.Page) -> bool:
        """Determine if a page is primarily scanned content"""
        # Extract text directly
        text = page.get_text().strip()

        # Get page dimensions
        rect = page.rect
        page_area = rect.width * rect.height

        # Get image list
        image_list = page.get_images()

        # Calculate criteria
        has_minimal_text = len(text) < 50
        has_large_images = any(
            self._get_image_coverage(page, img) > 0.5
            for img in image_list
        )

        # Consider it scanned if minimal text and large image coverage
        return has_minimal_text and has_large_images and len(image_list) > 0

    def _get_image_coverage(self, page: fitz.Page, img_info: tuple) -> float:
        """Calculate what percentage of page area an image covers"""
        try:
            img_rect = page.get_image_bbox(img_info)
            if img_rect:
                img_area = img_rect.width * img_rect.height
                page_area = page.rect.width * page.rect.height
                return img_area / page_area if page_area > 0 else 0
        except:
            pass
        return 0

    def _parse_scanned_page(self, page: fitz.Page, page_layout: PageLayout) -> PageLayout:
        """Parse a scanned page using vision-language model"""
        try:
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            # Preprocess image for better OCR
            image = self._preprocess_image(image)

            # Extract text using vision-language model
            extracted_text = self.vision_llm.extract_text_from_image(image)

            if extracted_text:
                # Create content element for the entire scanned page
                content_element = ContentElement(
                    content_type=ContentType.SCANNED_PAGE,
                    content=extracted_text,
                    bbox=BoundingBox(0, 0, page.rect.width, page.rect.height),
                    page_number=page_layout.page_number,
                    metadata={"extraction_method": "vision_llm", "is_scanned": True}
                )
                page_layout.elements.append(content_element)

        except Exception as e:
            logger.error(f"Error parsing scanned page {page_layout.page_number}: {e}")

        return page_layout

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Convert PIL to OpenCV format
        img_array = np.array(image)

        # Convert to grayscale if not already
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array

        # Apply denoising
        img_denoised = cv2.fastNlMeansDenoising(img_gray)

        # Apply adaptive thresholding for better contrast
        img_thresh = cv2.adaptiveThreshold(
            img_denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Convert back to PIL
        return Image.fromarray(img_thresh)

    def _parse_digital_page(self, page: fitz.Page, page_layout: PageLayout) -> PageLayout:
        """Parse a digital PDF page with native text extraction"""
        # Extract text blocks with position information
        blocks = page.get_text("dict")["blocks"]

        # Extract tables
        page_layout = self._extract_tables(page, page_layout)

        # Extract images
        page_layout = self._extract_images(page, page_layout)

        # Process text blocks
        for block in blocks:
            if "lines" in block:  # Text block
                text_content = self._extract_text_from_block(block)
                if text_content.strip():
                    bbox = BoundingBox(
                        block["bbox"][0], block["bbox"][1],
                        block["bbox"][2], block["bbox"][3]
                    )

                    content_element = ContentElement(
                        content_type=ContentType.TEXT,
                        content=text_content,
                        bbox=bbox,
                        page_number=page_layout.page_number,
                        metadata={
                            "font_info": self._get_font_info(block),
                            "extraction_method": "native"
                        }
                    )
                    page_layout.elements.append(content_element)

        return page_layout

    def _extract_text_from_block(self, block: Dict) -> str:
        """Extract text from a text block while preserving structure"""
        lines = []
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            if line_text.strip():
                lines.append(line_text.strip())
        return "\n".join(lines)

    def _get_font_info(self, block: Dict) -> Dict[str, Any]:
        """Extract font information from text block"""
        font_info = {"fonts": [], "sizes": []}
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_info["fonts"].append(span.get("font", ""))
                font_info["sizes"].append(span.get("size", 0))
        return font_info

    def _extract_tables(self, page: fitz.Page, page_layout: PageLayout) -> PageLayout:
        """Extract tables from page - basic implementation"""
        # This is a simplified table detection
        # For production, consider using specialized libraries like camelot-py
        tables = page.find_tables()

        for table in tables:
            try:
                # Extract table as pandas DataFrame
                df = table.extract()
                if df is not None and not df.empty:
                    # Convert DataFrame to string representation
                    table_content = df.to_string(index=False)

                    bbox = BoundingBox(
                        table.bbox[0], table.bbox[1],
                        table.bbox[2], table.bbox[3]
                    )

                    content_element = ContentElement(
                        content_type=ContentType.TABLE,
                        content=table_content,
                        bbox=bbox,
                        page_number=page_layout.page_number,
                        metadata={
                            "rows": len(df),
                            "columns": len(df.columns),
                            "extraction_method": "pymupdf"
                        }
                    )
                    page_layout.elements.append(content_element)
            except Exception as e:
                logger.warning(f"Error extracting table: {e}")

        return page_layout

    def _extract_images(self, page: fitz.Page, page_layout: PageLayout) -> PageLayout:
        """Extract images from page"""
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                # Get image
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)

                if pix.n - pix.alpha < 4:  # Not CMYK
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))

                    # Analyze image content
                    analysis = self.vision_llm.analyze_image_content(image)

                    # Get image description if it contains text or is complex
                    description = ""
                    if analysis.get("has_text", False):
                        description = self.vision_llm.extract_text_from_image(
                            image,
                            "Describe this image and extract any text it contains."
                        )

                    # Get image bounds
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        bbox = BoundingBox(
                            img_rects[0].x0, img_rects[0].y0,
                            img_rects[0].x1, img_rects[0].y1
                        )
                    else:
                        bbox = BoundingBox(0, 0, pix.width, pix.height)

                    content_element = ContentElement(
                        content_type=ContentType.IMAGE,
                        content=description,
                        bbox=bbox,
                        page_number=page_layout.page_number,
                        metadata={
                            "image_analysis": analysis,
                            "image_index": img_index,
                            "extraction_method": "pymupdf"
                        }
                    )
                    page_layout.elements.append(content_element)

                pix = None

            except Exception as e:
                logger.warning(f"Error extracting image {img_index}: {e}")

        return page_layout

    def identify_sections(self, page_layout: PageLayout) -> List[Dict[str, Any]]:
        """Identify sections based on layout and content analysis"""
        sections = []
        current_section = None

        # Sort elements by position (top to bottom, left to right)
        sorted_elements = sorted(
            page_layout.elements,
            key=lambda x: (x.bbox.y0, x.bbox.x0)
        )

        for element in sorted_elements:
            # Basic section detection based on font size and position
            if element.content_type == ContentType.TEXT:
                font_info = element.metadata.get("font_info", {})
                sizes = font_info.get("sizes", [])
                avg_size = sum(sizes) / len(sizes) if sizes else 12

                # Consider it a heading if font size is larger
                is_heading = avg_size > 14

                if is_heading or current_section is None:
                    if current_section:
                        sections.append(current_section)

                    current_section = {
                        "section_type": "heading" if is_heading else "content",
                        "title": element.content[:100] if is_heading else f"Section {len(sections) + 1}",
                        "elements": [element],
                        "bbox": element.bbox,
                        "page_number": element.page_number
                    }
                else:
                    current_section["elements"].append(element)
                    # Expand bounding box
                    current_section["bbox"] = self._merge_bboxes(
                        current_section["bbox"], element.bbox
                    )
            else:
                if current_section:
                    current_section["elements"].append(element)
                    current_section["bbox"] = self._merge_bboxes(
                        current_section["bbox"], element.bbox
                    )

        if current_section:
            sections.append(current_section)

        return sections

    def _merge_bboxes(self, bbox1: BoundingBox, bbox2: BoundingBox) -> BoundingBox:
        """Merge two bounding boxes"""
        return BoundingBox(
            min(bbox1.x0, bbox2.x0),
            min(bbox1.y0, bbox2.y0),
            max(bbox1.x1, bbox2.x1),
            max(bbox1.y1, bbox2.y1)
        )

    def should_merge_elements(self, elem1: ContentElement, elem2: ContentElement) -> bool:
        """Determine if two elements should be merged during chunking"""
        # Merge if elements are close and same type
        if elem1.content_type == elem2.content_type == ContentType.TEXT:
            vertical_distance = abs(elem1.bbox.y1 - elem2.bbox.y0)
            return vertical_distance < 20  # Merge if less than 20 points apart
        return False


class ShareholderMomentumParser(PDFParsingStrategy):
    """Specialized parser for Shareholder Momentum documents"""

    def __init__(self, vision_llm: VisionLLMInterface):
        super().__init__(vision_llm)

    def parse_document(self, pdf_path: str) -> DocumentStructure:
        """Parse shareholder momentum document with specialized logic"""
        # Start with generic parsing
        generic_parser = GenericPDFParser(self.vision_llm)
        doc_structure = generic_parser.parse_document(pdf_path)
        doc_structure.document_type = "shareholder_momentum"

        # Apply shareholder-specific processing
        self._identify_shareholder_sections(doc_structure)

        return doc_structure

    def _identify_shareholder_sections(self, doc_structure: DocumentStructure):
        """Identify sections specific to shareholder documents"""
        # Look for common sections in shareholder documents
        section_keywords = [
            "executive summary", "financial highlights", "key metrics",
            "shareholder returns", "dividend information", "stock performance"
        ]

        for page in doc_structure.pages:
            for section in page.sections:
                section_text = " ".join(elem.content.lower() for elem in section["elements"])

                for keyword in section_keywords:
                    if keyword in section_text:
                        section["shareholder_section"] = keyword.replace(" ", "_")
                        break

    def identify_sections(self, page_layout: PageLayout) -> List[Dict[str, Any]]:
        """Use generic section identification with shareholder-specific enhancements"""
        generic_parser = GenericPDFParser(self.vision_llm)
        return generic_parser.identify_sections(page_layout)

    def should_merge_elements(self, elem1: ContentElement, elem2: ContentElement) -> bool:
        """Shareholder-specific merging logic"""
        generic_parser = GenericPDFParser(self.vision_llm)
        return generic_parser.should_merge_elements(elem1, elem2)


class TermSheetParser(PDFParsingStrategy):
    """Specialized parser for Term Sheet documents"""

    def __init__(self, vision_llm: VisionLLMInterface):
        super().__init__(vision_llm)

    def parse_document(self, pdf_path: str) -> DocumentStructure:
        """Parse term sheet with specialized logic"""
        generic_parser = GenericPDFParser(self.vision_llm)
        doc_structure = generic_parser.parse_document(pdf_path)
        doc_structure.document_type = "term_sheet"

        # Apply term sheet specific processing
        self._identify_term_sections(doc_structure)

        return doc_structure

    def _identify_term_sections(self, doc_structure: DocumentStructure):
        """Identify sections specific to term sheets"""
        term_keywords = [
            "investment amount", "valuation", "liquidation preference",
            "anti-dilution", "board composition", "voting rights",
            "drag along", "tag along", "representations"
        ]

        for page in doc_structure.pages:
            for section in page.sections:
                section_text = " ".join(elem.content.lower() for elem in section["elements"])

                for keyword in term_keywords:
                    if keyword in section_text:
                        section["term_section"] = keyword.replace(" ", "_")
                        break

    def identify_sections(self, page_layout: PageLayout) -> List[Dict[str, Any]]:
        """Use generic section identification"""
        generic_parser = GenericPDFParser(self.vision_llm)
        return generic_parser.identify_sections(page_layout)

    def should_merge_elements(self, elem1: ContentElement, elem2: ContentElement) -> bool:
        """Term sheet specific merging logic"""
        generic_parser = GenericPDFParser(self.vision_llm)
        return generic_parser.should_merge_elements(elem1, elem2)


class PDFParserFactory:
    """Factory class to create appropriate PDF parser based on document type"""

    @staticmethod
    def create_parser(document_type: str, vision_llm: VisionLLMInterface) -> PDFParsingStrategy:
        """Create parser based on document type"""
        parsers = {
            "shareholder_momentum": ShareholderMomentumParser,
            "term_sheet": TermSheetParser,
            "generic": GenericPDFParser
        }

        parser_class = parsers.get(document_type.lower(), GenericPDFParser)
        return parser_class(vision_llm)


class IntelligentChunker:
    """Intelligent chunking system for processed PDF content"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def create_chunks(self, doc_structure: DocumentStructure) -> List[Document]:
        """Create intelligent chunks from document structure"""
        chunks = []

        for page in doc_structure.pages:
            # Process sections within page
            for section in page.sections:
                section_chunks = self._create_section_chunks(section, doc_structure.metadata)
                chunks.extend(section_chunks)

            # Process standalone elements not in sections
            standalone_elements = self._get_standalone_elements(page)
            for element in standalone_elements:
                element_chunks = self._create_element_chunks(element, doc_structure.metadata)
                chunks.extend(element_chunks)

        return chunks

    def _create_section_chunks(self, section: Dict[str, Any], doc_metadata: Dict[str, Any]) -> List[Document]:
        """Create chunks from a document section"""
        chunks = []

        # Combine section content
        section_content = []
        tables = []
        images = []

        for element in section["elements"]:
            if element.content_type == ContentType.TEXT or element.content_type == ContentType.SCANNED_PAGE:
                section_content.append(element.content)
            elif element.content_type == ContentType.TABLE:
                tables.append(element.content)
            elif element.content_type == ContentType.IMAGE:
                images.append(element.content)

        # Create text chunks
        combined_text = "\n\n".join(section_content)
        if combined_text.strip():
            text_chunks = self.text_splitter.split_text(combined_text)

            for i, chunk_text in enumerate(text_chunks):
                metadata = {
                    **doc_metadata,
                    "page_number": section["page_number"],
                    "section_title": section.get("title", ""),
                    "section_type": section.get("section_type", "content"),
                    "chunk_index": i,
                    "content_type": "text",
                    "extraction_method": "section_based"
                }

                # Add specialized metadata if available
                if "shareholder_section" in section:
                    metadata["shareholder_section"] = section["shareholder_section"]
                if "term_section" in section:
                    metadata["term_section"] = section["term_section"]

                chunks.append(Document(page_content=chunk_text, metadata=metadata))

        # Create separate chunks for tables
        for i, table_content in enumerate(tables):
            metadata = {
                **doc_metadata,
                "page_number": section["page_number"],
                "section_title": section.get("title", ""),
                "content_type": "table",
                "table_index": i,
                "extraction_method": "table_specific"
            }
            chunks.append(Document(page_content=table_content, metadata=metadata))

        # Create separate chunks for images with descriptions
        for i, image_content in enumerate(images):
            if image_content.strip():  # Only if we have description
                metadata = {
                    **doc_metadata,
                    "page_number": section["page_number"],
                    "section_title": section.get("title", ""),
                    "content_type": "image",
                    "image_index": i,
                    "extraction_method": "image_description"
                }
                chunks.append(Document(page_content=image_content, metadata=metadata))

        return chunks

    def _get_standalone_elements(self, page: PageLayout) -> List[ContentElement]:
        """Get elements not included in any section"""
        section_elements = set()
        for section in page.sections:
            for element in section["elements"]:
                section_elements.add(id(element))

        return [elem for elem in page.elements if id(elem) not in section_elements]

    def _create_element_chunks(self, element: ContentElement, doc_metadata: Dict[str, Any]) -> List[Document]:
        """Create chunks from a standalone element"""
        chunks = []

        if element.content.strip():
            if element.content_type in [ContentType.TEXT, ContentType.SCANNED_PAGE]:
                # Split text elements
                text_chunks = self.text_splitter.split_text(element.content)
                for i, chunk_text in enumerate(text_chunks):
                    metadata = {
                        **doc_metadata,
                        "page_number": element.page_number,
                        "content_type": element.content_type.value,
                        "chunk_index": i,
                        "bbox": {
                            "x0": element.bbox.x0, "y0": element.bbox.y0,
                            "x1": element.bbox.x1, "y1": element.bbox.y1
                        },
                        "extraction_method": element.metadata.get("extraction_method", "unknown")
                    }
                    chunks.append(Document(page_content=chunk_text, metadata=metadata))
            else:
                # Single chunk for tables and images
                metadata = {
                    **doc_metadata,
                    "page_number": element.page_number,
                    "content_type": element.content_type.value,
                    "bbox": {
                        "x0": element.bbox.x0, "y0": element.bbox.y0,
                        "x1": element.bbox.x1, "y1": element.bbox.y1
                    },
                    "extraction_method": element.metadata.get("extraction_method", "unknown")
                }
                chunks.append(Document(page_content=element.content, metadata=metadata))

        return chunks


class PDFProcessingPipeline:
    """Main pipeline for PDF processing with strategy pattern"""

    def __init__(self, vision_llm_client=None, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.vision_llm = VisionLLMInterface(vision_llm_client)
        self.chunker = IntelligentChunker(chunk_size, chunk_overlap)
        self.parser_cache = {}

    def process_pdf(
        self,
        pdf_path: str,
        document_type: str = "generic",
        enable_caching: bool = True
    ) -> Tuple[DocumentStructure, List[Document]]:
        """
        Process PDF end-to-end: parse structure and create chunks

        Args:
            pdf_path: Path to PDF file
            document_type: Type of document (generic, shareholder_momentum, term_sheet)
            enable_caching: Whether to cache parser instances

        Returns:
            Tuple of (DocumentStructure, List of Document chunks)
        """
        try:
            logger.info(f"Processing PDF: {pdf_path} as {document_type}")

            # Get appropriate parser
            parser = self._get_parser(document_type, enable_caching)

            # Parse document structure
            doc_structure = parser.parse_document(pdf_path)
            logger.info(f"Parsed {len(doc_structure.pages)} pages")

            # Create intelligent chunks
            chunks = self.chunker.create_chunks(doc_structure)
            logger.info(f"Created {len(chunks)} chunks")

            return doc_structure, chunks

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise

    def _get_parser(self, document_type: str, enable_caching: bool) -> PDFParsingStrategy:
        """Get parser instance with optional caching"""
        if enable_caching and document_type in self.parser_cache:
            return self.parser_cache[document_type]

        parser = PDFParserFactory.create_parser(document_type, self.vision_llm)

        if enable_caching:
            self.parser_cache[document_type] = parser

        return parser

    def extract_metadata_summary(self, doc_structure: DocumentStructure) -> Dict[str, Any]:
        """Extract comprehensive metadata summary from document structure"""
        summary = {
            "document_type": doc_structure.document_type,
            "total_pages": len(doc_structure.pages),
            "scanned_pages": sum(1 for page in doc_structure.pages if page.is_scanned),
            "total_elements": sum(len(page.elements) for page in doc_structure.pages),
            "content_distribution": {
                "text_elements": 0,
                "table_elements": 0,
                "image_elements": 0,
                "scanned_elements": 0
            },
            "sections_by_page": {},
            "extraction_methods": set(),
            "document_metadata": doc_structure.metadata
        }

        # Analyze content distribution
        for page in doc_structure.pages:
            summary["sections_by_page"][page.page_number] = len(page.sections)

            for element in page.elements:
                content_type = element.content_type.value
                if content_type == "text":
                    summary["content_distribution"]["text_elements"] += 1
                elif content_type == "table":
                    summary["content_distribution"]["table_elements"] += 1
                elif content_type == "image":
                    summary["content_distribution"]["image_elements"] += 1
                elif content_type == "scanned_page":
                    summary["content_distribution"]["scanned_elements"] += 1

                # Track extraction methods
                extraction_method = element.metadata.get("extraction_method", "unknown")
                summary["extraction_methods"].add(extraction_method)

        summary["extraction_methods"] = list(summary["extraction_methods"])
        return summary

    def validate_chunks(self, chunks: List[Document]) -> Dict[str, Any]:
        """Validate generated chunks and return quality metrics"""
        validation_results = {
            "total_chunks": len(chunks),
            "chunks_by_type": {},
            "chunks_by_page": {},
            "average_chunk_size": 0,
            "empty_chunks": 0,
            "oversized_chunks": 0,
            "quality_score": 0.0,
            "warnings": []
        }

        total_size = 0

        for chunk in chunks:
            content_length = len(chunk.page_content)
            total_size += content_length

            # Track by content type
            content_type = chunk.metadata.get("content_type", "unknown")
            validation_results["chunks_by_type"][content_type] = \
                validation_results["chunks_by_type"].get(content_type, 0) + 1

            # Track by page
            page_num = chunk.metadata.get("page_number", 0)
            validation_results["chunks_by_page"][page_num] = \
                validation_results["chunks_by_page"].get(page_num, 0) + 1

            # Check for empty chunks
            if content_length == 0:
                validation_results["empty_chunks"] += 1
                validation_results["warnings"].append(f"Empty chunk found on page {page_num}")

            # Check for oversized chunks (>2000 chars)
            if content_length > 2000:
                validation_results["oversized_chunks"] += 1
                validation_results["warnings"].append(f"Oversized chunk ({content_length} chars) on page {page_num}")

        # Calculate averages and quality score
        if chunks:
            validation_results["average_chunk_size"] = total_size / len(chunks)

            # Simple quality score based on chunk distribution and issues
            quality_score = 1.0
            if validation_results["empty_chunks"] > 0:
                quality_score -= 0.2
            if validation_results["oversized_chunks"] > len(chunks) * 0.1:  # More than 10% oversized
                quality_score -= 0.1
            if len(validation_results["chunks_by_type"]) < 2:  # Low content diversity
                quality_score -= 0.1

            validation_results["quality_score"] = max(0.0, quality_score)

        return validation_results

    def export_chunks_to_json(self, chunks: List[Document], output_path: str):
        """Export chunks to JSON format for vector database ingestion"""
        export_data = []

        for i, chunk in enumerate(chunks):
            chunk_data = {
                "id": f"chunk_{i}",
                "content": chunk.page_content,
                "metadata": chunk.metadata
            }
            export_data.append(chunk_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(chunks)} chunks to {output_path}")

    def get_processing_statistics(self, doc_structure: DocumentStructure) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        stats = {
            "document_info": {
                "type": doc_structure.document_type,
                "pages": len(doc_structure.pages),
                "title": doc_structure.metadata.get("title", "Unknown")
            },
            "processing_breakdown": {
                "scanned_pages": 0,
                "digital_pages": 0,
                "mixed_pages": 0
            },
            "content_analysis": {
                "text_blocks": 0,
                "tables": 0,
                "images": 0,
                "total_sections": 0
            },
            "extraction_methods": {
                "native_extraction": 0,
                "vision_llm": 0,
                "hybrid": 0
            },
            "quality_indicators": {
                "pages_with_sections": 0,
                "average_elements_per_page": 0,
                "pages_with_mixed_content": 0
            }
        }

        total_elements = 0

        for page in doc_structure.pages:
            # Page type analysis
            if page.is_scanned:
                stats["processing_breakdown"]["scanned_pages"] += 1
            else:
                stats["processing_breakdown"]["digital_pages"] += 1

            # Check for mixed content
            content_types = {elem.content_type for elem in page.elements}
            if len(content_types) > 1:
                stats["processing_breakdown"]["mixed_pages"] += 1
                stats["quality_indicators"]["pages_with_mixed_content"] += 1

            # Content analysis
            stats["content_analysis"]["total_sections"] += len(page.sections)
            if page.sections:
                stats["quality_indicators"]["pages_with_sections"] += 1

            total_elements += len(page.elements)

            for element in page.elements:
                if element.content_type == ContentType.TEXT:
                    stats["content_analysis"]["text_blocks"] += 1
                elif element.content_type == ContentType.TABLE:
                    stats["content_analysis"]["tables"] += 1
                elif element.content_type == ContentType.IMAGE:
                    stats["content_analysis"]["images"] += 1

                # Extraction method analysis
                method = element.metadata.get("extraction_method", "unknown")
                if method == "native":
                    stats["extraction_methods"]["native_extraction"] += 1
                elif method == "vision_llm":
                    stats["extraction_methods"]["vision_llm"] += 1
                else:
                    stats["extraction_methods"]["hybrid"] += 1

        # Calculate averages
        if doc_structure.pages:
            stats["quality_indicators"]["average_elements_per_page"] = \
                total_elements / len(doc_structure.pages)

        return stats


# Example usage and utility functions
def demo_pdf_processing():
    """Demonstration of the PDF processing pipeline"""

    # Initialize pipeline
    pipeline = PDFProcessingPipeline()

    # Example processing
    pdf_path = "example_document.pdf"
    document_type = "generic"  # or "shareholder_momentum", "term_sheet"

    try:
        # Process PDF
        doc_structure, chunks = pipeline.process_pdf(pdf_path, document_type)

        # Get metadata summary
        metadata_summary = pipeline.extract_metadata_summary(doc_structure)
        print("Document Metadata Summary:")
        print(json.dumps(metadata_summary, indent=2))

        # Validate chunks
        validation_results = pipeline.validate_chunks(chunks)
        print(f"\nChunk Validation Results:")
        print(f"Total chunks: {validation_results['total_chunks']}")
        print(f"Quality score: {validation_results['quality_score']:.2f}")
        print(f"Average chunk size: {validation_results['average_chunk_size']:.0f} characters")

        if validation_results['warnings']:
            print("Warnings:")
            for warning in validation_results['warnings'][:5]:  # Show first 5 warnings
                print(f"  - {warning}")

        # Export chunks
        pipeline.export_chunks_to_json(chunks, "processed_chunks.json")

        # Get processing statistics
        stats = pipeline.get_processing_statistics(doc_structure)
        print(f"\nProcessing Statistics:")
        print(f"Document type: {stats['document_info']['type']}")
        print(f"Pages processed: {stats['document_info']['pages']}")
        print(f"Scanned pages: {stats['processing_breakdown']['scanned_pages']}")
        print(f"Content elements: {stats['content_analysis']['text_blocks']} text, "
              f"{stats['content_analysis']['tables']} tables, "
              f"{stats['content_analysis']['images']} images")

        return doc_structure, chunks

    except Exception as e:
        logger.error(f"Demo processing failed: {e}")
        return None, None


def create_custom_parser_example():
    """Example of how to create a custom parser for a new document type"""

    class CustomReportParser(PDFParsingStrategy):
        """Example custom parser for specific report type"""

        def __init__(self, vision_llm: VisionLLMInterface):
            super().__init__(vision_llm)
            self.custom_keywords = ["executive summary", "key findings", "recommendations"]

        def parse_document(self, pdf_path: str) -> DocumentStructure:
            # Start with generic parsing
            generic_parser = GenericPDFParser(self.vision_llm)
            doc_structure = generic_parser.parse_document(pdf_path)
            doc_structure.document_type = "custom_report"

            # Apply custom processing
            self._identify_custom_sections(doc_structure)
            return doc_structure

        def _identify_custom_sections(self, doc_structure: DocumentStructure):
            """Custom section identification logic"""
            for page in doc_structure.pages:
                for section in page.sections:
                    section_text = " ".join(elem.content.lower() for elem in section["elements"])

                    for keyword in self.custom_keywords:
                        if keyword in section_text:
                            section["custom_section"] = keyword.replace(" ", "_")
                            section["priority"] = "high" if keyword == "executive summary" else "medium"
                            break

        def identify_sections(self, page_layout: PageLayout) -> List[Dict[str, Any]]:
            generic_parser = GenericPDFParser(self.vision_llm)
            return generic_parser.identify_sections(page_layout)

        def should_merge_elements(self, elem1: ContentElement, elem2: ContentElement) -> bool:
            generic_parser = GenericPDFParser(self.vision_llm)
            return generic_parser.should_merge_elements(elem1, elem2)

    # Register the custom parser
    def register_custom_parser():
        """Register custom parser in factory"""
        # You would extend PDFParserFactory to include the new parser
        # This is just a demonstration of the pattern
        pass

    return CustomReportParser


if __name__ == "__main__":
    # Run demonstration
    demo_pdf_processing()