"""
Agent 4: Document Intelligence Agent.

The lecturer's brief framing:
  "Reads sustainability reports, news articles, NGO documents, regulatory
   filings. Extracts structured information, identifies key claims, cites
   sources. Uses LLM-based extraction."

This agent processes unstructured PDFs using Gemini 2.5 Flash and outputs
structured DocumentExtraction objects.

Methodology:
1. Upload each PDF to Gemini File API
2. Send a structured extraction prompt requesting JSON output
3. Parse the response into a Pydantic DocumentExtraction object
4. Cache the result to outputs/cache/document_extractions/{company_id}.json
   to avoid re-running expensive Gemini calls

The output feeds Agent 8 (Greenwashing) which compares these CLAIMS
against the company's DISCLOSED numerical data.

Owner: Role D
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import re

from agents.base import BaseAgent
from schemas.document_extraction import DocumentExtraction, ExtractedClaim


# === Configuration ===
GEMINI_MODEL = "gemini-2.5-flash"
CACHE_DIR = Path("outputs/cache/document_extractions")
GEMINI_RATE_LIMIT_DELAY_SECONDS = 2  # Polite delay between calls


# === Extraction prompt ===
# Sent to Gemini for each document. Structured to produce JSON output.

EXTRACTION_PROMPT = """You are a sustainability analyst extracting structured information from a corporate sustainability report.

Read the attached document carefully and extract the following information.
Return your response as JSON ONLY, with no commentary before or after the JSON.

Required JSON structure:
{
  "company_name": "the company's name as it appears in the document",
  "document_year": <integer - the fiscal year the document describes, e.g. 2024>,
  "tnfd_adopter": <boolean - does the company explicitly adopt or align with TNFD framework?>,
  "no_deforestation_pledge": <boolean - does the company commit to no-deforestation in supply chain?>,
  "net_zero_year": <integer or null - the year by which the company commits to net-zero emissions>,
  "sbti_status": "validated" | "committed" | "none" | "unknown",
  "biodiversity_target_year": <integer or null - year for biodiversity-specific targets>,
  "water_stress_disclosed": <boolean - does the company disclose water stress exposure?>,
  "forest_risk_commodities": ["palm oil", "soy", "beef", "cocoa", "timber", "leather", ...],
  
  "biodiversity_commitments": [
    {
      "text": "verbatim quote of the commitment",
      "category": "biodiversity",
      "target_year": <integer or null>,
      "is_quantitative": <boolean>,
      "source_page": <integer or null>
    }
  ],
  
  "climate_targets": [...same structure, category: "climate"...],
  "water_disclosures": [...same structure, category: "water"...],
  "supply_chain_claims": [...same structure, category: "supply_chain"...],
  
  "top_sustainability_claims": [
    {
      "text": "the 3-5 main claims the company emphasises in the document",
      "category": "biodiversity" | "climate" | "water" | "supply_chain" | "other",
      "target_year": <integer or null>,
      "is_quantitative": <boolean>,
      "source_page": <integer or null>
    }
  ],
  
  "document_summary": "2-3 sentence summary of the document's key sustainability themes",
  
  "extraction_confidence": "high" | "medium" | "low",
  "extraction_notes": "any caveats — e.g. vague language, missing sections, document focused on one area"
}

Guidelines:
- For boolean fields: if the document does not explicitly address the topic, return false (not null)
- For target_year fields: extract the specific year mentioned. If multiple targets exist, use the nearest one
- For is_quantitative: true only if a specific number is attached (e.g. "reduce by 50%", "1000 hectares restored")
- For source_page: include only if you can identify the specific page; null otherwise
- For text fields: prefer verbatim quotes from the document where possible
- For forest_risk_commodities: list only commodities the company actually sources or trades
- For sbti_status: "validated" only if the company explicitly states SBTi-validated targets; "committed" if working toward validation; "none" otherwise
- Be honest about gaps. If a document focuses heavily on climate and barely mentions biodiversity, say so in extraction_notes and mark extraction_confidence as "medium" or "low" for biodiversity

Return ONLY the JSON object. No markdown code fences, no preamble, no explanation."""


class DocumentIntelligenceAgent(BaseAgent):
    """Extracts structured info from sustainability documents using Gemini.

    Inputs: PDF documents (typically 10 company sustainability reports)
    Outputs: List of DocumentExtraction objects
    """

    name = "document_intelligence"

    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        """
        Args:
            api_key: Gemini API key. If None, reads from GOOGLE_API_KEY env var.
            use_cache: If True, returns cached extractions instead of re-calling Gemini.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set GOOGLE_API_KEY environment "
                "variable or pass api_key=..."
            )
        self.use_cache = use_cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        documents: List[Dict[str, str]],
    ) -> List[DocumentExtraction]:
        """Extract structured info from a list of documents.

        Args:
            documents: List of dicts, each with keys:
                - company_id: e.g. "C00071"
                - company_name: e.g. "ASML Holding NV"
                - path: Path to the PDF file

        Returns:
            List of DocumentExtraction objects, one per document.
        """
        self.log(
            decision_type="document_intelligence_start",
            details={
                "n_documents": len(documents),
                "model": GEMINI_MODEL,
                "cache_enabled": self.use_cache,
                "cache_dir": str(CACHE_DIR),
            },
            confidence="judgement_based",
            notes=(
                f"Processing {len(documents)} sustainability documents via "
                f"Gemini {GEMINI_MODEL}. Cached results used where available."
            ),
        )

        extractions = []
        for i, doc in enumerate(documents):
            try:
                extraction = self._process_one_document(
                    company_id=doc["company_id"],
                    company_name=doc["company_name"],
                    pdf_path=Path(doc["path"]),
                )
                extractions.append(extraction)
                # Polite delay to avoid rate limits
                if i < len(documents) - 1:
                    time.sleep(GEMINI_RATE_LIMIT_DELAY_SECONDS)
            except Exception as e:
                self.log(
                    decision_type="document_extraction_failed",
                    company_id=doc["company_id"],
                    details={"error": str(e), "path": doc["path"]},
                    confidence="observed",
                    notes=f"Failed to extract from {doc['company_name']}: {e}",
                )
                # Don't stop the pipeline — continue with the next document

        self.log(
            decision_type="document_intelligence_complete",
            details={
                "n_documents_input": len(documents),
                "n_extractions_successful": len(extractions),
                "n_failures": len(documents) - len(extractions),
            },
            confidence="observed",
        )

        return extractions

    def _process_one_document(
        self,
        company_id: str,
        company_name: str,
        pdf_path: Path,
    ) -> DocumentExtraction:
        """Process a single document — cache check, Gemini call, parse."""
        cache_path = CACHE_DIR / f"{company_id}.json"

        # === Cache check ===
        if self.use_cache and cache_path.exists():
            self.log(
                decision_type="extraction_cache_hit",
                company_id=company_id,
                details={"cache_path": str(cache_path)},
                confidence="reported",
            )
            with open(cache_path) as f:
                return DocumentExtraction(**json.load(f))

        # === Gemini call ===
        self.log(
            decision_type="extraction_start",
            company_id=company_id,
            details={
                "company_name": company_name,
                "pdf_path": str(pdf_path),
                "pdf_size_mb": round(pdf_path.stat().st_size / 1_000_000, 1),
            },
            confidence="reported",
        )

        raw_response = self._call_gemini(pdf_path)
        parsed = self._parse_json_response(raw_response)

        # === Build DocumentExtraction ===
        extraction = self._build_extraction(
            company_id=company_id,
            company_name=company_name,
            pdf_path=pdf_path,
            parsed=parsed,
        )

        # === Cache the result ===
        try:
            with open(cache_path, "w") as f:
                f.write(extraction.model_dump_json(indent=2))
            self.log(
                decision_type="extraction_cached",
                company_id=company_id,
                details={"cache_path": str(cache_path)},
            )
        except Exception as e:
            # Cache write failure isn't fatal
            self.log(
                decision_type="cache_write_failed",
                company_id=company_id,
                details={"error": str(e)},
            )

        return extraction

    def _call_gemini(self, pdf_path: Path) -> str:
        """Upload the PDF to Gemini and request extraction.

        Returns the raw text response.
        """
        from google import genai

        client = genai.Client(api_key=self.api_key)

        # Upload the PDF — Gemini Files API
        uploaded_file = client.files.upload(file=str(pdf_path))

        # Generate extraction
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[uploaded_file, EXTRACTION_PROMPT],
        )

        # Clean up uploaded file (be a good API citizen)
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass  # Not critical if cleanup fails

        return response.text

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """Parse Gemini's JSON response, handling common formatting issues.

        Gemini commonly wraps JSON in markdown code fences (```json ... ```)
        despite instructions, sometimes with trailing whitespace or content.
        We use a layered approach: try direct parse, then fence-strip, then
        regex-extract the outermost JSON object.
        """
        # Layer 1: direct parse (clean JSON case)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Layer 2: strip markdown fences using regex (more robust than .startswith/.endswith)
        # Matches: ```json\n{...}\n``` OR ```\n{...}\n``` OR just {...}
        fence_pattern = re.compile(
            r"^\s*```(?:json|JSON)?\s*\n(.*?)\n```\s*$",
            re.DOTALL,
        )
        match = fence_pattern.match(raw)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Layer 3: find the outermost JSON object using brace counting.
        # This handles cases where there's text before/after AND inside braces.
        try:
            start = raw.index("{")
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(raw)):
                ch = raw[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = raw[start : i + 1]
                        return json.loads(candidate)
        except (ValueError, json.JSONDecodeError):
            pass

        # Layer 4: regex fallback (most permissive — may fail on nested braces in strings)
        regex_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if regex_match:
            try:
                return json.loads(regex_match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse Gemini response as JSON. "
                    f"First 500 chars: {raw[:500]}"
                ) from e

        raise ValueError(
            f"Failed to parse Gemini response as JSON. "
            f"First 500 chars: {raw[:500]}"
        )

    def _build_extraction(
        self,
        company_id: str,
        company_name: str,
        pdf_path: Path,
        parsed: Dict[str, Any],
    ) -> DocumentExtraction:
        """Build a DocumentExtraction Pydantic object from parsed JSON."""
        def make_claims(items: List[Dict]) -> List[ExtractedClaim]:
            claims = []
            for item in items or []:
                try:
                    claims.append(ExtractedClaim(
                        text=item.get("text", ""),
                        category=item.get("category", "other"),
                        target_year=item.get("target_year"),
                        is_quantitative=bool(item.get("is_quantitative", False)),
                        source_page=item.get("source_page"),
                    ))
                except Exception:
                    continue
            return claims

        return DocumentExtraction(
            company_id=company_id,
            company_name=parsed.get("company_name", company_name),
            document_path=str(pdf_path),
            document_year=parsed.get("document_year"),
            extraction_date=datetime.utcnow(),
            # Biodiversity
            biodiversity_commitments=make_claims(parsed.get("biodiversity_commitments", [])),
            tnfd_adopter=bool(parsed.get("tnfd_adopter", False)),
            no_deforestation_pledge=bool(parsed.get("no_deforestation_pledge", False)),
            biodiversity_target_year=parsed.get("biodiversity_target_year"),
            # Climate
            climate_targets=make_claims(parsed.get("climate_targets", [])),
            net_zero_year=parsed.get("net_zero_year"),
            sbti_status=parsed.get("sbti_status", "unknown"),
            # Water
            water_disclosures=make_claims(parsed.get("water_disclosures", [])),
            water_stress_disclosed=bool(parsed.get("water_stress_disclosed", False)),
            # Supply chain
            forest_risk_commodities_mentioned=parsed.get("forest_risk_commodities", []),
            supply_chain_claims=make_claims(parsed.get("supply_chain_claims", [])),
            # Overall
            top_sustainability_claims=make_claims(parsed.get("top_sustainability_claims", [])),
            document_summary=parsed.get("document_summary", ""),
            extraction_confidence=parsed.get("extraction_confidence", "medium"),
            extraction_notes=parsed.get("extraction_notes"),
        )
