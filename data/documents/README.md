# Sustainability Documents

This folder is **intentionally empty in the repository** — sustainability reports are
gitignored to avoid storing copyrighted PDFs and bloating the repo.

To reproduce the document extraction step, download these 10 reports from each
company's investor-relations website:

| Company | File name | Source |
|---|---|---|
| Iberdrola | `iberdrola_sustainability_2025.pdf` | iberdrola.com |
| Schneider Electric | `schneider_sustainability_2025.pdf` | se.com |
| Sanofi | `sanofi_sustainability_2025.pdf` | sanofi.com |
| ASML | `asml_sustainability_2025.pdf` | asml.com |
| Bayer | `bayer_sustainability_2025.pdf` | bayer.com |
| L'Oréal | `loreal_sustainability_2025.pdf` | loreal.com |
| TotalEnergies | `totalenergies_sustainability_2025.pdf` | totalenergies.com |
| Unilever | `unilever_sustainability_2025.pdf` | unilever.com |
| AB InBev | `abinbev_sustainability_2025.pdf` | ab-inbev.com |
| Allianz | `allianz_sustainability_2025.pdf` | allianz.com |

Cached extraction outputs are committed in `outputs/cache/document_extractions/`,
so re-running the pipeline does NOT require re-downloading the PDFs unless you
want to verify extraction quality or test with different prompts.
