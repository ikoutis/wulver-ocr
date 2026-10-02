# samples — inputs for the smoke run ([O-002])

Two public test pages, so the first run on Wulver needs nothing but a
`git pull`. Both are test fixtures redistributed by Apache-2.0 OCR projects;
the content itself is third-party (an IEEE paper, a scanned book page) and
is here for testing only.

| file | what | why | source |
|---|---|---|---|
| `stereo_matching_ieee2013.pdf` | Kowalczuk, Psota, Pérez, *Real-time Temporal Stereo Matching using Iterative Adaptive Support Weights*, IEEE 2013; 6 pages, born-digital | the clean case: two columns, nine numbered display equations (`cases`, `argmin`), a geometry figure, tables | MinerU `demo/pdfs/demo2.pdf` |
| `old_book_scan.pdf` | one page of a 19th-century book, yellowed, blurred, slightly skewed, with footnotes | the worst-input case: no math, but the kind of scan the readers claim to handle | olmOCR `tests/gnarly_pdfs/horribleocr.pdf` |

A third input, handwritten with graph drawings (an exam, lecture notes), is
added on Wulver by hand: nothing public that could be reached from the
development sandbox was a real page rather than a model's demo screenshot.

Run them (from the repo root, on Wulver):

```bash
export INPUTS="$PWD/samples /project/ikoutis/$USER/wocr/o002_in"   # + your own folder
export OUT=/project/ikoutis/$USER/wocr/runs/o002_smoke
sbatch --array=0-0 slurm/ocr.sbatch
```
