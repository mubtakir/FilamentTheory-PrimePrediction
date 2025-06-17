# Makefile for Filament Theory Research Paper
# Dr. Basel Yahya Abdullah

# Main document
MAIN = research_paper_latex
EQUATIONS = research_paper_equations

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Output directory
OUTDIR = output

# Default target
all: $(MAIN).pdf

# Create output directory
$(OUTDIR):
	mkdir -p $(OUTDIR)

# Compile main document
$(MAIN).pdf: $(MAIN).tex $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	$(BIBTEX) $(OUTDIR)/$(MAIN)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	cp $(OUTDIR)/$(MAIN).pdf .

# Compile equations appendix
equations: $(EQUATIONS).tex $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) $(EQUATIONS).tex
	cp $(OUTDIR)/$(EQUATIONS).pdf .

# Clean auxiliary files
clean:
	rm -rf $(OUTDIR)
	rm -f *.aux *.log *.bbl *.blg *.toc *.out *.fdb_latexmk *.fls

# Clean everything including PDFs
distclean: clean
	rm -f $(MAIN).pdf $(EQUATIONS).pdf

# Quick compile (single pass)
quick: $(MAIN).tex $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	cp $(OUTDIR)/$(MAIN).pdf .

# View PDF
view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || echo "Please open $(MAIN).pdf manually"

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Compile main document (default)"
	@echo "  equations - Compile equations appendix"
	@echo "  quick     - Quick compile (single pass)"
	@echo "  clean     - Remove auxiliary files"
	@echo "  distclean - Remove all generated files"
	@echo "  view      - Open PDF viewer"
	@echo "  help      - Show this help"

.PHONY: all equations clean distclean quick view help
