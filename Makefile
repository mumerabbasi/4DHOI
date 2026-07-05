# LuaLaTeX thesis build, mirroring the resume project's direct build style.

TEXFILE=main.tex
JOBNAME=Thesis
PDF=$(JOBNAME).pdf
LATEX_ENV=env LANG=C.UTF-8 LC_ALL=C.UTF-8 LANGUAGE=
LATEX_FLAGS=-jobname=$(JOBNAME) -synctex=1 -interaction=nonstopmode -file-line-error
BUILD_DIR=build
GS=gs
PDF_COMPRESS_FLAGS=-sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dNOPAUSE -dQUIET -dBATCH -dDetectDuplicateImages=true -dCompressFonts=true -dSubsetFonts=true -dDownsampleColorImages=false -dDownsampleGrayImages=false -dDownsampleMonoImages=false -dAutoFilterColorImages=false -dAutoFilterGrayImages=false -dColorImageFilter=/FlateEncode -dGrayImageFilter=/FlateEncode

.PHONY: all quick compress-pdf clean distclean FORCE

all: $(PDF)

$(PDF): FORCE $(TEXFILE) metadata.tex styles/thesis.sty chapters/*.tex frontmatter/*.tex backmatter/*.tex references/references.bib figures/TUM_Logo.png
	$(LATEX_ENV) lualatex $(LATEX_FLAGS) $(TEXFILE)
	-bibtex $(JOBNAME)
	$(LATEX_ENV) lualatex $(LATEX_FLAGS) $(TEXFILE)
	$(LATEX_ENV) lualatex $(LATEX_FLAGS) $(TEXFILE)
	$(MAKE) --no-print-directory compress-pdf

quick: FORCE
	$(LATEX_ENV) lualatex $(LATEX_FLAGS) $(TEXFILE)
	$(MAKE) --no-print-directory compress-pdf

compress-pdf:
	@if [ -f "$(PDF)" ]; then \
		if command -v $(GS) >/dev/null 2>&1; then \
			tmp="$(JOBNAME).uncompressed.pdf"; \
			mv "$(PDF)" "$$tmp"; \
			if $(GS) $(PDF_COMPRESS_FLAGS) -sOutputFile="$(PDF)" "$$tmp"; then \
				rm -f "$$tmp"; \
			else \
				mv "$$tmp" "$(PDF)"; \
				exit 1; \
			fi; \
		else \
			echo "Ghostscript not found; leaving $(PDF) uncompressed."; \
		fi; \
	fi

FORCE:

clean:
	@rm -f $(JOBNAME).aux $(JOBNAME).bbl $(JOBNAME).blg $(JOBNAME).lof $(JOBNAME).log $(JOBNAME).lot $(JOBNAME).out $(JOBNAME).pdf $(JOBNAME).uncompressed.pdf $(JOBNAME).synctex.gz $(JOBNAME).toc $(JOBNAME).fls $(JOBNAME).fdb_latexmk
	@rm -f main.aux main.bbl main.blg main.lof main.log main.lot main.out main.synctex.gz main.toc
	@rm -f chapters/*.aux frontmatter/*.aux backmatter/*.aux

distclean: clean
	@rm -rf $(BUILD_DIR)/*
