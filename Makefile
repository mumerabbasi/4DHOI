# LuaLaTeX thesis build, mirroring the resume project's direct build style.

TEXFILE=main.tex
JOBNAME=Thesis
PDF=$(JOBNAME).pdf
LATEX_ENV=env LANG=C.UTF-8 LC_ALL=C.UTF-8 LANGUAGE=
LATEX_FLAGS=-jobname=$(JOBNAME) -synctex=1 -interaction=nonstopmode -file-line-error
BUILD_DIR=build

.PHONY: all quick clean distclean FORCE

all: $(PDF)

$(PDF): FORCE $(TEXFILE) metadata.tex styles/thesis.sty chapters/*.tex frontmatter/*.tex backmatter/*.tex references/references.bib figures/TUM_Logo.png
	$(LATEX_ENV) lualatex $(LATEX_FLAGS) $(TEXFILE)
	-bibtex $(JOBNAME)
	$(LATEX_ENV) lualatex $(LATEX_FLAGS) $(TEXFILE)
	$(LATEX_ENV) lualatex $(LATEX_FLAGS) $(TEXFILE)

quick: FORCE
	$(LATEX_ENV) lualatex $(LATEX_FLAGS) $(TEXFILE)

FORCE:

clean:
	@rm -f $(JOBNAME).aux $(JOBNAME).bbl $(JOBNAME).blg $(JOBNAME).lof $(JOBNAME).log $(JOBNAME).lot $(JOBNAME).out $(JOBNAME).pdf $(JOBNAME).synctex.gz $(JOBNAME).toc $(JOBNAME).fls $(JOBNAME).fdb_latexmk
	@rm -f main.aux main.bbl main.blg main.lof main.log main.lot main.out main.synctex.gz main.toc
	@rm -f chapters/*.aux frontmatter/*.aux backmatter/*.aux

distclean: clean
	@rm -rf $(BUILD_DIR)/*
