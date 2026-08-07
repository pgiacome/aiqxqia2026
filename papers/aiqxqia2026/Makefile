# The CEUR class needs ccicons and elsarticle-num-names, which TeX Live 2023 on this
# machine does not ship. They are installed under $HOME/texmf, so every rule exports
# TEXMFHOME rather than relying on the ambient environment.
export TEXMFHOME := $(HOME)/texmf

.PHONY: all clean watch pages
all: main.pdf

main.pdf: main.tex refs.bib numbers.tex $(wildcard sections/*.tex) $(wildcard figures/*.pdf)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

pages: main.pdf
	@pdfinfo main.pdf | grep Pages

clean:
	latexmk -C
	rm -f *.bbl *.blg probe.*

watch:
	latexmk -pdf -pvc main.tex
