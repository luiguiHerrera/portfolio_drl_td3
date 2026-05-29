PDF = main.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): main.tex references.bib
	tectonic main.tex

clean:
	rm -f main.pdf
