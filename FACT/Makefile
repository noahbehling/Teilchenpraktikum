ifeq (,$(shell sh -c 'cygpath --version 2> /dev/null'))
  # Unix
  pwd := $$(pwd)
  translate = $1
else
  # Windows mit MSys2/Cygwin
  pwd := $$(cygpath -m "$$(pwd)")
  translate = $(shell echo '$1' | sed 's/:/;/g')
endif

# Farben/ Textformatierung
BACKGR=`tput setaf 0`
GREEN=`tput setaf 10`
GREENB=`tput setab 10`
BOLD=`tput bold`
RESET=`tput sgr0`


all: Protokoll.pdf

# hier Python-Skripte:
plots: plots/plot.py matplotlibrc header-matplotlib.tex | data
				@echo 'Python-Plots:'
				TEXINPUTS="$(call translate,$(pwd):)" python plots/plot.py
				@echo

again:
				@make clean
				@make all

clean:
				@rm -rf build
				@echo ${GREEN}${BOLD}Removing build folder${RESET}


build:
				mkdir -p build

data:
				mkdir -p data

graphics:
				mkdir -p graphics
#
# tables:
# 	mkdir -p tables

plots/plot.pdf: plots/plot.py matplotlibrc header-matplotlib.tex data
				@echo 'Python-Plots:'
				TEXINPUTS="$(call translate,$(pwd):)" python plots/plot.py
				@echo

.DELETE_ON_ERROR:
# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
Protokoll.pdf: main.tex content/* header.tex lit.bib programme.bib plots/plot.pdf graphics | build
				@TEXINPUTS="$(call translate,build:)" lualatex \
								--output-directory=build \
								--interaction=nonstopmode \
								--halt-on-error \
				main.tex|grep -B 12 -e'no output PDF file produced' -e'LuaTeX' --color=auto
				@echo
				@BIBINPUTS=build: biber build/main.bcf|grep -i -e'biber' -e'error' -e'errors' -e'warn' -e'warnings' --color=auto
				@echo
				@lualatex \
								--output-directory=build \
								--interaction=nonstopmode \
								--halt-on-error \
				main.tex>/dev/null
				@lualatex \
								--output-directory=build \
								--interaction=nonstopmode \
								--halt-on-error \
				main.tex|grep  -e'reference' -e'LuaTeX' --color=auto
				@mv build/main.pdf Protokoll.pdf
				@make clean
				@echo ${GREENB}${BACKGR}Success!${RESET}
