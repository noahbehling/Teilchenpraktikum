#!/bin/bash
for filename in /net/e4-nfs-home.e4.physik.tu-dortmund.de/home/zprime/E4/Final/samples/*; do
	./runSelection.exe $filename;
done
