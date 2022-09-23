#!/bin/bash
for filename in output_runSelection/*; do
	./plotDistribution.exe $filename;
done
