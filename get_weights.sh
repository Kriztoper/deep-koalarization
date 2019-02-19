#!/bin/bash

for i in {a..f}; do
	for j in {0..9}; do
		python -m dataset.weights -f $i$j*;
	done
	for k in {a..f}; do
		python -m dataset.weights -f $i$k*;
	done;
done
