#!/bin/bash

for i in {0..9}; do
	for j in {0..9}; do
		python -m dataset.weights -f $i$j*;
	done
	for k in {a..f}; do
		python -m dataset.weights -f $i$k*;
	done;
done

for i in {a..c}; do
	for j in {0..9}; do
		python -m dataset.weights -f $i$j*;
	done
	for k in {a..f}; do
		python -m dataset.weights -f $i$k*;
	done;
done

for i in {0..9}; do
	python -m dataset.weights -f da$i*;
done

for i in {a..f}; do
	python -m dataset.weights -f da$i*;
done

for i in {0..9}; do
	python -m dataset.weights -f d$i*;
done

for i in {b..f}; do
	python -m dataset.weights -f d$i*;
done

for i in {e..f}; do
	for j in {0..9}; do
		python -m dataset.weights -f $i$j*;
	done
	for k in {a..f}; do
		python -m dataset.weights -f $i$k*;
	done;
done
