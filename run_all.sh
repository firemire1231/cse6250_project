#!/bin/bash


python 01-Extract-Timeseries-Features.py
python 02-Select-SubClinicalNotes.py
python 03-Preprocess-Clinical-Notes.py
python 04-Apply-med7-on-Clinical-Notes.py
python 05-Represent-Entities-With-Different-Embeddings.py
python 06-Create-Timeseries-Data.py
python 07-TimeseriesBaseline.py
python 08-Multimodal-Baseline.py
python 09-Proposed-Model.py