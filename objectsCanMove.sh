#!/bin/bash
python initialDetection.py
python computeMatches.py
python computeConsistentTransformations.py
python prepareVariablesForCutPursuit.py
python SolveCutPursuit.py
