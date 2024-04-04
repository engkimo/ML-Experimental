# !/bin/sh
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
find . -name '.DS_Store' -type f -ls -delete
rm -r tiny_utils/checkpoints tiny_utils/log
rm -rf log __pycache__ */__pycache__ tool/tv_reference/__pycache__ checkpoints  INTERRUPTED.pth  predictions.jpg
