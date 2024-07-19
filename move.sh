#!/bin/bash 

# Folders to move result files between
to=0
from=1

for i in {0..1406}
do
  if ! test -f ./$to/$i; then
    if test -f ./$from/$i; then
      cp ./$from/$i ./$to/
    fi
  fi
done
