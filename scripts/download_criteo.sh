#!/bin/bash

for i in {0..8}; do
curl -O http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_$i.gz
done